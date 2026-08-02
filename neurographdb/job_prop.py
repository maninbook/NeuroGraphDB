# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11", "numpy", "pyarrow",
#     "transformers>=4.40", "torch",
#     "spacy>=3.7", "huggingface_hub>=0.28",
# ]
# ///
"""P — 명제·기호학 표현층 절제 실험. 사전등록은 ../PROPOSITION.md.

조건 네 개를 **같은 엣지 집합** 위에서 돌린다. 타입과 극성만 켜고 끈다.

    P0 통제군   무타입·무극성. 확산 제약 없음 → 기존 코드와 비트 단위로 같아야 한다
    P1 타입만   술어 타입 + 질의 술어 제약(alpha)
    P2 극성만   부정 엣지 감쇠(beta)
    P3 둘 다

절제를 셋으로 나눈 이유: 뭉쳐 돌리면 이겨도 **뭐가 일했는지** 모른다.

사전등록한 것 (돌린 뒤에 바꾸지 않는다)
    주 가설    MuSiQue 근거2개@10에서 P3 > P0, McNemar α=0.05
    안전장치   HotpotQA R@10이 1.0%p 넘게 떨어지면 주 가설이 통과해도 실패
    기전 예측  이득은 **정답이 dense 상위 20위 밖이던 질문**에 몰려야 한다

인자: DATASET N_QUESTIONS ALPHA BETA SEED
"""

import json
import os
import re
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

CODE_REPO = "goethe0101/neurographdb"
RESULTS_REPO = "goethe0101/neurographdb-results"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
N_SEEDS = 5
MAXK = 20
KS = (1, 2, 5, 10, 20)

DATASET = sys.argv[1] if len(sys.argv) > 1 else "musique"
N_Q = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3   # 사전등록 값
BETA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2    # 사전등록 값
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0

PARQUET_REV = "refs/convert/parquet"
DATASETS = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor/validation/0000.parquet"),
    "2wiki":    ("framolfese/2WikiMultihopQA", "default/validation/0000.parquet"),
    "musique":  ("dgslibisey/MuSiQue", "default/validation/0000.parquet"),
}


def build_core():
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(CODE_REPO, repo_type="dataset", allow_patterns="src/**"))
    src = root / "src"
    out = Path("/tmp/ngdb"); out.mkdir(exist_ok=True)
    (out / "__init__.py").write_text(
        "from ._ngdb_core import BM25, DenseIndex, Graph\n"
        '__all__ = ["BM25", "DenseIndex", "Graph"]\n')
    inc = subprocess.run([sys.executable, "-m", "pybind11", "--includes"],
                         capture_output=True, text=True, check=True).stdout.split()
    ext = sysconfig.get_config_var("EXT_SUFFIX")
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-fPIC", "-shared",
                    *inc, f"-I{src}", "-o", str(out / f"_ngdb_core{ext}"),
                    str(src / "bm25.cpp"), str(src / "dense.cpp"),
                    str(src / "graph.cpp"), str(src / "bindings.cpp")], check=True)
    sys.path.insert(0, "/tmp")


def fetch_propositions_module():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(CODE_REPO, "propositions.py", repo_type="dataset")
    sys.path.insert(0, str(Path(p).parent))


def load_pool(dataset, n_questions, seed):
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    name, path = DATASETS[dataset]
    f = hf_hub_download(name, path, repo_type="dataset", revision=PARQUET_REV)
    ds = pq.read_table(f).to_pylist()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:n_questions]
    pool, questions = {}, []
    for i in idx:
        row = ds[int(i)]
        if dataset == "musique":
            passages = {p["title"]: p["paragraph_text"] for p in row["paragraphs"]}
            gold = sorted({p["title"] for p in row["paragraphs"] if p.get("is_supporting")})
        else:
            passages = {t: " ".join(s) for t, s in
                        zip(row["context"]["title"], row["context"]["sentences"])}
            gold = sorted(set(row["supporting_facts"]["title"]))
        if not gold:
            continue
        for t, x in passages.items():
            pool.setdefault(t, x)
        questions.append({"q": row["question"], "gold": gold,
                          "answer": str(row["answer"])})
    titles = list(pool)
    return titles, [pool[t] for t in titles], questions


def build_mention_edges(titles, texts, max_title_words=6):
    """기존과 **한 글자도 다르지 않게** 유지한다. 엣지가 달라지면 절제가 무의미해진다."""
    norm = lambda s: re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    lookup = {}
    for i, t in enumerate(titles):
        key = " ".join(norm(t).split())
        if len(key) >= 5:
            lookup.setdefault(key, i)
    edges = []
    for i, body in enumerate(texts):
        toks = norm(body).split()
        hit = set()
        for n in range(1, max_title_words + 1):
            for a in range(len(toks) - n + 1):
                j = lookup.get(" ".join(toks[a:a + n]))
                if j is not None and j != i:
                    hit.add(j)
        edges.extend((i, j, 1.0) for j in hit)
    return edges


def mcnemar(pairs):
    from math import comb
    b = sum(1 for a, x in pairs if not a and x)
    c = sum(1 for a, x in pairs if a and not x)
    n = b + c
    if n == 0:
        return {"gain": 0, "loss": 0, "n_discordant": 0, "p_value": 1.0}
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return {"gain": b, "loss": c, "n_discordant": n, "p_value": min(1.0, 2 * tail)}


def main():
    import numpy as np
    t0 = time.time()
    build_core()
    fetch_propositions_module()
    from ngdb import DenseIndex, Graph
    import propositions as prop

    titles, texts, questions = load_pool(DATASET, N_Q, SEED)
    edges = build_mention_edges(titles, texts)
    print(f"[{time.time()-t0:6.1f}s] {DATASET}: 질문 {len(questions)} | 풀 {len(titles)} "
          f"| 엣지 {len(edges):,}")

    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                   check=True, stdout=subprocess.DEVNULL)
    import spacy
    nlp = spacy.load("en_core_web_sm")
    typed, vocab, pstats = prop.annotate_edges(titles, texts, edges, nlp)
    assert len(typed) == len(edges), "엣지 개수가 변했다 — 절제가 성립하지 않는다"
    print(f"[{time.time()-t0:6.1f}s] 명제 부착: 타입있음 {pstats['n_typed']:,}"
          f"/{pstats['n_edges']:,} ({pstats['coverage']:.1%}) | "
          f"부정 {pstats['n_negative']:,} | 술어어휘 {pstats['n_vocab']}")
    print(f"  상위 술어: {', '.join(f'{p}({c})' for p, c in pstats['top_predicates'][:8])}")
    if pstats["coverage"] < 0.10:
        print("  ! 타입 부착률이 너무 낮다. P1/P3가 통제군과 거의 같아질 것이다")

    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    etok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    emodel = AutoModel.from_pretrained(EMBED_MODEL).to(dev).eval()

    def encode(items, batch=128):
        out = []
        for a in range(0, len(items), batch):
            b = etok(items[a:a + batch], padding=True, truncation=True,
                     max_length=512, return_tensors="pt").to(dev)
            with torch.no_grad():
                h = emodel(**b).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    P = encode([f"{t}. {x}" for t, x in zip(titles, texts)])
    dense = DenseIndex(P.shape[1]); dense.add_batch(P)
    Q = encode([QUERY_PREFIX + q["q"] for q in questions])
    dense_hits = [dense.search(Q[i], MAXK) for i in range(len(questions))]
    print(f"[{time.time()-t0:6.1f}s] 벡터 색인 {dense.size}")

    # dense 단독을 먼저 재현한다. 통제군이 어긋날 때 원인이 임베딩인지 그래프층인지
    # 이걸로 갈린다. G1 기록: hotpotqa 0.857 / 2wiki 0.486 / musique 0.266
    d_all10 = float(np.mean([
        float(all(g in [titles[d] for d, _ in dense_hits[i]][:10] for g in q["gold"]))
        for i, q in enumerate(questions)]))
    d_expect = {"hotpotqa": 0.857, "2wiki": 0.486, "musique": 0.266}[DATASET]
    print(f"  dense 단독 근거2개@10 = {d_all10:.3f} (G1 기록 {d_expect:.3f}, "
          f"차이 {abs(d_all10-d_expect):.3f})")

    qtypes = [prop.question_types(q["q"], vocab, nlp) for q in questions]
    n_qt = sum(1 for x in qtypes if x)
    print(f"  질문 술어 추출: {n_qt}/{len(questions)} ({n_qt/len(questions):.1%}) "
          f"— 못 뽑으면 제약 없이 돌아 통제군과 같아진다")

    # 조건별 그래프. 엣지 집합은 넷 다 동일하고 타입/극성만 다르다.
    g_ctrl = Graph(len(titles)); g_ctrl.add_edges(edges)
    g_type = Graph(len(titles)); g_type.add_edges_typed([(s, d, w, t, 1) for s, d, w, t, _ in typed])
    g_pol  = Graph(len(titles)); g_pol.add_edges_typed([(s, d, w, -1, p) for s, d, w, _, p in typed])
    g_both = Graph(len(titles)); g_both.add_edges_typed(typed)

    def retrieve(g, i, qt, alpha, beta):
        hits = dense_hits[i][:N_SEEDS]
        acts = g.spread([d for d, _ in hits], [float(s) for _, s in hits],
                        3, 0.65, 0.02, MAXK, qt, alpha, beta)
        # 중복 제거는 **문서 id로만** 한다. id를 넣고 제목으로 조회하면
        # 영원히 안 맞아서 dense 결과가 통째로 중복 추가되고, 진짜 결과가 밀려난다.
        # (그렇게 해서 통제군이 0.350 대신 0.336으로 나왔다)
        seen, out = set(), []
        for d, _ in acts:
            if d not in seen:
                seen.add(d); out.append(d)
        for d, _ in dense_hits[i]:
            if d not in seen:
                seen.add(d); out.append(d)
        return [titles[d] for d in out]

    conds = {
        "P0_통제군": (g_ctrl, lambda i: [],        1.0,   1.0),
        "P1_타입만": (g_type, lambda i: qtypes[i], ALPHA, 1.0),
        "P2_극성만": (g_pol,  lambda i: [],        1.0,   BETA),
        "P3_둘다":   (g_both, lambda i: qtypes[i], ALPHA, BETA),
    }

    results, per_q = {}, {}
    for name, (g, qtf, a, b) in conds.items():
        rows = []
        for i, q in enumerate(questions):
            ranked = retrieve(g, i, qtf(i), a, b)[:MAXK]
            gold = q["gold"]
            r = {f"r@{k}": sum(1 for gg in gold if gg in ranked[:k]) / len(gold) for k in KS}
            r["all@10"] = float(all(gg in ranked[:10] for gg in gold))
            rows.append(r)
        per_q[name] = rows
        results[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    print(f"\n{'='*78}\n명제층 절제 {DATASET} — alpha={ALPHA} beta={BETA} n={len(questions)}\n{'='*78}")
    print(f"{'조건':<12}" + "".join(f"{'R@'+str(k):>9}" for k in KS) + f"{'근거2개@10':>12}")
    for name, v in results.items():
        print(f"{name:<12}" + "".join(f"{v[f'r@{k}']:>9.3f}" for k in KS)
              + f"{v['all@10']:>12.3f}")

    # 통제군은 G1에서 기록한 값과 같아야 한다. 다르면 절제가 성립하지 않으므로
    # 결과를 해석하기 전에 여기서 잡는다. (실제로 중복제거 버그를 이걸로 발견했다)
    expected = {"hotpotqa": 0.940, "2wiki": 0.907, "musique": 0.350}
    got = results["P0_통제군"]["all@10"]
    gap = abs(got - expected[DATASET])
    print(f"\n통제군 대조: 근거2개@10 = {got:.3f} (G1 기록 {expected[DATASET]:.3f}, 차이 {gap:.3f})")
    if gap > 0.005:
        print("  ! 통제군이 기존 기록과 다르다. 절제 결과를 신뢰할 수 없다")

    base = per_q["P0_통제군"]
    tests = {}
    print()
    for name in ("P1_타입만", "P2_극성만", "P3_둘다"):
        mc = mcnemar([(bool(a["all@10"]), bool(b["all@10"]))
                      for a, b in zip(base, per_q[name])])
        tests[name] = mc
        verdict = ("앞섬" if mc["gain"] > mc["loss"] and mc["p_value"] < 0.05 else
                   "뒤짐" if mc["loss"] > mc["gain"] and mc["p_value"] < 0.05 else
                   "차이 없음")
        print(f"{name} vs 통제군: 이득 {mc['gain']} 손실 {mc['loss']} "
              f"(불일치 {mc['n_discordant']}, p={mc['p_value']:.4f}) → {verdict}")

    # ── 기전 예측 검증 — 이득이 예상한 자리에 몰렸는가 ──────────────────────
    print(f"\n기전 확인 (사전등록: 이득은 정답이 dense 상위20 밖이던 질문에 몰려야 한다)")
    dense_titles = [[titles[d] for d, _ in h] for h in dense_hits]
    inside, outside = [], []
    for i, q in enumerate(questions):
        covered = all(g in dense_titles[i] for g in q["gold"])
        (inside if covered else outside).append(i)
    for name in ("P3_둘다",):
        for label, ids in (("dense가 이미 담음", inside), ("dense 밖 (그래프가 일할 자리)", outside)):
            if not ids:
                continue
            mc = mcnemar([(bool(base[i]["all@10"]), bool(per_q[name][i]["all@10"])) for i in ids])
            print(f"  {label:<28} n={len(ids):>4}  이득 {mc['gain']:>3} 손실 {mc['loss']:>3}")

    # ── 사전등록한 안전장치 ─────────────────────────────────────────────────
    drop = results["P0_통제군"]["r@10"] - results["P3_둘다"]["r@10"]
    if drop > 0.01:
        print(f"\n! 안전장치 위반 — R@10이 {drop*100:.1f}%p 떨어졌다 (한계 1.0%p). 실패로 기록한다")

    payload = {"dataset": DATASET, "n_questions": len(questions), "pool_size": len(titles),
               "alpha": ALPHA, "beta": BETA, "seed": SEED,
               "prop_stats": pstats, "question_type_rate": n_qt / len(questions),
               "results": results, "mcnemar": tests, "runtime_sec": time.time() - t0}
    out = Path(f"/tmp/prop_{DATASET}_a{ALPHA}_b{BETA}.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True, private=True)
    api.upload_file(path_or_fileobj=str(out), path_in_repo=f"runs/{out.name}",
                    repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"\n업로드: runs/{out.name}")


if __name__ == "__main__":
    main()
