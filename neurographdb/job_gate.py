# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11", "numpy", "pyarrow",
#     "transformers>=4.40", "torch", "huggingface_hub>=0.28",
# ]
# ///
"""G — 질의 게이팅(억제). 어텐션처럼 **질문과의 일치도로 확산을 눌러본다.**

아이디어(김기인): 단순 연결보다 억제가 필요하다. 어텐션도 결국 점수를 매기는 일이다.

사전 측정에서 차수 기반 억제 두 형태는 걸렀다 —
출차수는 편차가 없고(최대 5), 허브 억제는 HotpotQA에서 정답을 누른다(정답이 더 허브).
남은 건 어텐션의 본래 기준인 **질의-키 일치**다.

    지금        cand = act[u] × w × 0.65
    질의 게이트  cand = act[u] × w × 0.65 × g

    g = clamp((sim(q,v) - lo) / (hi - lo), floor, 1.0)

sim은 이미 계산된 bge 내적이다. 추가 비용이 없다.

**floor를 0으로 두지 않는다.** 하드 차단은 파싱·임베딩 실패 하나가 정답 경로를
통째로 막는다(명제층에서 겪었다). 약하게 누르는 편이 실패에 견딘다.

또 하나 — 사전 측정에서 별칭 실험의 희석 원인이 **합치기 순서**임이 드러났다.
확산 결과를 전부 앞에 넣으면 좋은 dense 결과가 밀려난다. 그것도 같이 잰다.

── 사전등록 ────────────────────────────────────────────────────────────────
조건    B0 통제군      현재 그대로 (게이트 없음, 확산 우선 합치기)
        G1 질의게이트  게이트만 켬
        M1 점수합치기  게이트 없이 **합치기 순서만** 점수 기준으로
        GM 둘 다
주 가설  HotpotQA 근거2개@10에서 GM > B0. dense가 강해 희석 손해가 가장 컸던 곳이다.
        McNemar 대응 이분, α=0.05, n=1000.
부 가설  2Wiki·MuSiQue 동일 지표. 유의하지 않아도 보고한다.
안전장치 어느 데이터셋이든 R@10이 1.0%p 넘게 떨어지면 실패로 기록한다.
기전 예측 이득은 **확산이 dense 결과를 밀어냈던 질문**에 몰려야 한다.
파라미터 lo=0.30 hi=0.75 floor=0.25로 **미리 고정한다.** 돌린 뒤 바꾸지 않는다.
실패 조건 유의하지 않으면 그대로 적고 파라미터를 재탐색하지 않는다.
────────────────────────────────────────────────────────────────────────────

인자: DATASET N_QUESTIONS SEED
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
MAXW = 6
LO, HI, FLOOR = 0.30, 0.75, 0.25          # 사전등록 값. 실행 후 변경 금지

DATASET = sys.argv[1] if len(sys.argv) > 1 else "hotpotqa"
N_Q = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0

PARQUET_REV = "refs/convert/parquet"
DATASETS = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor/validation/0000.parquet"),
    "2wiki":    ("framolfese/2WikiMultihopQA", "default/validation/0000.parquet"),
    "musique":  ("dgslibisey/MuSiQue", "default/validation/0000.parquet"),
}
_N = re.compile(r"[^a-z0-9 ]+")


def norm(s):
    return " ".join(_N.sub(" ", s.lower()).split())


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
        questions.append({"q": row["question"], "gold": gold})
    titles = list(pool)
    return titles, [pool[t] for t in titles], questions


def build_edges(titles, texts):
    lookup = {}
    for i, t in enumerate(titles):
        k = norm(t)
        if len(k) >= 5:
            lookup.setdefault(k, i)
    edges = []
    for i, body in enumerate(texts):
        toks = norm(body).split()
        hit = set()
        for n in range(1, MAXW + 1):
            for a in range(len(toks) - n + 1):
                j = lookup.get(" ".join(toks[a:a + n]))
                if j is not None and j != i:
                    hit.add(j)
        edges.extend((i, j) for j in hit)
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
    from ngdb import DenseIndex, Graph

    titles, texts, questions = load_pool(DATASET, N_Q, SEED)
    edges = build_edges(titles, texts)
    print(f"[{time.time()-t0:6.1f}s] {DATASET}: 질문 {len(questions)} 풀 {len(titles)} "
          f"엣지 {len(edges):,}")

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
    hits = [dense.search(Q[i], MAXK) for i in range(len(questions))]
    print(f"[{time.time()-t0:6.1f}s] 벡터 색인 {dense.size}")

    def gate_for(i):
        """질문 i에 대한 노드별 게이트. 이미 있는 벡터의 내적이라 추가 비용이 없다."""
        sim = P @ Q[i]
        g = (sim - LO) / (HI - LO)
        return np.clip(g, FLOOR, 1.0).astype(np.float32)

    # 게이트를 엣지 가중치에 실어 넣는다 — 목적지 v의 질의 일치도를 곱한다.
    # C++ spread는 질의를 모르므로, 질문마다 가중치를 바꿔 그래프를 새로 짓는다.
    # 엣지가 수천~수만이라 그래도 싸다.
    def graph_plain():
        g = Graph(len(titles))
        g.add_edges([(a, b, 1.0) for a, b in edges])
        return g

    g_plain = graph_plain()

    def rank(i, gated, score_merge):
        sh = hits[i][:N_SEEDS]
        if gated:
            gt = gate_for(i)
            g = Graph(len(titles))
            g.add_edges([(a, b, float(gt[b])) for a, b in edges])
        else:
            g = g_plain
        acts = g.spread([d for d, _ in sh], [float(s) for _, s in sh],
                        3, 0.65, 0.02, MAXK)
        if not score_merge:
            seen, out = set(), []
            for d, _ in acts:
                if d not in seen:
                    seen.add(d); out.append(d)
            for d, _ in hits[i]:
                if d not in seen:
                    seen.add(d); out.append(d)
            return out
        # 점수 기준 합치기 — 확산과 dense를 같은 자에 놓고 큰 쪽을 남긴다
        best = {}
        for d, s in hits[i]:
            best[d] = max(best.get(d, 0.0), float(s))
        for d, s in acts:
            best[d] = max(best.get(d, 0.0), float(s))
        return [d for d, _ in sorted(best.items(), key=lambda x: -x[1])]

    def evaluate(gated, score_merge):
        rows = []
        for i in range(len(questions)):
            ranked = [titles[d] for d in rank(i, gated, score_merge)[:MAXK]]
            gold = questions[i]["gold"]
            r = {f"r@{k}": sum(1 for gg in gold if gg in ranked[:k]) / len(gold) for k in KS}
            r["all@10"] = float(all(gg in ranked[:10] for gg in gold))
            rows.append(r)
        return rows

    conds = {"B0_통제군": (False, False), "G1_질의게이트": (True, False),
             "M1_점수합치기": (False, True), "GM_둘다": (True, True)}
    per_q, agg = {}, {}
    for name, (ga, sm) in conds.items():
        rows = evaluate(ga, sm)
        per_q[name] = rows
        agg[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print(f"[{time.time()-t0:6.1f}s] {name} 완료")

    print(f"\n{'='*78}\n질의 게이팅 {DATASET} — lo={LO} hi={HI} floor={FLOOR} "
          f"n={len(questions)}\n{'='*78}")
    print(f"{'조건':<15}" + "".join(f"{'R@'+str(k):>9}" for k in KS) + f"{'근거2개@10':>12}")
    for name, v in agg.items():
        print(f"{name:<15}" + "".join(f"{v[f'r@{k}']:>9.3f}" for k in KS)
              + f"{v['all@10']:>12.3f}")

    expected = {"hotpotqa": 0.940, "2wiki": 0.907, "musique": 0.350}
    print(f"\n통제군 대조: {agg['B0_통제군']['all@10']:.3f} "
          f"(G1 기록 {expected[DATASET]:.3f})")

    base = per_q["B0_통제군"]
    tests = {}
    print()
    for name in ("G1_질의게이트", "M1_점수합치기", "GM_둘다"):
        mc = mcnemar([(bool(a["all@10"]), bool(b["all@10"]))
                      for a, b in zip(base, per_q[name])])
        tests[name] = mc
        v = ("앞섬" if mc["gain"] > mc["loss"] and mc["p_value"] < 0.05 else
             "뒤짐" if mc["loss"] > mc["gain"] and mc["p_value"] < 0.05 else "차이 없음")
        print(f"{name} vs 통제군: 이득 {mc['gain']} 손실 {mc['loss']} "
              f"(불일치 {mc['n_discordant']}, p={mc['p_value']:.4f}) → {v}")

    # 기전 — 확산이 dense를 밀어냈던 질문에 이득이 몰렸는가
    pushed = []
    for i in range(len(questions)):
        dense_ids = [d for d, _ in hits[i]][:10]
        b0 = rank(i, False, False)[:10]
        if any(d not in b0 for d in dense_ids):
            pushed.append(i)
    print(f"\n기전 확인 (사전등록: 확산이 dense를 밀어낸 질문에 이득이 몰려야 한다)")
    for label, ids in (("밀어낸 질문", pushed),
                       ("그 외", [i for i in range(len(questions)) if i not in set(pushed)])):
        if not ids:
            continue
        m = mcnemar([(bool(base[i]["all@10"]), bool(per_q["GM_둘다"][i]["all@10"]))
                     for i in ids])
        print(f"  {label:<14} n={len(ids):>4}  이득 {m['gain']:>3} 손실 {m['loss']:>3}")

    drop = agg["B0_통제군"]["r@10"] - agg["GM_둘다"]["r@10"]
    if drop > 0.01:
        print(f"\n! 안전장치 위반 — R@10이 {drop*100:.1f}%p 떨어졌다. 실패로 기록한다")

    payload = {"dataset": DATASET, "n_questions": len(questions),
               "pool_size": len(titles), "edges": len(edges),
               "lo": LO, "hi": HI, "floor": FLOOR, "seed": SEED,
               "results": agg, "mcnemar": tests, "n_pushed": len(pushed),
               "runtime_sec": time.time() - t0}
    out = Path(f"/tmp/gate_{DATASET}_n{len(questions)}.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True, private=True)
    api.upload_file(path_or_fileobj=str(out), path_in_repo=f"runs/{out.name}",
                    repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"\n업로드: runs/{out.name}")


if __name__ == "__main__":
    main()
