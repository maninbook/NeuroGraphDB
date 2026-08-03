# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11", "numpy", "pyarrow",
#     "transformers>=4.40", "torch",
#     "huggingface_hub>=0.28",
# ]
# ///
"""H1 — 인출이 DB를 바꾸는가. 헤비안 강화를 켜고 측정한다.

여태 못 켰던 이유: 1회성 QA 벤치마크에는 **강화할 사용 이력이 없다.**
질문 하나를 한 번 풀고 끝나면 학습할 것이 없다. 그래서 평가 프로토콜을 바꾼다.

프로토콜 — 질의 흐름 분할
    질문 N개를 고정 시드로 정렬
      Phase A (준비)  앞 절반 — 검색하며 공동인출 쌍의 엣지가 강화됨
      Phase B (측정)  뒤 절반 — 여기서만 점수를 냄
      통제군          Phase A를 강화 없이 지나간 뒤 **같은** Phase B
    같은 측정 질문을 두 조건이 푸는 대응 설계 → McNemar

강화 신호 — **공동인출만 쓴다.** 정답 라벨을 보고 강화하면 숫자는 예쁘지만 라벨 누수다.
상위 M개 결과가 함께 떴다는 사실만으로 서로의 엣지를 올린다. 배포 환경에서도 얻을 수 있는
신호이고, 그래서 정직하다.

**사전 점검을 먼저 한다.** Phase B의 정답 문단이 Phase A에서 한 번이라도 인출됐는지 센다.
그 비율이 낮으면 학습할 것이 애초에 없다는 뜻이고, 실험을 돌려도 "효과 없음"만 나온다.
(slowcode에서 경합 구간이 10개뿐이라 아무것도 측정 못 한 함정과 같다.)

**부익부를 같이 잰다.** 자주 인출된 문단이 더 세지면 드문 주제가 손해를 본다.
평균만 보면 안 보이므로 Phase A 노출 빈도로 머리/꼬리를 갈라 따로 보고한다.
꼬리가 나빠지면 그대로 쓴다.

인자: DATASET N_QUESTIONS DELTA REINFORCE_TOP SEED
"""

import json
import os
import re
import subprocess
import sys
import sysconfig
import time
from collections import Counter
from pathlib import Path

CODE_REPO = "goethe0101/neurographdb"
RESULTS_REPO = "goethe0101/neurographdb-results"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
N_SEEDS = 5
MAXK = 20

DATASET = sys.argv[1] if len(sys.argv) > 1 else "hotpotqa"
N_Q = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
DELTA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
REINFORCE_TOP = int(sys.argv[4]) if len(sys.argv) > 4 else 8
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


def build_mention_edges(titles, texts, max_title_words=6):
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
    from ngdb import DenseIndex, Graph

    titles, texts, questions = load_pool(DATASET, N_Q, SEED)
    tidx = {t: i for i, t in enumerate(titles)}
    edges = build_mention_edges(titles, texts)
    print(f"[{time.time()-t0:6.1f}s] {DATASET}: 질문 {len(questions)} | 풀 {len(titles)} "
          f"| 기본 엣지 {len(edges):,}")

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
    print(f"[{time.time()-t0:6.1f}s] 벡터 색인 {dense.size}")

    dense_hits = [dense.search(Q[i], MAXK) for i in range(len(questions))]

    def retrieve(g, i):
        hits = dense_hits[i][:N_SEEDS]
        acts = g.spread([d for d, _ in hits], [float(s) for _, s in hits],
                        3, 0.65, 0.02, MAXK)
        seen, out = set(), []
        for d, _ in acts:
            if d not in seen:
                seen.add(d); out.append(d)
        for d, _ in dense_hits[i]:
            if d not in seen:
                seen.add(d); out.append(d)
        return out

    half = len(questions) // 2
    phase_a, phase_b = range(half), range(half, len(questions))
    print(f"[{time.time()-t0:6.1f}s] Phase A {len(phase_a)} / Phase B {len(phase_b)}")

    # ── 사전 점검: Phase B의 정답이 Phase A에서 인출된 적 있나 ──────────────
    g0 = Graph(len(titles)); g0.add_edges(edges)
    exposure = Counter()
    for i in phase_a:
        for d in retrieve(g0, i)[:REINFORCE_TOP]:
            exposure[d] += 1

    covered = 0
    for i in phase_b:
        gold_ids = [tidx[t] for t in questions[i]["gold"] if t in tidx]
        if any(exposure.get(d, 0) > 0 for d in gold_ids):
            covered += 1
    cover_rate = covered / max(len(phase_b), 1)
    print(f"\n{'='*72}")
    print(f"사전 점검 — Phase B 정답이 Phase A에서 인출된 적 있는 질문: "
          f"{covered}/{len(phase_b)} = {cover_rate:.1%}")
    print(f"  Phase A에서 노출된 고유 문단 {len(exposure):,}/{len(titles):,} "
          f"({len(exposure)/len(titles):.1%})")
    if cover_rate < 0.15:
        print("  ! 낮다. 강화할 신호가 거의 없어 '효과 없음'이 나올 수밖에 없다")
    print(f"{'='*72}")

    def eval_b(g):
        rows = []
        for i in phase_b:
            ranked = [titles[d] for d in retrieve(g, i)[:MAXK]]
            gold = questions[i]["gold"]
            rows.append({
                "r@10": sum(1 for gg in gold if gg in ranked[:10]) / len(gold),
                "all@10": float(all(gg in ranked[:10] for gg in gold)),
            })
        return rows

    # ── 통제군: 강화 없이 Phase A를 지나간다 ────────────────────────────────
    ctrl = eval_b(g0)

    # ── 실험군: Phase A에서 공동인출 쌍을 강화한다 ──────────────────────────
    g1 = Graph(len(titles)); g1.add_edges(edges)
    for i in phase_a:
        co = retrieve(g1, i)[:REINFORCE_TOP]
        g1.reinforce(co, DELTA, 1.0)
    print(f"[{time.time()-t0:6.1f}s] 강화 후 엣지 {g1.n_edges:,} "
          f"(기본 {len(edges):,} → +{g1.n_edges-len(edges):,})")
    heb = eval_b(g1)

    def agg(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in ("r@10", "all@10")}

    a_ctrl, a_heb = agg(ctrl), agg(heb)
    mc = mcnemar([(bool(c["all@10"]), bool(h["all@10"])) for c, h in zip(ctrl, heb)])
    verdict = ("헤비안이 통제군을 넘음" if mc["gain"] > mc["loss"] and mc["p_value"] < 0.05
               else "헤비안이 더 나쁨" if mc["loss"] > mc["gain"] and mc["p_value"] < 0.05
               else f"판정 불가 — 불일치 {mc['n_discordant']}" if mc["n_discordant"] < 10
               else "유의하지 않음")

    print(f"\n{'='*72}\nH1 {DATASET} — delta={DELTA} 강화상위={REINFORCE_TOP}\n{'='*72}")
    print(f"{'조건':<14}{'R@10':>10}{'근거2개@10':>13}")
    print(f"{'통제군':<14}{a_ctrl['r@10']:>10.3f}{a_ctrl['all@10']:>13.3f}")
    print(f"{'헤비안':<14}{a_heb['r@10']:>10.3f}{a_heb['all@10']:>13.3f}")
    print(f"\nMcNemar: 이득 {mc['gain']} 손실 {mc['loss']} "
          f"(불일치 {mc['n_discordant']}, p={mc['p_value']:.4f}) → {verdict}")

    # ── 부익부 점검: Phase A 노출 빈도로 머리/꼬리를 가른다 ─────────────────
    head_ctrl, head_heb, tail_ctrl, tail_heb = [], [], [], []
    for n, i in enumerate(phase_b):
        gold_ids = [tidx[t] for t in questions[i]["gold"] if t in tidx]
        exp = max((exposure.get(d, 0) for d in gold_ids), default=0)
        (head_ctrl if exp > 0 else tail_ctrl).append(ctrl[n]["all@10"])
        (head_heb if exp > 0 else tail_heb).append(heb[n]["all@10"])
    print(f"\n부익부 점검 (Phase A 노출 여부로 분할)")
    print(f"  머리(노출됨) n={len(head_ctrl):>4}  통제 {np.mean(head_ctrl):.3f} → "
          f"헤비안 {np.mean(head_heb):.3f}")
    if tail_ctrl:
        print(f"  꼬리(미노출) n={len(tail_ctrl):>4}  통제 {np.mean(tail_ctrl):.3f} → "
              f"헤비안 {np.mean(tail_heb):.3f}")
        if np.mean(tail_heb) < np.mean(tail_ctrl) - 1e-9:
            print("  ! 꼬리가 나빠졌다. 자주 쓰인 것이 세지며 드문 주제를 밀어낸다")

    payload = {"dataset": DATASET, "n_questions": len(questions),
               "pool_size": len(titles), "delta": DELTA,
               "reinforce_top": REINFORCE_TOP, "seed": SEED,
               "precheck_cover_rate": cover_rate,
               "exposed_passages": len(exposure),
               "base_edges": len(edges), "edges_after": g1.n_edges,
               "control": a_ctrl, "hebbian": a_heb, "mcnemar": mc, "verdict": verdict,
               "head_control": float(np.mean(head_ctrl)) if head_ctrl else None,
               "head_hebbian": float(np.mean(head_heb)) if head_heb else None,
               "tail_control": float(np.mean(tail_ctrl)) if tail_ctrl else None,
               "tail_hebbian": float(np.mean(tail_heb)) if tail_heb else None,
               "runtime_sec": time.time() - t0}
    out = Path(f"/tmp/hebbian_{DATASET}_n{len(questions)}_d{DELTA}_t{REINFORCE_TOP}_s{SEED}.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True, private=True)
    api.upload_file(path_or_fileobj=str(out), path_in_repo=f"runs/{out.name}",
                    repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"\n업로드: https://huggingface.co/datasets/{RESULTS_REPO}")


if __name__ == "__main__":
    main()
