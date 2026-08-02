# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11", "numpy", "pyarrow",
#     "transformers>=4.40", "torch", "huggingface_hub>=0.28",
# ]
# ///
"""K — 깊게 뽑고 크로스인코더로 재순위. 표준 RAG 방식을 처음으로 붙인다.

정정에서 나온 것: "재순위 소진"은 **상위 20개 재배열**의 얘기였다.
깊게 뽑으면 정답이 거기 있다 — MuSiQue top-10 0.341 vs top-100 **0.647**.
top-100 재순위의 상한이 **+30.6%p**로, 이 프로젝트에서 잰 어떤 여지보다 한 자릿수 크다.

재순위기는 크로스인코더(`BAAI/bge-reranker-v2-m3`, 568M)다. **LLM이 아니다.**
그래프는 여전히 LLM 0회이고, 재순위는 작은 전용 모델이다.

── 사전등록 ────────────────────────────────────────────────────────────────
조건    B0  현재 그대로 (그래프, top-10, 재순위 없음)  — 통제군
        R1  **그래프 top-100 → 크로스인코더 → top-10**   — 주 조건
        R2  dense top-100 → 크로스인코더 → top-10        — 그래프 기여가 살아남나
주 가설  MuSiQue 근거2개@10에서 R1 > B0. McNemar 대응 이분, α=0.05, n=1000.
부 가설  HotpotQA·2Wiki 동일 지표. 그리고 **R1 vs R2** — 재순위 뒤에도 그래프가
        기여하는가. 이게 우리 방법이 표준 파이프라인 안에서 살아남는지를 가른다.
안전장치 어느 데이터셋이든 R@10이 1.0%p 넘게 떨어지면 실패로 기록한다.
기전 예측 이득은 **정답이 11~100위에 있던 질문**에 몰려야 한다.
        top-10 안에 이미 다 있던 질문에서는 손실이 거의 없어야 한다.
파라미터 후보 100개, max_length 512, fp16. 실행 전 고정하고 바꾸지 않는다.
실패 조건 유의하지 않으면 그대로 적고 후보 수·재순위기를 바꿔 재시도하지 않는다.
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
RERANKER = "BAAI/bge-reranker-v2-m3"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
N_SEEDS = 5
CAND = 100            # 사전등록: 후보 100개
MAXK = 20
KS = (1, 2, 5, 10, 20)
MAXW = 6

DATASET = sys.argv[1] if len(sys.argv) > 1 else "musique"
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
    import torch
    t0 = time.time()
    build_core()
    from ngdb import DenseIndex, Graph

    titles, texts, questions = load_pool(DATASET, N_Q, SEED)
    tidx = {t: i for i, t in enumerate(titles)}
    g = Graph(len(titles)); g.add_edges(build_edges(titles, texts))
    print(f"[{time.time()-t0:6.1f}s] {DATASET}: 질문 {len(questions)} 풀 {len(titles)}")

    from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
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

    docs = [f"{t}. {x}" for t, x in zip(titles, texts)]
    P = encode(docs)
    dense = DenseIndex(P.shape[1]); dense.add_batch(P)
    Q = encode([QUERY_PREFIX + q["q"] for q in questions])
    print(f"[{time.time()-t0:6.1f}s] 벡터 색인 {dense.size}")

    dense_cand, graph_cand = [], []
    for i in range(len(questions)):
        hits = dense.search(Q[i], CAND)
        d_ids = [d for d, _ in hits]
        dense_cand.append(d_ids)
        sh = hits[:N_SEEDS]
        acts = g.spread([d for d, _ in sh], [float(s) for _, s in sh],
                        3, 0.65, 0.02, CAND)
        seen, gr = set(), []
        for d, _ in acts:
            if d not in seen:
                seen.add(d); gr.append(d)
        for d in d_ids:
            if d not in seen:
                seen.add(d); gr.append(d)
        graph_cand.append(gr[:CAND])

    del emodel
    torch.cuda.empty_cache()
    rtok = AutoTokenizer.from_pretrained(RERANKER)
    rmodel = AutoModelForSequenceClassification.from_pretrained(
        RERANKER, torch_dtype=torch.float16).to(dev).eval()
    print(f"[{time.time()-t0:6.1f}s] 재순위기 로드 {RERANKER}")

    def rerank(cands, batch=128):
        out = []
        for i, ids in enumerate(cands):
            q = questions[i]["q"]
            scores = []
            for a in range(0, len(ids), batch):
                chunk = ids[a:a + batch]
                enc = rtok([q] * len(chunk), [docs[d] for d in chunk],
                           padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to(dev)
                with torch.no_grad():
                    s = rmodel(**enc).logits.view(-1).float().cpu().numpy()
                scores.append(s)
            s = np.concatenate(scores)
            order = np.argsort(-s)
            out.append([ids[j] for j in order])
            if (i + 1) % 250 == 0:
                print(f"  [{time.time()-t0:6.1f}s] 재순위 {i+1}/{len(cands)}")
        return out

    r1 = rerank(graph_cand)
    r2 = rerank(dense_cand)

    def score(ranks):
        rows = []
        for i, q in enumerate(questions):
            R = [titles[d] for d in ranks[i][:MAXK]]
            gold = q["gold"]
            r = {f"r@{k}": sum(1 for gv in gold if gv in R[:k]) / len(gold) for k in KS}
            r["all@10"] = float(all(gv in R[:10] for gv in gold))
            rows.append(r)
        return rows

    conds = {"B0_통제군": [c[:MAXK] for c in graph_cand],
             "R1_그래프+재순위": r1, "R2_dense+재순위": r2}
    per_q, agg = {}, {}
    for name, ranks in conds.items():
        rows = score(ranks)
        per_q[name] = rows
        agg[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    print(f"\n{'='*80}\n깊은 후보 + 크로스인코더 재순위 {DATASET} — 후보 {CAND}, n={len(questions)}"
          f"\n{'='*80}")
    print(f"{'조건':<18}" + "".join(f"{'R@'+str(k):>9}" for k in KS) + f"{'근거2개@10':>12}")
    for name, v in agg.items():
        print(f"{name:<18}" + "".join(f"{v[f'r@{k}']:>9.3f}" for k in KS)
              + f"{v['all@10']:>12.3f}")

    expected = {"hotpotqa": 0.940, "2wiki": 0.907, "musique": 0.350}
    print(f"\n통제군 대조: {agg['B0_통제군']['all@10']:.3f} (G1 기록 {expected[DATASET]:.3f})")

    base = per_q["B0_통제군"]
    tests = {}
    print()
    for name in ("R1_그래프+재순위", "R2_dense+재순위"):
        mc = mcnemar([(bool(a["all@10"]), bool(b["all@10"]))
                      for a, b in zip(base, per_q[name])])
        tests[f"B0 vs {name}"] = mc
        v = ("앞섬" if mc["gain"] > mc["loss"] and mc["p_value"] < 0.05 else
             "뒤짐" if mc["loss"] > mc["gain"] and mc["p_value"] < 0.05 else "차이 없음")
        print(f"{name} vs 통제군: 이득 {mc['gain']} 손실 {mc['loss']} "
              f"(불일치 {mc['n_discordant']}, p={mc['p_value']:.4f}) → {v}")

    mc = mcnemar([(bool(a["all@10"]), bool(b["all@10"]))
                  for a, b in zip(per_q["R2_dense+재순위"], per_q["R1_그래프+재순위"])])
    tests["R2 vs R1"] = mc
    v = ("그래프가 재순위 뒤에도 기여한다" if mc["gain"] > mc["loss"] and mc["p_value"] < 0.05
         else "그래프 기여가 재순위에 흡수됐다" if mc["n_discordant"] < 10 or mc["p_value"] >= 0.05
         else "그래프가 오히려 해가 된다")
    print(f"R1 vs R2 (그래프 기여): 이득 {mc['gain']} 손실 {mc['loss']} "
          f"(불일치 {mc['n_discordant']}, p={mc['p_value']:.4f}) → {v}")

    # 기전 — 정답이 11~100위에 있던 질문에 이득이 몰렸는가
    deep, shallow = [], []
    for i, q in enumerate(questions):
        gold = [tidx[t] for t in q["gold"] if t in tidx]
        (shallow if all(gv in graph_cand[i][:10] for gv in gold) else deep).append(i)
    print(f"\n기전 확인 (사전등록: 이득은 정답이 11~100위에 있던 질문에 몰려야 한다)")
    for label, ids in (("정답이 깊이 있었음", deep), ("이미 top-10", shallow)):
        if not ids:
            continue
        m = mcnemar([(bool(base[i]["all@10"]), bool(per_q["R1_그래프+재순위"][i]["all@10"]))
                     for i in ids])
        print(f"  {label:<18} n={len(ids):>4}  이득 {m['gain']:>3} 손실 {m['loss']:>3}")

    drop = agg["B0_통제군"]["r@10"] - agg["R1_그래프+재순위"]["r@10"]
    if drop > 0.01:
        print(f"\n! 안전장치 위반 — R@10이 {drop*100:.1f}%p 떨어졌다. 실패로 기록한다")

    payload = {"dataset": DATASET, "n_questions": len(questions),
               "pool_size": len(titles), "candidates": CAND, "reranker": RERANKER,
               "seed": SEED, "results": agg, "mcnemar": tests,
               "runtime_sec": time.time() - t0}
    out = Path(f"/tmp/rerank_{DATASET}_s{SEED}.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=str(out), path_in_repo=f"runs/{out.name}",
                        repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"\n업로드: runs/{out.name}")


if __name__ == "__main__":
    main()
