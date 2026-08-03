# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11", "numpy", "pyarrow",
#     "transformers>=4.40", "torch", "huggingface_hub>=0.28",
# ]
# ///
"""L — 질의당 검색 지연시간 실측. 에이전트 반복질의 비용을 따지려면 이게 필요하다.

배경. 2026년 무게중심이 **강화학습으로 훈련된 검색 에이전트**로 옮겨갔다
(AutoSearch, Agentic-R, Search-P1, CuSearch, TreePS-RAG …).
에이전트는 한 질문에 검색을 5~20회 부른다. 그러면 **질의당 비용이 그만큼 곱해진다.**

공개된 질의당 비용 (LogicRAG 논문 표 5, 2Wiki, GPT-4o-mini + RTX 3090):

    VanillaRAG    4.28초    490 토큰
    HippoRAG 2    5.89초  2,809 토큰
    LogicRAG      9.83초  1,778 토큰
    GraphRAG     13.05초  4,700 토큰

논문이 명시한다 — **"이 비교는 그래프 구축 비용을 제외한 것"**이고
그래프 기반은 구축에 수십~수백 분이 든다.

우리는 **질의 시점에 LLM을 안 부른다.** 그러니 토큰은 0이고 지연은 임베딩+확산뿐이다.
그 값을 실측해서 저 표에 나란히 놓을 수 있게 한다.

**비교의 한계를 미리 못 박는다.**
  - 하드웨어가 다르다(우리 L4 vs 저쪽 RTX 3090). 절대 시간은 직접 비교 불가.
  - 코퍼스 크기가 다르다. 우리 풀은 수천, 저쪽도 비슷하지만 동일하지 않다.
  - **토큰 0은 하드웨어와 무관하다.** 이쪽이 진짜 비교 가능한 축이다.
그래서 결론은 "우리가 몇 배 빠르다"가 아니라 **"질의당 LLM 토큰이 0이라
에이전트 루프에서 곱해지지 않는다"**로만 낸다.

측정 항목
  1. 질의 인코딩 (bge-base)
  2. dense 검색 (C++ brute force)
  3. 그래프 확산 (C++)
  4. 합치기
각각 중앙값·p95를 낸다. 워밍업 후 측정한다.

인자: DATASET N_QUESTIONS
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
MAXW = 6
WARMUP = 20

DATASET = sys.argv[1] if len(sys.argv) > 1 else "2wiki"
N_Q = int(sys.argv[2]) if len(sys.argv) > 2 else 500
SEED = 0

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


def main():
    import numpy as np
    import torch
    t0 = time.time()
    build_core()
    from ngdb import DenseIndex, Graph

    titles, texts, questions = load_pool(DATASET, N_Q, SEED)
    t_edge0 = time.perf_counter()
    edges = build_edges(titles, texts)
    t_edge = time.perf_counter() - t_edge0
    print(f"[{time.time()-t0:6.1f}s] {DATASET}: 질문 {len(questions)} 풀 {len(titles)} "
          f"엣지 {len(edges):,}")
    print(f"  색인(엣지 생성) {t_edge:.2f}초 = 문단당 {t_edge/len(titles)*1000:.3f}ms")

    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    etok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    emodel = AutoModel.from_pretrained(EMBED_MODEL).to(dev).eval()

    def encode_one(text):
        b = etok([text], padding=True, truncation=True, max_length=512,
                 return_tensors="pt").to(dev)
        with torch.no_grad():
            h = emodel(**b).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(h, dim=-1).cpu().numpy()[0].astype(np.float32)

    def encode_batch(items, batch=128):
        out = []
        for a in range(0, len(items), batch):
            b = etok(items[a:a + batch], padding=True, truncation=True,
                     max_length=512, return_tensors="pt").to(dev)
            with torch.no_grad():
                h = emodel(**b).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    P = encode_batch([f"{t}. {x}" for t, x in zip(titles, texts)])
    dense = DenseIndex(P.shape[1]); dense.add_batch(P)
    g = Graph(len(titles)); g.add_edges(edges)
    print(f"[{time.time()-t0:6.1f}s] 벡터 색인 {dense.size}")

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    qs = [q["q"] for q in questions]
    # 워밍업 — 커널 컴파일·캐시 효과를 측정에서 뺀다
    for q in qs[:WARMUP]:
        v = encode_one(QUERY_PREFIX + q)
        h = dense.search(v, MAXK)
        g.spread([d for d, _ in h[:N_SEEDS]], [float(s) for _, s in h[:N_SEEDS]],
                 3, 0.65, 0.02, MAXK)
    sync()

    t_enc, t_srch, t_spr, t_mrg, t_all = [], [], [], [], []
    for q in qs:
        a0 = time.perf_counter()
        v = encode_one(QUERY_PREFIX + q); sync()
        a1 = time.perf_counter()
        h = dense.search(v, MAXK)
        a2 = time.perf_counter()
        acts = g.spread([d for d, _ in h[:N_SEEDS]], [float(s) for _, s in h[:N_SEEDS]],
                        3, 0.65, 0.02, MAXK)
        a3 = time.perf_counter()
        seen, out = set(), []
        for d, _ in acts:
            if d not in seen:
                seen.add(d); out.append(d)
        for d, _ in h:
            if d not in seen:
                seen.add(d); out.append(d)
        out = out[:MAXK]
        a4 = time.perf_counter()
        t_enc.append(a1-a0); t_srch.append(a2-a1); t_spr.append(a3-a2)
        t_mrg.append(a4-a3); t_all.append(a4-a0)

    def stat(v):
        v = np.array(v)*1000
        return float(np.median(v)), float(np.percentile(v,95)), float(v.mean())

    parts = [("질의 인코딩(bge)", t_enc), ("dense 검색", t_srch),
             ("그래프 확산", t_spr), ("합치기", t_mrg), ("전체", t_all)]
    print(f"\n{'='*72}\n질의당 지연 {DATASET} — 풀 {len(titles)}, 질문 {len(questions)}"
          f", {dev}\n{'='*72}")
    print(f"  {'단계':<20}{'중앙(ms)':>11}{'p95(ms)':>11}{'평균(ms)':>11}")
    res = {}
    for nm, v in parts:
        m,p,a = stat(v); res[nm] = {"median_ms":m,"p95_ms":p,"mean_ms":a}
        print(f"  {nm:<20}{m:>11.2f}{p:>11.2f}{a:>11.2f}")

    med = res["전체"]["median_ms"]
    print(f"\n  **질의당 LLM 토큰 0** — 이 값은 하드웨어와 무관하다")
    print(f"\n  에이전트가 질문 1건에 검색을 k번 부를 때 (우리)")
    for k in (1,5,10,20):
        print(f"    k={k:>2}: {med*k/1000:.3f}초, LLM 토큰 0")
    print(f"\n  !! 공개 수치와 나란히 놓지 말 것 — 재는 대상이 다르다.")
    print(f"     LogicRAG 논문 표(4.28~13.05초)는 **답변 생성까지 포함한 질의 처리 전체**이고,")
    print(f"     위 8.54ms는 **검색만**이다. 답변 LLM을 돌리지 않았다. 배수로 말하면 거짓이 된다.")
    print(f"     비교 가능한 축은 **검색에 쓰는 LLM 토큰**뿐이다:")
    print(f"       우리        검색 LLM 토큰 0      (에이전트가 k번 불러도 0)")
    print(f"       HippoRAG 2  질의당 사실 재순위 LLM 호출 1회  → k배로 늘어남")
    print(f"       LogicRAG    질의당 분해+DAG+가지치기 LLM 호출 → k배로 늘어남")

    payload = {"dataset": DATASET, "n_questions": len(questions),
               "pool_size": len(titles), "n_edges": len(edges), "device": dev,
               "index_edge_seconds": t_edge,
               "index_ms_per_passage": t_edge/len(titles)*1000,
               "query_llm_tokens": 0, "latency": res,
               "note": "하드웨어가 달라 공개 수치와 절대 시간 비교 불가. 토큰 0만 비교 가능"}
    out_p = Path(f"/tmp/latency_{DATASET}_n{len(questions)}.json")
    out_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=str(out_p), path_in_repo=f"runs/{out_p.name}",
                        repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"\n업로드: runs/{out_p.name}")


if __name__ == "__main__":
    main()
