# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11", "numpy", "pyarrow",
#     "transformers>=4.40", "torch", "huggingface_hub>=0.28",
# ]
# ///
"""사전 측정 — 후보를 깊게 뽑으면 정답이 거기 있는가.

**앞선 주장의 정정.** "재순위는 소진됐다(상한 1~5%p)"고 적었는데,
그건 **상위 20개를 재배열했을 때**의 상한이었다. 후보를 100·200개로 늘리고
그중에서 고르는 것은 다른 문제이고 재본 적이 없다.
실제 RAG 시스템은 깊게 뽑고(top-100) 세게 재순위하고 얕게 읽는다(top-10).

여기서 재는 것:
  1. k를 20 → 50 → 100 → 200으로 늘리면 근거 전부 포함률이 얼마나 오르나
  2. 그게 **재순위기가 노릴 수 있는 진짜 상한**이다
  3. dense 단독과 dense+그래프가 깊은 구간에서 어떻게 갈리나
     (그래프가 깊이에서도 이득을 주면 깊은 후보 생성에도 가치가 있다)

상한이 낮으면 재순위기(크로스인코더 등)를 붙일 이유가 없다.
높으면 그게 지금 열린 가장 큰 여지다.

LLM 불필요.
"""

import os
import re
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

CODE_REPO = "goethe0101/neurographdb"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
N_SEEDS = 5
MAXW = 6
DEEP = (10, 20, 50, 100, 200, 500)

N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
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
    t0 = time.time()
    build_core()
    from ngdb import DenseIndex, Graph

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

    for ds in DATASETS:
        titles, texts, questions = load_pool(ds, N_Q, SEED)
        tidx = {t: i for i, t in enumerate(titles)}
        P = encode([f"{t}. {x}" for t, x in zip(titles, texts)])
        dense = DenseIndex(P.shape[1]); dense.add_batch(P)
        Q = encode([QUERY_PREFIX + q["q"] for q in questions])
        g = Graph(len(titles)); g.add_edges(build_edges(titles, texts))
        big = max(DEEP)

        cov = {("dense", k): 0 for k in DEEP}
        cov.update({("graph", k): 0 for k in DEEP})
        for i, q in enumerate(questions):
            hits = dense.search(Q[i], big)
            d_ids = [d for d, _ in hits]
            sh = hits[:N_SEEDS]
            acts = g.spread([d for d, _ in sh], [float(s) for _, s in sh],
                            3, 0.65, 0.02, big)
            seen, gr = set(), []
            for d, _ in acts:
                if d not in seen:
                    seen.add(d); gr.append(d)
            for d in d_ids:
                if d not in seen:
                    seen.add(d); gr.append(d)
            gold = [tidx[t] for t in q["gold"] if t in tidx]
            for k in DEEP:
                if all(gv in d_ids[:k] for gv in gold):
                    cov[("dense", k)] += 1
                if all(gv in gr[:k] for gv in gold):
                    cov[("graph", k)] += 1

        n = len(questions)
        print(f"\n{'='*74}\n{ds}  질문 {n}  풀 {len(titles)}\n{'='*74}")
        print(f"  근거 전부를 상위 k에 담은 비율")
        print(f"  {'k':>6}" + "".join(f"{x:>12}" for x in ("dense", "dense+graph")))
        for k in DEEP:
            print(f"  {k:>6}{cov[('dense',k)]/n:>12.3f}{cov[('graph',k)]/n:>12.3f}")
        c10 = cov[("graph", 10)] / n
        for k in (50, 100, 200, 500):
            gain = cov[("graph", k)] / n - c10
            print(f"  → top-{k}까지 뽑으면 재순위기가 노릴 수 있는 상한 +{gain*100:.1f}%p")
    print(f"\n({time.time()-t0:.0f}초)")


if __name__ == "__main__":
    main()
