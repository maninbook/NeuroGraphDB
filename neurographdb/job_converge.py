# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11", "numpy", "pyarrow",
#     "transformers>=4.40", "torch", "huggingface_hub>=0.28",
# ]
# ///
"""사전 측정 — 수렴 증거를 합산하면 바뀔 게 있는가.

지금 확산은 여러 경로로 도달해도 **최댓값만 남긴다**(graph.cpp의 `cand <= act[v]`).
seed 3개에서 동시에 도달한 문단이 1개에서 도달한 것과 점수가 같다.
실제 뉴런은 시냅스 입력을 **적분**한다. ACT-R의 활성 방정식도 합이다:

    A_i = B_i + Σ_j W_j · S_ji ,  S_ji = S − ln(fan_j)

**합산이 의미가 있으려면 정답 문단이 잡음 문단보다 더 많은 seed에서 도달해야 한다.**
둘 다 대부분 seed 1개에서만 도달하면 합산해도 순위가 안 바뀐다.
그걸 먼저 잰다. 실험을 짜는 건 이 숫자를 보고 나서다.

fan 할인도 같이 본다 — 정답에 이르는 경로의 출처 fan이 잡음 경로보다 작은가.

LLM 불필요(임베딩만).
"""

import os
import re
import subprocess
import sys
import sysconfig
import time
from collections import defaultdict
from pathlib import Path

CODE_REPO = "goethe0101/neurographdb"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
N_SEEDS = 5
MAXK = 20
MAXW = 6
DEPTH, DECAY, MINACT = 3, 0.65, 0.02

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


def build_adj(titles, texts):
    lookup = {}
    for i, t in enumerate(titles):
        k = norm(t)
        if len(k) >= 5:
            lookup.setdefault(k, i)
    adj = defaultdict(set)
    for i, body in enumerate(texts):
        toks = norm(body).split()
        for n in range(1, MAXW + 1):
            for a in range(len(toks) - n + 1):
                j = lookup.get(" ".join(toks[a:a + n]))
                if j is not None and j != i:
                    adj[i].add(j)
    return {k: sorted(v) for k, v in adj.items()}


def reach_from(adj, src, act0):
    """seed 하나에서 도달하는 노드와 그 활성. 현재 확산과 같은 규칙."""
    act = {src: act0}
    frontier = [(src, 0)]
    while frontier:
        u, d = frontier.pop()
        if d >= DEPTH:
            continue
        for v in adj.get(u, ()):
            cand = act[u] * 1.0 * DECAY
            if cand < MINACT or cand <= act.get(v, 0.0):
                continue
            act[v] = cand
            frontier.append((v, d + 1))
    return act


def main():
    import numpy as np
    t0 = time.time()
    build_core()
    from ngdb import DenseIndex

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
        adj = build_adj(titles, texts)
        fan = {u: len(v) for u, v in adj.items()}

        P = encode([f"{t}. {x}" for t, x in zip(titles, texts)])
        dense = DenseIndex(P.shape[1]); dense.add_batch(P)
        Q = encode([QUERY_PREFIX + q["q"] for q in questions])

        g_conv, n_conv = [], []          # 정답/잡음이 몇 개 seed에서 도달했나
        g_fan, n_fan = [], []            # 도달 경로 출처의 fan
        multi_gold = 0                   # seed 2개 이상에서 도달한 정답이 있는 질문
        n_with_reach = 0
        for i, q in enumerate(questions):
            hits = dense.search(Q[i], MAXK)
            seeds = hits[:N_SEEDS]
            seed_ids = {d for d, _ in seeds}
            per_seed = [reach_from(adj, d, float(s)) for d, s in seeds]

            count = defaultdict(int)
            src_fan = defaultdict(list)
            for si, (a, (sd, _)) in enumerate(zip(per_seed, seeds)):
                for v in a:
                    if v in seed_ids:      # seed 자신은 그래프 이득과 무관
                        continue
                    count[v] += 1
                    src_fan[v].append(fan.get(sd, 0))
            if not count:
                continue
            n_with_reach += 1
            gold_ids = {tidx[t] for t in q["gold"] if t in tidx}
            hit_multi = False
            for v, c in count.items():
                mf = float(np.mean(src_fan[v]))
                if v in gold_ids:
                    g_conv.append(c); g_fan.append(mf)
                    if c >= 2:
                        hit_multi = True
                else:
                    n_conv.append(c); n_fan.append(mf)
            multi_gold += hit_multi

        gc, nc = np.array(g_conv), np.array(n_conv)
        print(f"\n{'='*74}\n{ds}  질문 {len(questions)}  확산이 뭔가 도달한 질문 {n_with_reach}"
              f"\n{'='*74}")
        print(f"  도달한 노드가 **몇 개 seed에서** 왔나 (seed 자신 제외)")
        print(f"    {'정답 문단':<12} n={len(gc):>6}  평균 {gc.mean():.2f}  "
              f"2개이상 {100*(gc>=2).mean():>5.1f}%  3개이상 {100*(gc>=3).mean():>5.1f}%")
        print(f"    {'그 외':<12} n={len(nc):>6}  평균 {nc.mean():.2f}  "
              f"2개이상 {100*(nc>=2).mean():>5.1f}%  3개이상 {100*(nc>=3).mean():>5.1f}%")
        ratio = gc.mean() / max(nc.mean(), 1e-9)
        print(f"    정답/그외 수렴도 = {ratio:.2f}배 → "
              f"{'합산이 정답을 밀어올린다' if ratio > 1.1 else ''}"
              f"{'합산해도 순위가 거의 안 바뀐다' if 0.9 <= ratio <= 1.1 else ''}"
              f"{'합산은 잡음을 밀어올린다' if ratio < 0.9 else ''}")
        print(f"    seed 2개 이상에서 도달한 정답이 있는 질문: "
              f"{multi_gold}/{n_with_reach} ({100*multi_gold/max(n_with_reach,1):.1f}%)")

        gf, nf = np.array(g_fan), np.array(n_fan)
        print(f"\n  도달 경로 출처의 fan (ACT-R의 S_ji = S − ln(fan_j))")
        print(f"    {'정답 경로':<12} 평균 {gf.mean():.2f}")
        print(f"    {'그 외 경로':<12} 평균 {nf.mean():.2f}")
        fr = gf.mean() / max(nf.mean(), 1e-9)
        print(f"    정답/그외 fan = {fr:.2f}배 → "
              f"{'fan 할인이 정답에 유리' if fr < 0.9 else ''}"
              f"{'fan 할인이 정답을 깎는다' if fr > 1.1 else ''}"
              f"{'fan 할인은 무차별' if 0.9 <= fr <= 1.1 else ''}")
    print(f"\n({time.time()-t0:.0f}초)")


if __name__ == "__main__":
    main()
