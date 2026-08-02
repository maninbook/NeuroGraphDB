# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11", "numpy", "pyarrow",
#     "transformers>=4.40", "torch", "huggingface_hub>=0.28",
# ]
# ///
"""사전 측정 — 측면 억제(탈상관)로 얻을 게 있는가, 그리고 상한은 얼마인가.

앞서 시도한 "허브 억제"는 차수 기준이었고 데이터셋마다 부호가 뒤집혀 걸렀다.
**실제 측면 억제는 차수가 아니라 유사도 기준이다.** 피질에서 서로를 억제하는 것은
연결이 많은 뉴런이 아니라 **비슷한 것에 반응하는 뉴런**이고, 목적은 탈상관이다.

우리 순위에는 다양성 항이 없다. 확산이 한 개체 주변 문단을 한꺼번에 끌어올리면
top-10이 거의 같은 내용으로 찬다. 그런데 우리 지표(근거 전부가 상위 10에)는
본질적으로 **커버리지**다. 중복 문단은 칸만 먹고 커버리지에 기여하지 않는다.

**상한부터 잰다.** 재배열로 고칠 수 있는 질문이 몇 개인지 모르면 실험할 이유가 없다.

  1. 재배열 상한 — 정답이 top-20에는 다 있는데 top-10에 못 든 질문의 비율.
     **이게 MMR이든 뭐든 상위 20개를 재배열하는 모든 방법의 하드 상한이다.**
  2. 그 질문들의 top-10이 실제로 중복돼 있나 (쌍별 유사도).
     중복이 없으면 뺄 게 없으니 탈상관은 무의미하다.
  3. 밀려난 정답이 중복 덩어리보다 아래에 있나 — 교체가 실제로 가능한 배치인가.

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
MAXK = 20
MAXW = 6

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
        edges = build_edges(titles, texts)
        P = encode([f"{t}. {x}" for t, x in zip(titles, texts)])
        dense = DenseIndex(P.shape[1]); dense.add_batch(P)
        Q = encode([QUERY_PREFIX + q["q"] for q in questions])
        g = Graph(len(titles)); g.add_edges(edges)

        ok10 = fixable = unreachable = 0
        red_fix, red_ok = [], []      # top-10 안의 최대 쌍별 유사도
        gold_sim, mixed_sim = [], []  # 정답끼리 vs 정답-오답 유사도
        miss_rank, blockers = [], []

        for i, q in enumerate(questions):
            hits = dense.search(Q[i], MAXK)
            sh = hits[:N_SEEDS]
            acts = g.spread([d for d, _ in sh], [float(s) for _, s in sh],
                            3, 0.65, 0.02, MAXK)
            seen, order = set(), []
            for d, _ in acts:
                if d not in seen:
                    seen.add(d); order.append(d)
            for d, _ in hits:
                if d not in seen:
                    seen.add(d); order.append(d)
            order = order[:MAXK]
            gold = [tidx[t] for t in q["gold"] if t in tidx]

            top10 = order[:10]
            got10 = all(gv in top10 for gv in gold)
            got20 = all(gv in order for gv in gold)

            # top-10 안의 중복도 — 가장 비슷한 쌍의 유사도
            if len(top10) >= 2:
                S = P[top10] @ P[top10].T
                np.fill_diagonal(S, -1.0)
                mx = float(S.max())
            else:
                mx = 0.0

            if got10:
                ok10 += 1
                red_ok.append(mx)
            elif got20:
                fixable += 1
                red_fix.append(mx)
                # 밀려난 정답의 순위와, 그보다 위에 있는 중복 쌍 개수
                for gv in gold:
                    if gv not in top10 and gv in order:
                        miss_rank.append(order.index(gv) + 1)
                dup = 0
                for a in range(len(top10)):
                    for b in range(a + 1, len(top10)):
                        if float(P[top10[a]] @ P[top10[b]]) > 0.85:
                            dup += 1
                blockers.append(dup)
            else:
                unreachable += 1

            if len(gold) >= 2:
                gold_sim.append(float(P[gold[0]] @ P[gold[1]]))
            wrong = [d for d in top10 if d not in gold]
            if gold and wrong:
                mixed_sim.append(float(np.max(P[gold] @ P[wrong].T)))

        n = len(questions)
        print(f"\n{'='*76}\n{ds}  질문 {n}  풀 {len(titles)}\n{'='*76}")
        print(f"  이미 top-10 안에 정답 전부      {ok10:>5} ({ok10/n:>5.1%})")
        print(f"  **재배열로 고칠 수 있음**       {fixable:>5} ({fixable/n:>5.1%})"
              f"   ← 상위20 재배열의 하드 상한")
        print(f"  top-20에도 없음 (재배열 불가)   {unreachable:>5} ({unreachable/n:>5.1%})")

        if red_fix:
            print(f"\n  top-10 안의 최대 쌍별 유사도 — 뺄 중복이 있나")
            print(f"    고칠 수 있는 질문  평균 {np.mean(red_fix):.3f}  "
                  f">0.85인 비율 {100*np.mean(np.array(red_fix)>0.85):.1f}%")
            print(f"    이미 맞은 질문    평균 {np.mean(red_ok):.3f}  "
                  f">0.85인 비율 {100*np.mean(np.array(red_ok)>0.85):.1f}%")
            r = np.mean(red_fix) / max(np.mean(red_ok), 1e-9)
            print(f"    비 {r:.3f} → "
                  f"{'실패 질문이 더 중복돼 있다. 탈상관이 맞는 방향' if r > 1.02 else ''}"
                  f"{'중복도에 차이가 없다. 탈상관으로 얻을 게 없다' if 0.98 <= r <= 1.02 else ''}"
                  f"{'실패 질문이 오히려 덜 중복돼 있다' if r < 0.98 else ''}")
            print(f"    고칠 수 있는 질문의 top-10 내 유사도>0.85 쌍 수: "
                  f"평균 {np.mean(blockers):.2f}개")
            print(f"    밀려난 정답의 실제 순위: 중앙 {np.median(miss_rank):.0f}, "
                  f"11~13위 비율 {100*np.mean((np.array(miss_rank)<=13)):.0f}%")
        if gold_sim and mixed_sim:
            print(f"\n  정답끼리 유사도 {np.mean(gold_sim):.3f} vs "
                  f"정답-오답 최대 유사도 {np.mean(mixed_sim):.3f}")
            print(f"    → {'정답끼리는 서로 다르다. 다양성 항이 정답을 안 깎는다' if np.mean(gold_sim) < np.mean(mixed_sim) else '정답끼리도 비슷하다. 다양성 항이 정답을 깎을 수 있다'}")
    print(f"\n({time.time()-t0:.0f}초)")


if __name__ == "__main__":
    main()
