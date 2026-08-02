# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "huggingface_hub>=0.28"]
# ///
"""근거쌍이 **몇 홉 만에** 이어지는가 — 진단을 가른다.

앞선 측정에서 MuSiQue 근거쌍의 직접 연결(1홉)은 32.4%였다.
그걸 보고 "경로가 없어서 진다"고 읽었는데, 우리 확산은 **3홉까지** 간다.
A와 B가 개체 C를 공유하면 A→C→B로 2홉이면 닿는다.

그러면 문제가 둘 중 하나다:
  (a) 정말 경로가 없다        → 엣지 정의를 바꿔야 한다
  (b) 경로는 있는데 순위가 안 오른다 → 감쇠·점수를 손봐야 한다

2·3홉 도달률을 재면 갈린다. 도달률이 높은데 성능이 안 나오면 (b)다.

무작위쌍 도달률도 같이 잰다. 3홉이면 거의 다 닿을 수 있는데,
그러면 도달 자체는 의미가 없고 **순위**만 남는다.

LLM·GPU 불필요.
"""

import random
import re
import sys
import time
from collections import deque
from itertools import permutations

PARQUET_REV = "refs/convert/parquet"
DATASETS = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor/validation/0000.parquet"),
    "2wiki":    ("framolfese/2WikiMultihopQA", "default/validation/0000.parquet"),
    "musique":  ("dgslibisey/MuSiQue", "default/validation/0000.parquet"),
}
N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SEED = 0
N_RAND = 3000
_N = re.compile(r"[^a-z0-9 ]+")


def norm(s):
    return " ".join(_N.sub(" ", s.lower()).split())


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
        questions.append({"gold": gold})
    return list(pool), [pool[t] for t in pool], questions


def build_adj(titles, texts, max_title_words=6):
    lookup = {}
    for i, t in enumerate(titles):
        k = norm(t)
        if len(k) >= 5:
            lookup.setdefault(k, i)
    adj = [[] for _ in titles]
    seen = set()
    for i, body in enumerate(texts):
        toks = norm(body).split()
        for n in range(1, max_title_words + 1):
            for a in range(len(toks) - n + 1):
                j = lookup.get(" ".join(toks[a:a + n]))
                if j is not None and j != i and (i, j) not in seen:
                    seen.add((i, j)); adj[i].append(j)
    return adj


def hop_dist(adj, src, targets, max_depth=3):
    """src에서 각 target까지 최단 홉. 확산과 같은 방향(단방향)으로 잰다."""
    need = set(targets)
    out = {}
    dist = {src: 0}
    q = deque([src])
    while q and need:
        u = q.popleft()
        if dist[u] >= max_depth:
            continue
        for v in adj[u]:
            if v in dist:
                continue
            dist[v] = dist[u] + 1
            if v in need:
                out[v] = dist[v]; need.discard(v)
            q.append(v)
    return out


def main():
    t0 = time.time()
    rng = random.Random(SEED)
    for ds in DATASETS:
        titles, texts, questions = load_pool(ds, N_Q, SEED)
        tidx = {t: i for i, t in enumerate(titles)}
        adj = build_adj(titles, texts)
        n = len(titles)

        gold_sets = []
        for q in questions:
            g = [tidx[t] for t in q["gold"] if t in tidx]
            if len(g) >= 2:
                gold_sets.append(g)

        def reach_rate(pairs_iter, total):
            """근거들이 서로 d홉 이내로 닿는 질문 비율 (방향 무관)."""
            cnt = {1: 0, 2: 0, 3: 0}
            for g in pairs_iter:
                best = 99
                for a, b in permutations(g, 2):
                    d = hop_dist(adj, a, [b]).get(b)
                    if d is not None:
                        best = min(best, d)
                for k in (1, 2, 3):
                    cnt[k] += best <= k
            return {k: cnt[k] / max(total, 1) for k in cnt}

        gold = reach_rate(gold_sets, len(gold_sets))
        rand_sets = [[rng.randrange(n), rng.randrange(n)] for _ in range(N_RAND)]
        rand_sets = [g for g in rand_sets if g[0] != g[1]]
        rnd = reach_rate(rand_sets, len(rand_sets))

        print(f"\n{ds}  풀 {n}  질문 {len(gold_sets)}  엣지 {sum(len(a) for a in adj):,}")
        print(f"  {'':<10}{'1홉':>9}{'2홉':>9}{'3홉':>9}")
        print(f"  {'정답쌍':<10}{gold[1]:>8.1%}{gold[2]:>9.1%}{gold[3]:>9.1%}")
        print(f"  {'무작위쌍':<10}{rnd[1]:>8.1%}{rnd[2]:>9.1%}{rnd[3]:>9.1%}")
        lift3 = gold[3] / rnd[3] if rnd[3] > 0 else float("inf")
        print(f"  3홉 lift {lift3:.1f}배")
    print(f"\n({time.time()-t0:.0f}초, LLM·GPU 없음)")


if __name__ == "__main__":
    main()
