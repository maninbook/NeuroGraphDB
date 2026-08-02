# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "huggingface_hub>=0.28"]
# ///
"""억제를 넣기 전 사전 측정 — 눌러줄 편중이 실제로 있는가.

아이디어(김기인): 단순 연결보다 **억제**가 필요하다. 어텐션도 결국 점수를 매겨
중요한 것과 아닌 것을 가르는 일이다.

지금 확산에는 억제가 없다. `cand = act[u] × w × 0.65` — u는 이웃이 1개든 30개든
**각각에게 같은 양**을 준다. 예산이 없다. 별칭 엣지로 출차수가 늘자마자
희석 손해가 난 이유다.

억제를 넣을 자리 둘:
  출력 정규화   u의 활성을 이웃 수로 나눠 배분한다 (softmax의 분모에 해당)
  허브 억제     모두가 가리키는 노드를 누른다 (IR의 IDF, 신경과학의 측면 억제)

**둘 다 분포가 편중돼 있을 때만 의미가 있다.** 평평하면 정규화해도 순위가 안 바뀐다.
그래서 먼저 잰다:
  1. 출차수·입차수 분포가 얼마나 치우쳤나
  2. **정답 문단과 잡음 문단의 입차수가 다른가** — 이게 핵심이다.
     정답이 저차수(구체적 개체)이고 잡음이 고차수(총칭 개체)면 허브 억제가 맞다.
     차이가 없으면 허브 억제는 정답도 같이 누른다.

LLM·GPU 불필요.
"""

import re
import sys
import time

PARQUET_REV = "refs/convert/parquet"
DATASETS = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor/validation/0000.parquet"),
    "2wiki":    ("framolfese/2WikiMultihopQA", "default/validation/0000.parquet"),
    "musique":  ("dgslibisey/MuSiQue", "default/validation/0000.parquet"),
}
N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SEED = 0
MAXW = 6
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


def build_edges(titles, texts):
    lookup = {}
    for i, t in enumerate(titles):
        k = norm(t)
        if len(k) >= 5:
            lookup.setdefault(k, i)
    edges = set()
    for i, body in enumerate(texts):
        toks = norm(body).split()
        for n in range(1, MAXW + 1):
            for a in range(len(toks) - n + 1):
                j = lookup.get(" ".join(toks[a:a + n]))
                if j is not None and j != i:
                    edges.add((i, j))
    return edges


def pct(a, p):
    import numpy as np
    return float(np.percentile(a, p)) if len(a) else 0.0


def main():
    import numpy as np
    t0 = time.time()
    for ds in DATASETS:
        titles, texts, questions = load_pool(ds, N_Q, SEED)
        tidx = {t: i for i, t in enumerate(titles)}
        edges = build_edges(titles, texts)
        n = len(titles)

        outd = np.zeros(n, dtype=int)
        ind = np.zeros(n, dtype=int)
        for a, b in edges:
            outd[a] += 1; ind[b] += 1

        gold_ids = set()
        for q in questions:
            for t in q["gold"]:
                if t in tidx:
                    gold_ids.add(tidx[t])
        gold_ids = sorted(gold_ids)
        nongold = [i for i in range(n) if i not in set(gold_ids)]

        # 확산이 실제로 지나가는 노드만 본다 — 입차수 0이면 어차피 도달 불가
        reach = [i for i in range(n) if ind[i] > 0]
        g_reach = [i for i in gold_ids if ind[i] > 0]
        ng_reach = [i for i in nongold if ind[i] > 0]

        print(f"\n{'='*72}\n{ds}  풀 {n}  엣지 {len(edges):,}\n{'='*72}")
        print(f"  출차수  중앙 {pct(outd,50):.0f}  90% {pct(outd,90):.0f}  "
              f"99% {pct(outd,99):.0f}  최대 {outd.max()}  "
              f"(0인 노드 {100*(outd==0).mean():.0f}%)")
        print(f"  입차수  중앙 {pct(ind,50):.0f}  90% {pct(ind,90):.0f}  "
              f"99% {pct(ind,99):.0f}  최대 {ind.max()}  "
              f"(0인 노드 {100*(ind==0).mean():.0f}%)")

        # 상위 1% 허브가 전체 입력 엣지의 몇 %를 먹나
        k = max(1, n // 100)
        top = np.sort(ind)[::-1][:k].sum()
        print(f"  입차수 상위 1%({k}개)가 전체 엣지의 {100*top/max(len(edges),1):.0f}%를 차지")

        print(f"\n  도달 가능한 노드({len(reach)})의 입차수 비교 — 허브 억제가 맞는가")
        print(f"    {'정답 문단':<12} n={len(g_reach):>5}  중앙 {pct(ind[g_reach],50):>5.0f}"
              f"  평균 {ind[g_reach].mean():>6.2f}  90% {pct(ind[g_reach],90):>5.0f}")
        print(f"    {'그 외':<12} n={len(ng_reach):>5}  중앙 {pct(ind[ng_reach],50):>5.0f}"
              f"  평균 {ind[ng_reach].mean():>6.2f}  90% {pct(ind[ng_reach],90):>5.0f}")
        r = ind[g_reach].mean() / max(ind[ng_reach].mean(), 1e-9)
        print(f"    정답/그외 입차수 비 = {r:.2f}배"
              f"  → {'정답이 오히려 허브다. 허브 억제는 정답을 누른다' if r > 1.15 else ''}"
              f"{'정답이 저차수다. 허브 억제가 맞는 방향' if r < 0.85 else ''}"
              f"{'차이 없음. 허브 억제는 무차별이다' if 0.85 <= r <= 1.15 else ''}")

        # 출력 정규화가 바꿀 여지 — 확산 출발점이 될 만한 노드의 출차수 편중
        busy = outd[outd > 0]
        print(f"\n  출차수>0 노드({len(busy)})의 편중 — 출력 정규화가 바꿀 여지")
        print(f"    중앙 {pct(busy,50):.0f}  90% {pct(busy,90):.0f}  최대 {busy.max()}"
              f"  (최대/중앙 = {busy.max()/max(pct(busy,50),1):.0f}배)")
    print(f"\n({time.time()-t0:.0f}초, LLM·GPU 없음)")


if __name__ == "__main__":
    main()
