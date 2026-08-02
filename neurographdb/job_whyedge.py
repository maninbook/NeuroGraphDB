# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "huggingface_hub>=0.28"]
# ///
"""왜 2Wiki에서 이기고 MuSiQue에서 지는가 — 사후 설명을 숫자로 검증한다.

가설: 우리 엣지는 "문단 A의 본문에 문단 B의 제목이 있다"이다.
      2Wiki는 Wikidata 관계에서 템플릿 생성돼 **근거 문단 쌍이 거의 항상
      직접 제목 언급 관계**다. 그래서 dense가 1홉만 찾아주면 한 홉에 2홉째가 온다.
      MuSiQue는 서로 다른 문서의 단문 질문을 합성해 만들어 그 연결이 없다.

검증: 질문마다 근거 문단들이 우리 엣지로 서로 **직접 연결돼 있는지** 센다.
      가설이 맞으면 2Wiki가 압도적으로 높고 MuSiQue가 낮아야 한다.
      아니면 내 설명이 틀린 것이고 그렇게 적는다.

LLM도 임베딩도 필요 없다. cpu로 충분하다.
"""

import re
import sys
import time
from itertools import permutations

PARQUET_REV = "refs/convert/parquet"
DATASETS = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor/validation/0000.parquet"),
    "2wiki":    ("framolfese/2WikiMultihopQA", "default/validation/0000.parquet"),
    "musique":  ("dgslibisey/MuSiQue", "default/validation/0000.parquet"),
}
N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SEED = 0


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
    titles = list(pool)
    return titles, [pool[t] for t in titles], questions


def build_mention_edges(titles, texts, max_title_words=6):
    norm = lambda s: re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    lookup = {}
    for i, t in enumerate(titles):
        key = " ".join(norm(t).split())
        if len(key) >= 5:
            lookup.setdefault(key, i)
    edges = set()
    for i, body in enumerate(texts):
        toks = norm(body).split()
        for n in range(1, max_title_words + 1):
            for a in range(len(toks) - n + 1):
                j = lookup.get(" ".join(toks[a:a + n]))
                if j is not None and j != i:
                    edges.add((i, j))
    return edges


def main():
    t0 = time.time()
    print(f"{'데이터셋':<10}{'근거쌍 직접연결':>16}{'양방향':>10}{'근거2개':>10}"
          f"{'엣지/노드':>11}")
    for ds in DATASETS:
        titles, texts, questions = load_pool(ds, N_Q, SEED)
        tidx = {t: i for i, t in enumerate(titles)}
        edges = build_mention_edges(titles, texts)

        linked = both = pairs_total = 0
        n_two = 0
        for q in questions:
            g = [tidx[t] for t in q["gold"] if t in tidx]
            if len(g) < 2:
                continue
            n_two += 1
            # 근거들 사이에 방향 무관 엣지가 하나라도 있으면 "연결됨"
            any_link = any((a, b) in edges for a, b in permutations(g, 2))
            both_link = any((a, b) in edges and (b, a) in edges
                            for a, b in permutations(g, 2))
            linked += any_link
            both += both_link
            pairs_total += 1

        print(f"{ds:<10}{linked/max(pairs_total,1):>15.1%}{both/max(pairs_total,1):>10.1%}"
              f"{n_two:>10}{len(edges)/len(titles):>11.2f}")
    print(f"\n({time.time()-t0:.0f}초, LLM·GPU 없음)")


if __name__ == "__main__":
    main()
