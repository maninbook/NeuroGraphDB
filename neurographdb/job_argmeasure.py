# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "spacy>=3.7", "huggingface_hub>=0.28"]
# ///
"""사전 측정 — 문장 단위 명제의 **논항 공유**가 근거쌍을 이어주는가.

배경: 지금 우리 엣지는 "문단 A 본문에 문단 B의 제목이 있다"를 요구한다.
      MuSiQue는 근거쌍 연결률이 32.4%뿐이라 갈 길이 없었고, 그래서 졌다.
      문장 단위 명제는 **제목 언급 없이 논항만 겹쳐도** 이어질 수 있다.

이 스크립트는 실험이 아니라 **실험을 할 가치가 있는지**를 먼저 본다.
연결률이 안 오르면 돌릴 이유가 없다.

**연결률만 보면 안 된다.** 전부 이어버리면 100%가 나오지만 그래프는 쓸모가 없다.
그래서 무작위쌍 연결률(=그래프 밀도)을 같이 재고 **lift = 정답쌍/무작위쌍**을 본다.
lift가 낮으면 신호가 아니라 잡음이다.

세 가지 논항 정의를 비교한다 — 느슨할수록 연결률은 오르고 lift는 떨어진다.
    all    명사구 전부
    propn  고유명사를 포함한 명사구만
    propn6 고유명사 포함 + 6글자 이상

LLM·GPU 불필요.
"""

import random
import re
import subprocess
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
N_RANDOM_PAIRS = 200_000
_NORM = re.compile(r"[^a-z0-9 ]+")
STOP = {"he", "she", "it", "they", "him", "her", "them", "this", "that", "these",
        "those", "who", "which", "what", "his", "hers", "its", "their", "the film",
        "the movie", "the book", "the band", "the song", "the album", "the city",
        "the game", "the show", "the series", "the team", "the group", "the company"}


def norm(s):
    return " ".join(_NORM.sub(" ", s.lower()).split())


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


def title_edges(titles, texts, max_title_words=6):
    """현재 방식 — 비교 기준선."""
    lookup = {}
    for i, t in enumerate(titles):
        k = norm(t)
        if len(k) >= 5:
            lookup.setdefault(k, i)
    edges = set()
    for i, body in enumerate(texts):
        toks = norm(body).split()
        for n in range(1, max_title_words + 1):
            for a in range(len(toks) - n + 1):
                j = lookup.get(" ".join(toks[a:a + n]))
                if j is not None and j != i:
                    edges.add((i, j))
    return edges


def extract_args(doc):
    """문장마다 명제를 뽑고 그 **논항**을 모은다. (술어는 여기서 안 쓴다)

    명제 = (주어, 동사, 목적어). 논항은 그 주어/목적어가 속한 명사구다.
    대명사와 "the film" 같은 총칭구는 버린다 — 이런 게 남으면 전부 이어져버린다.
    """
    chunk_of, chunk_propn, chunk_txt = {}, {}, {}
    for ch in doc.noun_chunks:
        has_propn = any(t.pos_ == "PROPN" for t in ch)
        for t in ch:
            chunk_of[t.i] = ch.start
        chunk_propn[ch.start] = has_propn
        chunk_txt[ch.start] = norm(ch.text)

    args_all, args_propn, args_propn6 = set(), set(), set()
    for tok in doc:
        if tok.pos_ not in ("VERB", "AUX"):
            continue
        cand = []
        for c in tok.children:
            if c.dep_ in ("nsubj", "nsubjpass", "dobj", "attr", "oprd"):
                cand.append(c)
            elif c.dep_ == "prep":
                cand += [g for g in c.children if g.dep_ == "pobj"]
        for c in cand:
            k = chunk_of.get(c.i)
            if k is None:
                continue
            txt = chunk_txt[k]
            if len(txt) < 3 or txt in STOP:
                continue
            args_all.add(txt)
            if chunk_propn[k]:
                args_propn.add(txt)
                if len(txt) >= 6:
                    args_propn6.add(txt)
    return args_all, args_propn, args_propn6


def main():
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                   check=True, stdout=subprocess.DEVNULL)
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    rng = random.Random(SEED)

    for ds in DATASETS:
        titles, texts, questions = load_pool(ds, N_Q, SEED)
        tidx = {t: i for i, t in enumerate(titles)}
        n = len(titles)

        A = [set() for _ in range(n)]
        P = [set() for _ in range(n)]
        P6 = [set() for _ in range(n)]
        for i, doc in enumerate(nlp.pipe(texts, batch_size=64)):
            A[i], P[i], P6[i] = extract_args(doc)

        te = title_edges(titles, texts)

        def linked_title(a, b):
            return (a, b) in te or (b, a) in te

        schemes = {"제목언급(현재)": None, "논항 all": A, "논항 propn": P,
                   "논항 propn6": P6}

        gold_pairs = []
        for q in questions:
            g = [tidx[t] for t in q["gold"] if t in tidx]
            if len(g) >= 2:
                gold_pairs.append(g)

        rand_pairs = [(rng.randrange(n), rng.randrange(n)) for _ in range(N_RANDOM_PAIRS)]
        rand_pairs = [(a, b) for a, b in rand_pairs if a != b]

        print(f"\n{ds}  풀 {n}  질문 {len(gold_pairs)}")
        print(f"  {'연결 정의':<16}{'정답쌍':>9}{'무작위쌍':>11}{'lift':>10}"
              f"{'논항/문단':>11}")
        for name, S in schemes.items():
            if S is None:
                hit = sum(1 for g in gold_pairs
                          if any(linked_title(a, b) for a, b in permutations(g, 2)))
                rnd = sum(1 for a, b in rand_pairs if linked_title(a, b))
                avg = ""
            else:
                hit = sum(1 for g in gold_pairs
                          if any(S[a] & S[b] for a, b in permutations(g, 2)))
                rnd = sum(1 for a, b in rand_pairs if S[a] & S[b])
                avg = f"{sum(len(x) for x in S)/n:.1f}"
            gr = hit / max(len(gold_pairs), 1)
            rr = rnd / max(len(rand_pairs), 1)
            lift = gr / rr if rr > 0 else float("inf")
            print(f"  {name:<16}{gr:>8.1%}{rr:>11.3%}{lift:>10.0f}{avg:>11}")

    print(f"\n({time.time()-t0:.0f}초, LLM·GPU 없음)")


if __name__ == "__main__":
    main()
