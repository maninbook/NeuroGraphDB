# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "huggingface_hub>=0.28"]
# ///
"""제목 별칭을 추가하면 연결이 얼마나 늘고 잡음이 얼마나 느는가.

진단에서 나온 것: 위키 제목의 **동음이의 괄호**와 **쉼표 접미사** 때문에
개체가 본문에 멀쩡히 있는데도 매칭이 안 된다.
    'Test (wrestler)'  → 본문에는 "Test"
    'The Shining (film)' → 본문에는 "The Shining"
    'Maryland gubernatorial election, 2018' → 본문에는 "Maryland's 2018 governor's race"

괄호를 떼면 잡히지만, 위키가 괄호를 붙인 **이유가 모호해서**다.
"Spam"만으로 색인하면 엉뚱한 문단이 잔뜩 붙는다. 그래서 **잡음을 같이 재야 한다.**

원 제목 키는 유지하고 별칭 키를 **추가**한다(빼지 않는다). 그래야 기존 매칭을 안 잃는다.

지표는 앞서와 같다 — 정답쌍 연결률 / 무작위쌍 연결률 / lift.
lift가 크게 안 떨어지면서 연결률이 오르면 채택할 가치가 있다.

LLM·GPU 불필요.
"""

import random
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
N_RAND = 200_000
MAXW = 6
_N = re.compile(r"[^a-z0-9 ]+")
_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_COMMA = re.compile(r",\s*[^,]+$")


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


def keys_base(t):
    k = norm(t)
    return [k] if len(k) >= 5 else []


def keys_paren(t):
    out = keys_base(t)
    stripped = _PAREN.sub("", t)
    if stripped != t:
        k = norm(stripped)
        if len(k) >= 5:
            out.append(k)
    return out


def keys_comma(t):
    out = keys_base(t)
    stripped = _COMMA.sub("", t)
    if stripped != t:
        k = norm(stripped)
        if len(k) >= 5:
            out.append(k)
    return out


def keys_both(t):
    s = _PAREN.sub("", t)
    s = _COMMA.sub("", s)
    out = keys_base(t)
    if s != t:
        k = norm(s)
        if len(k) >= 5 and k not in out:
            out.append(k)
    return out


def build_edges(titles, texts, keyfn):
    lookup = {}
    for i, t in enumerate(titles):
        for k in keyfn(t):
            lookup.setdefault(k, i)          # 충돌 시 먼저 온 것을 남긴다
    edges = set()
    for i, body in enumerate(texts):
        toks = norm(body).split()
        for n in range(1, MAXW + 1):
            for a in range(len(toks) - n + 1):
                j = lookup.get(" ".join(toks[a:a + n]))
                if j is not None and j != i:
                    edges.add((i, j))
    return edges


def main():
    t0 = time.time()
    rng = random.Random(SEED)
    schemes = {"현재": keys_base, "+괄호제거": keys_paren,
               "+쉼표제거": keys_comma, "+둘다": keys_both}
    for ds in DATASETS:
        titles, texts, questions = load_pool(ds, N_Q, SEED)
        tidx = {t: i for i, t in enumerate(titles)}
        n = len(titles)
        gold_sets = []
        for q in questions:
            g = [tidx[t] for t in q["gold"] if t in tidx]
            if len(g) >= 2:
                gold_sets.append(g)
        rp = [(rng.randrange(n), rng.randrange(n)) for _ in range(N_RAND)]
        rp = [(a, b) for a, b in rp if a != b]

        n_paren = sum(1 for t in titles if _PAREN.search(t))
        print(f"\n{ds}  풀 {n}  근거쌍 {len(gold_sets)}  "
              f"괄호 달린 제목 {n_paren} ({n_paren/n:.1%})")
        print(f"  {'별칭 규칙':<12}{'정답쌍':>9}{'무작위쌍':>11}{'lift':>9}{'엣지/노드':>11}")
        for name, fn in schemes.items():
            e = build_edges(titles, texts, fn)
            hit = sum(1 for g in gold_sets
                      if any((a, b) in e or (b, a) in e for a, b in permutations(g, 2)))
            rnd = sum(1 for a, b in rp if (a, b) in e or (b, a) in e)
            gr, rr = hit / max(len(gold_sets), 1), rnd / max(len(rp), 1)
            lift = gr / rr if rr > 0 else float("inf")
            print(f"  {name:<12}{gr:>8.1%}{rr:>11.3%}{lift:>9.0f}{len(e)/n:>11.2f}")
    print(f"\n({time.time()-t0:.0f}초, LLM·GPU 없음)")


if __name__ == "__main__":
    main()
