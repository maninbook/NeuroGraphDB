# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "huggingface_hub>=0.28"]
# ///
"""안 이어지는 근거쌍을 **왜** 안 이어지는지 분류하고 실물을 뽑아 본다.

진단이 여기까지 왔다: 우리 병목은 의미 표현이 아니라 **연결성**이다.
근거쌍의 39%(Hot)·31%(2W)·63%(MuS)가 어떤 홉으로도 안 닿는다.

그럼 그 쌍들은 왜 안 이어지나. 고칠 수 있는 것과 없는 것을 가른다:

  고칠 수 있음  제목이 5글자 미만이라 사전에서 뺐다 (우리가 만든 규칙)
                제목이 6단어 초과라 n-gram 범위 밖이다 (우리가 만든 규칙)
                성만 나온다 / 약어로 나온다 / 어순이 다르다 (표기 변형)
  못 고침        본문에 어휘 겹침이 아예 없다 (관계 표현 자체가 없다)

앞의 것이 많으면 **엣지 정의를 손봐서 연결을 늘릴 수 있다.**
뒤의 것이 많으면 이 접근의 상한이고, 다른 신호가 필요하다.

분류만 믿지 않고 **실제 제목과 본문 조각을 찍어서 눈으로 확인한다.**

LLM·GPU 불필요.
"""

import re
import sys
import time
from collections import Counter
from itertools import permutations

PARQUET_REV = "refs/convert/parquet"
DATASETS = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor/validation/0000.parquet"),
    "2wiki":    ("framolfese/2WikiMultihopQA", "default/validation/0000.parquet"),
    "musique":  ("dgslibisey/MuSiQue", "default/validation/0000.parquet"),
}
N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SEED = 0
MAXW = 6
_N = re.compile(r"[^a-z0-9 ]+")
SMALL = {"the", "of", "a", "an", "and", "in", "on", "at", "to", "for", "de", "la",
         "el", "il", "der", "die", "das", "van", "von"}


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
        questions.append({"gold": gold, "q": row["question"]})
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


def diagnose(body_norm, title):
    """A의 본문에 B의 제목이 왜 안 잡혔는지."""
    bt = norm(title)
    bw = bt.split()
    if len(bt) < 5:
        return "제목 5글자 미만 — 우리가 사전에서 뺐다", None
    if len(bw) > MAXW:
        return f"제목 {len(bw)}단어 — n-gram 범위(6) 밖", None

    padded = f" {body_norm} "
    if f" {bt} " in padded:
        return "실제로는 본문에 있음 — 분류 버그", None

    content = [w for w in bw if w not in SMALL and len(w) > 2]
    if not content:
        return "제목이 기능어뿐", None
    present = [w for w in content if f" {w} " in padded]

    if len(present) == len(content):
        return "제목 단어가 전부 있으나 붙어있지 않음 — 어순·삽입어", present
    if len(present) >= 2 or (len(content) == 1 and present):
        return f"부분 일치 {len(present)}/{len(content)}단어 — 성만/약칭", present
    if len(present) == 1:
        return f"약한 부분 일치 1/{len(content)}단어", present

    acr = "".join(w[0] for w in bw if len(w) > 2)
    if len(acr) >= 2 and f" {acr} " in padded:
        return "약어로 등장", [acr]
    return "어휘 겹침 없음 — 관계 표현 자체가 없다", None


def window(body, needles, width=70):
    if not needles:
        return body[:width] + "…"
    low = body.lower()
    for nd in needles:
        p = low.find(nd)
        if p >= 0:
            s = max(0, p - width // 2)
            return ("…" if s else "") + body[s:s + width] + "…"
    return body[:width] + "…"


def main():
    t0 = time.time()
    for ds in DATASETS:
        titles, texts, questions = load_pool(ds, N_Q, SEED)
        tidx = {t: i for i, t in enumerate(titles)}
        edges = build_edges(titles, texts)
        body_norm = [norm(x) for x in texts]

        buckets, samples = Counter(), {}
        n_missing = n_total = 0
        for q in questions:
            g = [tidx[t] for t in q["gold"] if t in tidx]
            if len(g) < 2:
                continue
            n_total += 1
            if any((a, b) in edges for a, b in permutations(g, 2)):
                continue
            n_missing += 1
            # 양방향 중 "가장 가까운" 설명을 고른다
            best, best_rank, best_pair = None, 99, None
            order = ["실제로는", "제목 단어가 전부", "부분 일치", "약한 부분",
                     "약어로", "제목 5글자", "제목 ", "제목이 기능어", "어휘 겹침"]
            for a, b in permutations(g, 2):
                why, hit = diagnose(body_norm[a], titles[b])
                rank = next((i for i, p in enumerate(order) if why.startswith(p)), 98)
                if rank < best_rank:
                    best, best_rank, best_pair = (why, hit), rank, (a, b)
            why, hit = best
            key = why.split(" — ")[0]
            key = re.sub(r"\d+/\d+단어", "n/m단어", key)
            buckets[key] += 1
            if key not in samples or len(samples[key]) < 3:
                a, b = best_pair
                samples.setdefault(key, []).append(
                    (titles[b], titles[a], window(texts[a], hit)))

        print(f"\n{'='*78}\n{ds}  근거쌍 {n_total}개 중 **안 이어진 {n_missing}개** "
              f"({n_missing/max(n_total,1):.1%})\n{'='*78}")
        for key, c in buckets.most_common():
            print(f"  {c:>4}  ({c/max(n_missing,1):>5.1%})  {key}")
        print()
        for key, c in buckets.most_common(4):
            print(f"  [{key}]")
            for tgt, src, win in samples[key][:2]:
                print(f"    찾는 제목: {tgt!r}")
                print(f"    있는 문단: {src!r}")
                print(f"      본문: {win}")
            print()
    print(f"({time.time()-t0:.0f}초, LLM·GPU 없음)")


if __name__ == "__main__":
    main()
