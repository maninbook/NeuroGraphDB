# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "huggingface_hub>=0.28"]
# ///
"""E1 관문 — 공식 평가셋(osunlp/HippoRAG_v2)에서 근거쌍 연결률을 잰다.

SOTA.md 사전등록: **연결률이 40% 미만이면 E1을 돌리지 않는다.**
우리 측정에서 이 값이 그래프의 이득을 예측해 왔다
(2Wiki 68.8% → +38.6%p / HotpotQA 61.0% → +6.6%p / MuSiQue 32.4% → +6.4%p).

동시에 이 스크립트가 **공식 파일을 우리 형식으로 읽는 경로를 확정**한다.
이후 실험은 전부 이 로더를 쓴다. 자체 풀 구성과 섞이면 또 코퍼스가 갈린다
(그 일이 실제로 있었다 — RESULTS.md 「결과 감사」 참조).

LLM·GPU 불필요.
"""

import json
import random
import re
import sys
import time
from itertools import permutations

REPO = "osunlp/HippoRAG_v2"
RESULTS_REPO = "goethe0101/neurographdb-results"
DATASETS = {"musique": "musique", "2wiki": "2wikimultihopqa", "hotpotqa": "hotpotqa"}
MAXW = 6
N_RAND = 200_000
_N = re.compile(r"[^a-z0-9 ]+")


def norm(s):
    return " ".join(_N.sub(" ", s.lower()).split())


def load_official(name):
    """공식 코퍼스와 질문을 읽어 (titles, texts, questions[{q, gold_titles}])로 만든다.

    정답 표현이 데이터셋마다 다르다:
      musique  paragraphs[{title, is_supporting}]
      hotpot   supporting_facts [[title, sent_idx], ...]
      2wiki    supporting_facts [[title, sent_idx], ...]
    """
    from huggingface_hub import hf_hub_download

    key = DATASETS[name]
    corpus = json.load(open(hf_hub_download(REPO, f"{key}_corpus.json", repo_type="dataset")))
    qs = json.load(open(hf_hub_download(REPO, f"{key}.json", repo_type="dataset")))

    titles = [c["title"] for c in corpus]
    texts = [c["text"] for c in corpus]

    out = []
    for r in qs:
        if name == "musique":
            gold = sorted({p["title"] for p in r["paragraphs"] if p.get("is_supporting")})
        else:
            gold = sorted({sf[0] for sf in r["supporting_facts"]})
        out.append({"q": r["question"], "gold": gold,
                    "answer": str(r.get("answer", ""))})
    return titles, texts, out


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


def main():
    t0 = time.time()
    rng = random.Random(0)
    report = {}
    for name in DATASETS:
        titles, texts, questions = load_official(name)
        tidx = {t: i for i, t in enumerate(titles)}
        edges = build_edges(titles, texts)
        n = len(titles)

        missing = sum(1 for q in questions for g in q["gold"] if g not in tidx)
        gold_sets, drop = [], 0
        for q in questions:
            ids = [tidx[g] for g in q["gold"] if g in tidx]
            if len(ids) != len(q["gold"]):
                drop += 1
            if len(ids) >= 2:
                gold_sets.append(ids)

        linked = sum(1 for g in gold_sets
                     if any((a, b) in edges for a, b in permutations(g, 2)))
        rate = linked / max(len(gold_sets), 1)

        rp = [(rng.randrange(n), rng.randrange(n)) for _ in range(N_RAND)]
        rp = [(a, b) for a, b in rp if a != b]
        rnd = sum(1 for a, b in rp if (a, b) in edges or (b, a) in edges)
        rr = rnd / max(len(rp), 1)
        lift = rate / rr if rr > 0 else float("inf")

        print(f"\n{'='*72}\n{name} — 공식 코퍼스\n{'='*72}")
        print(f"  문단 {n:,} · 질문 {len(questions):,} · 엣지 {len(edges):,} "
              f"(노드당 {len(edges)/n:.2f})")
        print(f"  코퍼스에 없는 정답 제목 {missing} · 근거 일부 유실 질문 {drop}")
        print(f"  근거 2개 이상 질문 {len(gold_sets):,}")
        print(f"  **근거쌍 연결률 {rate:.1%}** · 무작위쌍 {rr:.3%} · lift {lift:.0f}")
        gate = "통과 — E1 진행" if rate >= 0.40 else "미달 — 사전등록대로 E1을 돌리지 않는다"
        print(f"  관문(40%): {gate}")
        report[name] = {"n_passages": n, "n_questions": len(questions),
                        "n_edges": len(edges), "edges_per_node": len(edges)/n,
                        "missing_gold_titles": missing, "questions_with_dropped_gold": drop,
                        "n_multi_gold": len(gold_sets), "gold_pair_linkage": rate,
                        "random_pair_linkage": rr, "lift": lift,
                        "gate_pass": bool(rate >= 0.40)}

    print(f"\n참고 — 우리 자체 풀에서의 값: 2Wiki 68.8% / HotpotQA 61.0% / MuSiQue 32.4%")
    print(f"({time.time()-t0:.0f}초, LLM·GPU 없음)")

    from pathlib import Path
    from huggingface_hub import HfApi
    out = Path("/tmp/official_linkage.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    HfApi().upload_file(path_or_fileobj=str(out), path_in_repo=f"runs/{out.name}",
                        repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"업로드: runs/{out.name}")


if __name__ == "__main__":
    main()
