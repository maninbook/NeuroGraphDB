"""B0 — baseline 측정. 우리가 넘어야 할 선을 숫자로 확정한다.

NeuroDB를 고치기 전에 이걸 먼저 한다. 목표선을 모르면 개선을 평가할 수 없다.

**논문 공개 수치와 비교하지 않는다.** LLM·프롬프트·검색 예산·문서 집합이 다르면
비교가 아니다. 여기서는 baseline을 전부 직접 돌린다.
(기존 benchmark_vs_logicrag.py가 공개 수치를 쓰고 있어 이 파일이 대체한다.)

과제: HotpotQA distractor. 질문마다 정답 근거 문단 2개가 지정돼 있다.
      질문별 10개 문단만 놓고 고르면 너무 쉬우므로 **전체 질문의 문단을 하나로 모아**
      풀(pool)을 만들고 그 안에서 찾게 한다. HippoRAG 계열이 쓰는 설정이다.

지표: recall@k — 정답 근거 문단을 상위 k개 안에 얼마나 담아내는가.
      LLM 호출이 없어 값싸고, 생성 품질에 오염되지 않아 검색 자체를 잰다.

색인·검색·채점은 전부 C++(ngdb)에서 돈다. 파이썬은 데이터 준비와 실행만 한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from ngdb import BM25, DenseIndex

DATASET = "hotpotqa/hotpot_qa"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
# bge 계열은 질의에 접두어를 붙여야 성능이 나온다. 안 붙이면 baseline을 부당하게 깎는다.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
KS = (1, 2, 5, 10, 20)


def load_pool(n_questions: int, seed: int = 0):
    """질문 n개를 뽑고, 그 문단들을 하나의 풀로 합친다."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, "distractor", split="validation")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:n_questions]

    passages: dict[str, str] = {}     # title → text
    questions = []
    for i in idx:
        row = ds[int(i)]
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
            passages.setdefault(title, " ".join(sents))
        gold = sorted(set(row["supporting_facts"]["title"]))
        questions.append({"q": row["question"], "gold": gold, "answer": row["answer"]})

    titles = list(passages)
    return titles, [passages[t] for t in titles], questions


def recall_at(ranked_titles: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    hit = sum(1 for g in gold if g in ranked_titles[:k])
    return hit / len(gold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n-questions", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/baseline_hotpotqa.json")
    args = ap.parse_args()

    t0 = time.time()
    titles, texts, questions = load_pool(args.n_questions, args.seed)
    print(f"[{time.time()-t0:6.1f}s] 질문 {len(questions)}개 | 문단 풀 {len(titles)}개")

    # ── BM25 (C++) ───────────────────────────────────────────────────────────
    bm = BM25()
    for t, x in zip(titles, texts):
        bm.add(f"{t} {x}")          # 제목도 색인한다. 정답 단위가 제목이므로
    bm.finalize()
    print(f"[{time.time()-t0:6.1f}s] BM25 색인 완료 ({bm.size}개)")

    # ── 조밀 벡터 (임베딩은 파이썬, 색인·검색은 C++) ────────────────────────
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()
    P = model.encode([f"{t}. {x}" for t, x in zip(titles, texts)],
                     batch_size=64, convert_to_numpy=True,
                     show_progress_bar=True).astype(np.float32)
    dense = DenseIndex(dim)
    dense.add_batch(P)
    print(f"[{time.time()-t0:6.1f}s] 벡터 색인 완료 ({dense.size}개, {dim}차원)")

    Q = model.encode([QUERY_PREFIX + q["q"] for q in questions],
                     batch_size=64, convert_to_numpy=True,
                     show_progress_bar=True).astype(np.float32)

    # ── 평가 ────────────────────────────────────────────────────────────────
    maxk = max(KS)
    methods = {"bm25": [], "dense": [], "hybrid": []}
    per_q = []

    for i, q in enumerate(questions):
        b_hits = bm.search(q["q"], maxk)
        d_hits = dense.search(Q[i], maxk)
        b_titles = [titles[d] for d, _ in b_hits]
        d_titles = [titles[d] for d, _ in d_hits]

        # 하이브리드 — RRF(Reciprocal Rank Fusion). 표준적이고 튜닝 파라미터가 없다
        rr: dict[int, float] = {}
        for rank, (doc, _) in enumerate(b_hits):
            rr[doc] = rr.get(doc, 0.0) + 1.0 / (60 + rank + 1)
        for rank, (doc, _) in enumerate(d_hits):
            rr[doc] = rr.get(doc, 0.0) + 1.0 / (60 + rank + 1)
        h_titles = [titles[d] for d, _ in sorted(rr.items(), key=lambda x: -x[1])[:maxk]]

        row = {"q": q["q"], "gold": q["gold"]}
        for name, ranked in (("bm25", b_titles), ("dense", d_titles), ("hybrid", h_titles)):
            r = {f"r@{k}": recall_at(ranked, q["gold"], k) for k in KS}
            # 정답 2개를 모두 담았는가 — 멀티홉에서는 이게 실질 성공 조건이다
            r["all@10"] = float(all(g in ranked[:10] for g in q["gold"]))
            methods[name].append(r)
            row[name] = r
        per_q.append(row)

    print(f"\n{'='*66}\nHotpotQA distractor — 질문 {len(questions)} / 풀 {len(titles)}\n{'='*66}")
    header = "방법        " + "".join(f"{'R@'+str(k):>9}" for k in KS) + f"{'정답2개@10':>12}"
    print(header)
    summary = {}
    for name, rows in methods.items():
        vals = {f"r@{k}": float(np.mean([r[f"r@{k}"] for r in rows])) for k in KS}
        vals["all@10"] = float(np.mean([r["all@10"] for r in rows]))
        summary[name] = vals
        line = f"{name:<12}" + "".join(f"{vals[f'r@{k}']:>9.3f}" for k in KS)
        print(line + f"{vals['all@10']:>12.3f}")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps({
        "dataset": "hotpotqa/hotpot_qa distractor validation",
        "n_questions": len(questions), "pool_size": len(titles),
        "seed": args.seed, "embed_model": EMBED_MODEL,
        "summary": summary, "per_question": per_q,
    }, indent=2, ensure_ascii=False))
    print(f"\n저장: {out}  ({time.time()-t0:.1f}초)")


if __name__ == "__main__":
    main()
