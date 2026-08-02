# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "huggingface_hub>=0.28"]
# ///
"""사전 측정 — 위키피디아 전체(BEIR HotpotQA, 520만 문단)에서 우리 방법이 성립하는가.

MTEB의 `SearchProtocol`(index + search)에 우리 시스템이 들어맞고,
MTEB 검색 과제에 BEIR HotpotQA가 있다. 그 코퍼스가 **위키피디아 전체 520만 문단**이다.
LLM 기반 그래프 RAG(문단당 LLM 2회)는 여기에 못 온다 — 우리 측정치로 환산하면 약 52일이다.

**그런데 규모가 되는 것과 방법이 통하는 것은 다르다.** 둘 다 잰다:

  1. 실현 가능성 — 색인 시간, 메모리, 엣지 수. 몇 시간 넘게 걸리면 접는다.
  2. **근거쌍 연결률** — 이게 진짜 관문이다.
     우리 지금까지 측정에서 이 값이 우리 방법의 이득을 예측했다
     (2Wiki 68.8% → +38.6%p, MuSiQue 32.4% → +6.4%p).
     520만 문단 코퍼스에서 이 값이 낮으면 규모가 돼도 이득이 없다.

메모리를 아끼려고 전체 엣지는 **세기만 하고 저장하지 않는다.**
연결률 계산에 필요한 것은 정답 문단끼리의 엣지뿐이므로 그것만 담는다.

LLM·GPU 불필요.
"""

import re
import sys
import time
from collections import defaultdict
from itertools import permutations

PARQUET_REV = "refs/convert/parquet"
CORPUS_REPO = "BeIR/hotpotqa"
QRELS_REPO = "BeIR/hotpotqa-qrels"
MAXW = 6
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0     # 0 = 전체
_N = re.compile(r"[^a-z0-9 ]+")


def norm(s):
    return " ".join(_N.sub(" ", s.lower()).split())


def load_corpus():
    """corpus 설정의 parquet 조각을 순서대로 읽는다. 5.2M이라 파일이 여러 개다."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    # 기본 리비전에 corpus/corpus-00000-of-00001.parquet 하나로 들어 있다.
    # refs/convert/parquet은 경로가 corpus/corpus/0000.parquet으로 달라 섞으면 404가 난다.
    files = sorted(f for f in api.list_repo_files(CORPUS_REPO, repo_type="dataset")
                   if f.startswith("corpus/") and f.endswith(".parquet"))
    print(f"  corpus parquet 조각 {len(files)}개: {files}")
    for f in files:
        p = hf_hub_download(CORPUS_REPO, f, repo_type="dataset")
        t = pq.read_table(p)
        yield t, t.column_names


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    print("1) 코퍼스 적재")
    ids, titles, texts = [], [], []
    for t, cols in load_corpus():
        d = t.to_pydict()
        idc = "_id" if "_id" in cols else cols[0]
        ttc = "title" if "title" in cols else cols[1]
        txc = "text" if "text" in cols else cols[2]
        ids.extend(d[idc]); titles.extend(d[ttc]); texts.extend(d[txc])
        print(f"  [{time.time()-t0:6.1f}s] 누적 {len(ids):,}")
        if LIMIT and len(ids) >= LIMIT:
            ids, titles, texts = ids[:LIMIT], titles[:LIMIT], texts[:LIMIT]
            break
    n = len(ids)
    idx_of = {v: i for i, v in enumerate(ids)}
    print(f"  [{time.time()-t0:6.1f}s] 문단 {n:,}")

    print("\n2) 제목 사전 구축")
    lookup = {}
    dup = 0
    for i, t in enumerate(titles):
        k = norm(t)
        if len(k) >= 5:
            if k in lookup:
                dup += 1
            else:
                lookup[k] = i
    print(f"  [{time.time()-t0:6.1f}s] 고유 제목 키 {len(lookup):,} "
          f"(중복 제목 {dup:,} — 먼저 온 것을 남김)")

    print("\n3) qrels 적재 (정답 문단)")
    qf = hf_hub_download(QRELS_REPO, "default/test/0000.parquet",
                         repo_type="dataset", revision=PARQUET_REV)
    qt = pq.read_table(qf).to_pydict()
    qcol = "query-id" if "query-id" in qt else list(qt)[0]
    dcol = "corpus-id" if "corpus-id" in qt else list(qt)[1]
    gold = defaultdict(list)
    missing = 0
    for q, dd in zip(qt[qcol], qt[dcol]):
        j = idx_of.get(str(dd))
        if j is None:
            missing += 1
        else:
            gold[str(q)].append(j)
    multi = {q: v for q, v in gold.items() if len(v) >= 2}
    print(f"  [{time.time()-t0:6.1f}s] 질의 {len(gold):,} "
          f"(근거 2개 이상 {len(multi):,}, 코퍼스에 없는 정답 {missing:,})")

    gold_ids = {d for v in multi.values() for d in v}
    print(f"  정답 문단 고유 {len(gold_ids):,}")

    print("\n4) 언급 엣지 생성 — 전체는 세기만, 정답끼리만 저장")
    n_edges = 0
    gg = set()
    step = max(n // 20, 1)
    for i, body in enumerate(texts):
        toks = norm(body).split()
        hit = set()
        for w in range(1, MAXW + 1):
            for a in range(len(toks) - w + 1):
                j = lookup.get(" ".join(toks[a:a + w]))
                if j is not None and j != i:
                    hit.add(j)
        n_edges += len(hit)
        if i in gold_ids:
            for j in hit:
                if j in gold_ids:
                    gg.add((i, j))
        if (i + 1) % step == 0:
            el = time.time() - t0
            print(f"  [{el:6.1f}s] {i+1:,}/{n:,} ({(i+1)/n:.0%}) "
                  f"엣지 {n_edges:,} · 남은 예상 {el*(n-i-1)/(i+1)/60:.0f}분")

    linked = sum(1 for v in multi.values()
                 if any((a, b) in gg for a, b in permutations(v, 2)))
    rate = linked / max(len(multi), 1)

    print(f"\n{'='*72}\nBEIR HotpotQA 전체 위키 — 결과\n{'='*72}")
    print(f"  문단            {n:,}")
    print(f"  언급 엣지        {n_edges:,}  (노드당 {n_edges/n:.2f})")
    print(f"  색인 총 시간     {(time.time()-t0)/60:.1f}분")
    print(f"  **근거쌍 연결률  {rate:.1%}**  ({linked:,}/{len(multi):,})")
    print(f"\n  참고 — 우리 소규모 풀에서의 값과 그때의 이득")
    print(f"    2Wiki    68.8%  → dense 대비 +38.6%p")
    print(f"    HotpotQA 61.0%  → +6.6%p (dense가 이미 0.874라 여지가 없었음)")
    print(f"    MuSiQue  32.4%  → +6.4%p")
    print(f"\n  LLM 기반 그래프 RAG였다면: {n/4943*74/60/24:.0f}일 (문단당 LLM 2회 기준)")


if __name__ == "__main__":
    main()
