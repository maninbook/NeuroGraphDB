"""D-HOPS — MuSiQue 이득을 **홉 수별로** 가른다. 2라운드가 필요한지 재기만 한다.

PropRAG과의 격차가 거의 전부 MuSiQue에서 나온다(74.51 vs 78.3, -3.8%p).
2Wiki는 이미 넘었고(94.92 vs 94.1) HotpotQA는 -1.1이다.

MuSiQue는 단일홉 질문을 **2~4개 합성**해서 만든다. 그런데 우리는 후속 질의를
**한 번만** 쓴다. 3홉·4홉 질문은 한 번으로 부족할 수밖에 없다.

짓기 전에 잰다:
  A. 정답 문단 개수별 분포와 **각 층의 기준 Recall@5**
  B. 후속 질의 이득이 층별로 어떻게 갈리는가
     2홉에서만 오르고 3~4홉이 0이면 → 2라운드가 답이다
     3~4홉에서도 이미 오르면 → 다른 데를 봐야 한다
  C. 각 층의 **천장** — 그 층에서 더 얻을 게 남아 있는가
     (상위 5 밖에 있는 정답이 상위 20 안에는 있는가 = 순위 문제인가 회수 문제인가)

캐시된 후속 질의(llama3.1:8b)를 그대로 쓴다. 무료.
"""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from ngdb import BM25, DenseIndex, Graph

RESULTS_REPO = "goethe0101/neurographdb-results"
OFFICIAL = "osunlp/HippoRAG_v2"
EMBED_TAG = "NV-Embed-v2"
DEPTH, WIDTH, NSEED, SCORE_MODE = 3, 4, 5, 0
MAXK, MAXW, A = 20, 6, 4          # A=4 는 E5에서 dev로 고른 예산
CACHE = Path(__file__).parent / "cache" / "musique_llama31.json"
_N = re.compile(r"[^a-z0-9 ]+")
norm = lambda s: " ".join(_N.sub(" ", s.lower()).split())


def main():
    t0 = time.time()
    corpus = json.load(open(hf_hub_download(OFFICIAL, "musique_corpus.json", repo_type="dataset")))
    qs = json.load(open(hf_hub_download(OFFICIAL, "musique.json", repo_type="dataset")))
    titles = [c["title"] for c in corpus]
    texts = [c["text"] for c in corpus]
    nz = lambda s: " ".join(s.split())
    by_text = {}
    for i, c in enumerate(corpus):
        by_text.setdefault(nz(c["text"]), i)
    questions = [{"q": r["question"],
                  "gold": sorted({by_text[nz(p["paragraph_text"])] for p in r["paragraphs"]
                                  if p.get("is_supporting") and nz(p["paragraph_text"]) in by_text})}
                 for r in qs]
    n = len(titles)

    P = np.load(hf_hub_download(RESULTS_REPO, f"emb/musique_{EMBED_TAG}_P.npy",
                                repo_type="dataset")).astype(np.float32)
    Q = np.load(hf_hub_download(RESULTS_REPO, f"emb/musique_{EMBED_TAG}_Q.npy",
                                repo_type="dataset")).astype(np.float32)
    follow = json.loads(CACHE.read_text())["followups"]

    dense = DenseIndex(P.shape[1]); dense.add_batch(P)
    lookup = {}
    for i, t in enumerate(titles):
        k = norm(t)
        if len(k) >= 5:
            lookup.setdefault(k, i)
    es = set()
    for i, body in enumerate(texts):
        toks = norm(body).split()
        for w in range(1, MAXW + 1):
            for a in range(len(toks) - w + 1):
                j = lookup.get(" ".join(toks[a:a + w]))
                if j is not None and j != i:
                    es.add((i, j))
    g = Graph(n); g.add_edges([(a, b, 1.0) for a, b in es])
    bm = BM25()
    for t, b in zip(titles, texts):
        bm.add(t + " " + b)
    bm.finalize()
    QSIM = (P @ Q.T).astype(np.float32)
    print(f"[{time.time()-t0:6.1f}s] 색인 완료 (엣지 {g.n_edges:,})", flush=True)

    hop1 = []
    for i in range(len(questions)):
        h = dense.search(Q[i], MAXK)
        bs = g.beam_search([x for x, _ in h[:NSEED]], QSIM[:, i].tolist(),
                           DEPTH, WIDTH, 0.0, MAXK, SCORE_MODE)
        seen, out = set(), []
        for x, _ in bs:
            if x not in seen:
                seen.add(x); out.append(x)
        for x, _ in h[:MAXK]:
            if x not in seen:
                seen.add(x); out.append(x)
        hop1.append(out[:MAXK])
    hop2 = [[x for x, _ in (bm.search(f, MAXK) if f and not f.upper().startswith("NONE") else [])]
            for f in follow]

    def merged(i, a):
        out = list(hop1[i][:a])
        for x in hop2[i]:
            if len(out) >= 5:
                break
            if x not in out:
                out.append(x)
        for x in hop1[i][a:]:
            if len(out) >= 5:
                break
            if x not in out:
                out.append(x)
        return out[:5]

    strata = {}
    for i, r in enumerate(questions):
        k = len(r["gold"])
        if k:
            strata.setdefault(k, []).append(i)

    print(f"\n{'='*76}\nA/B — 홉 수(정답 문단 개수)별 분포와 이득\n{'='*76}")
    print(f"  {'정답수':>6}{'질문':>7}{'기준R@5':>10}{'a=4':>9}{'차이':>9}{'회수':>7}{'손실':>7}")
    tot_g = tot_l = 0
    for k in sorted(strata):
        idx = strata[k]
        b = sum(len(set(hop1[i][:5]) & set(questions[i]["gold"])) / k for i in idx) / len(idx) * 100
        v = sum(len(set(merged(i, A)) & set(questions[i]["gold"])) / k for i in idx) / len(idx) * 100
        gn = sum(len(set(merged(i, A)) & set(questions[i]["gold"])
                     - set(hop1[i][:5])) for i in idx)
        ls = sum(len(set(hop1[i][:5]) & set(questions[i]["gold"])
                     - set(merged(i, A))) for i in idx)
        tot_g += gn; tot_l += ls
        print(f"  {k:>6}{len(idx):>7}{b:>10.2f}{v:>9.2f}{v-b:>+9.2f}{gn:>7}{ls:>7}")
    print(f"  {'합계':>6}{sum(len(v) for v in strata.values()):>7}"
          f"{'':>10}{'':>9}{'':>9}{tot_g:>7}{tot_l:>7}")

    print(f"\n{'='*76}\nC — 층별 천장. 상위5 밖 정답이 상위20 안에 있는가\n{'='*76}")
    print(f"  {'정답수':>6}{'빠진정답':>10}{'상위20안':>10}{'(순위문제)':>12}"
          f"{'상위20밖':>10}{'(회수문제)':>12}")
    for k in sorted(strata):
        idx = strata[k]
        miss = inside = outside = 0
        for i in idx:
            gold = set(questions[i]["gold"])
            m = gold - set(merged(i, A))
            miss += len(m)
            top20 = set(hop1[i]) | set(hop2[i][:MAXK])
            inside += len(m & top20)
            outside += len(m - top20)
        print(f"  {k:>6}{miss:>10}{inside:>10}{inside/max(miss,1)*100:>11.1f}%"
              f"{outside:>10}{outside/max(miss,1)*100:>11.1f}%")

    print(f"\n  ※ 순위 문제 = 후보엔 있는데 상위5에 못 넣음 (칸 5개 제약)")
    print(f"     회수 문제 = 후보에도 없음 → 2라운드 후속 질의가 필요한 자리")


if __name__ == "__main__":
    main()
