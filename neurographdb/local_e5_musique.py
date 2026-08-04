"""E5-부분 — E4의 MuSiQue 이득(+1.62)을 **dev/test로** 검증한다. 로컬, 무료.

HF Jobs·인퍼런스 크레딧이 소진되어(402) 남은 것만 한다.
MuSiQue 후속 질의는 E4에서 Hub에 캐시해뒀으므로 **재생성 없이** 검증 가능하다.
2Wiki·HotpotQA는 새 LLM 호출이 필요해 규칙 A는 여기서 못 막는다 — 그대로 남겨 둔다.

막는 구멍:
  1. 예산을 dev(짝수 500)에서 고르고 test(홀수 500)에서 잰다
  2. 예산 전부를 dev·test 양쪽에 찍어 작동점의 취약성을 드러낸다
  4. a=5(2홉 미사용)가 안전장치 — 기준선과 정확히 같아야 한다
못 막는 구멍:
  3. 규칙 A — 다른 두 데이터셋을 해치지 않는지. **크레딧 필요.**

프롬프트·설정은 E4와 동일. 한 글자도 바꾸지 않는다.
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from ngdb import BM25, DenseIndex, Graph

RESULTS_REPO = "goethe0101/neurographdb-results"
OFFICIAL = "osunlp/HippoRAG_v2"
EMBED_TAG = "NV-Embed-v2"
DEPTH, WIDTH, NSEED, SCORE_MODE = 3, 4, 5, 0
MAXK, MAXW = 20, 6
BUDGETS = (5, 4, 3, 2)
_N = re.compile(r"[^a-z0-9 ]+")
norm = lambda s: " ".join(_N.sub(" ", s.lower()).split())


def main():
    t0 = time.time()
    corpus = json.load(open(hf_hub_download(OFFICIAL, "musique_corpus.json",
                                            repo_type="dataset")))
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
    follow = json.load(open(hf_hub_download(RESULTS_REPO, "decomp/musique_followups.json",
                                            repo_type="dataset")))["followups"]
    print(f"[{time.time()-t0:6.1f}s] 적재 완료. 문단 {n:,} 질문 {len(questions):,} "
          f"후속질의 {len(follow):,}", flush=True)

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
    print(f"[{time.time()-t0:6.1f}s] 색인 완료. 엣지 {g.n_edges:,} (E4 기록 13,610)", flush=True)

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
    hop2 = [bm.search(f, MAXK) if f and not f.upper().startswith("NONE") else []
            for f in follow]
    print(f"[{time.time()-t0:6.1f}s] 검색 완료", flush=True)

    def r5(idxs, a):
        tot = gain = loss = 0.0
        m = 0
        for i in idxs:
            gold = set(questions[i]["gold"])
            if not gold:
                continue
            m += 1
            out = list(hop1[i][:a])
            for x, _ in hop2[i]:
                if len(out) >= 5:
                    break
                if x not in out:
                    out.append(x)
            for x in hop1[i][a:]:
                if len(out) >= 5:
                    break
                if x not in out:
                    out.append(x)
            new, old = set(out[:5]) & gold, set(hop1[i][:5]) & gold
            tot += len(new) / len(gold)
            gain += len(new - old); loss += len(old - new)
        return tot / m * 100, gain, loss

    nq = len(questions)
    dev, test, full = list(range(0, nq, 2)), list(range(1, nq, 2)), list(range(nq))

    print(f"\n{'='*74}\n안전장치 — a=5(2홉 미사용)가 E4 기준선 72.98을 재현하는가\n{'='*74}")
    print(f"  전체 1000  {r5(full,5)[0]:.2f}")

    print(f"\n{'='*74}\ndev(짝수 500)에서 예산을 고른다\n{'='*74}")
    print(f"  {'1홉칸':>6}{'R@5':>9}{'기준대비':>10}{'회수':>7}{'손실':>7}")
    devs = {}
    b5 = r5(dev, 5)[0]
    for a in BUDGETS:
        v, gn, ls = r5(dev, a)
        devs[a] = v
        print(f"  {a:>6}{v:>9.2f}{v-b5:>+10.2f}{gn:>7.0f}{ls:>7.0f}")
    best = max(devs, key=devs.get)
    print(f"\n  dev 선택: a={best}")

    print(f"\n{'='*74}\ntest(홀수 500) — 예산 전부를 찍어 취약성을 드러낸다\n{'='*74}")
    t5 = r5(test, 5)[0]
    print(f"  {'1홉칸':>6}{'R@5':>9}{'기준대비':>10}{'회수':>7}{'손실':>7}")
    tests = {}
    for a in BUDGETS:
        v, gn, ls = r5(test, a)
        tests[a] = v
        mk = "  ← dev 선택" if a == best else ""
        print(f"  {a:>6}{v:>9.2f}{v-t5:>+10.2f}{gn:>7.0f}{ls:>7.0f}{mk}")

    d = tests[best] - t5
    print(f"\n{'='*74}\n판정\n{'='*74}")
    print(f"  test 기준선(2홉 미사용)  {t5:.2f}")
    print(f"  test dev선택 a={best}       {tests[best]:.2f}  ({d:+.2f}%p)")
    print(f"  dev→test 하락            {devs[best]-tests[best]:+.2f}%p")
    print(f"  HippoRAG 2 = 74.7 / PropRAG = 78.3")
    if d < 0.3:
        print(f"  → **이득이 test에서 살아남지 못했다.** E4의 +1.62는 전체 1000 값이었다")
    elif tests[best] > 74.7:
        print(f"  → test에서 HippoRAG 2를 넘는다. 단 규칙 A(다른 두 데이터셋) 미검증")
    else:
        print(f"  → 이득은 살아남았으나 HippoRAG 2(74.7)에는 못 미친다")
    print(f"\n  ※ 규칙 A는 크레딧 소진으로 검증 못 함. 한 데이터셋 결과만으로는 채택 불가.")

    Path("/tmp/e5_musique.json").write_text(json.dumps(
        {"base_full": r5(full, 5)[0], "dev": devs, "test": tests,
         "chosen": best, "test_gain": d}, indent=2))


if __name__ == "__main__":
    main()
