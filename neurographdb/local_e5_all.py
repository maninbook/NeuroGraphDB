"""E5 — 후속 질의를 **세 데이터셋 전부**에서 검증한다. 로컬 Ollama, 완전 무료.

HF 크레딧이 소진(402)되어 Jobs·인퍼런스가 둘 다 막혔다. 로컬 CPU/Metal로 옮긴다.

── 모델이 바뀐다는 것의 의미 ────────────────────────────────────────────────
E4는 라우터의 Llama-3.1-8B였다. 여기서는 로컬 **gemma3 4.3B Q4_K_M**을 쓴다.
그래서 **MuSiQue도 다시 생성한다.** 데이터셋마다 다른 모델을 쓰면 비교가 성립하지 않는다.
덤으로 검증이 하나 붙는다 — E4의 MuSiQue 이득이 **모델에 의존하는가.**
8B 풀정밀 → 4B 4비트 양자화로 내려가도 살아남으면 그건 기제가 튼튼하다는 뜻이다.

── 막는 구멍 (E4가 못 막은 것 전부) ─────────────────────────────────────────
  1. 예산을 dev(짝수 500)에서 고르고 test(홀수 500)에서 잰다
  2. 예산 전부를 dev·test 양쪽에 찍어 작동점의 취약성을 드러낸다
  3. **규칙 A** — 세 데이터셋 전부에서 -0.3%p 넘게 떨어지면 기각
  4. a=5(2홉 미사용)가 안전장치 — 기준선과 정확히 같아야 한다

프롬프트·문단 수(3)·잘라넣는 길이(600자)는 E4와 동일. 한 글자도 바꾸지 않는다.
LLM이 불리언 연산자를 뱉어 BM25가 손해를 보지만 결과를 보고 고치면 그것이 과적합이다.

후속 질의는 데이터셋별로 디스크에 저장한다 — 중간에 끊겨도 이어서 돈다.
"""

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from ngdb import BM25, DenseIndex, Graph

RESULTS_REPO = "goethe0101/neurographdb-results"
OFFICIAL = "osunlp/HippoRAG_v2"
EMBED_TAG = "NV-Embed-v2"
DATASETS = {"musique": "musique", "2wiki": "2wikimultihopqa", "hotpotqa": "hotpotqa"}
PROPRAG = {"musique": 78.3, "2wiki": 94.1, "hotpotqa": 97.4}
HIPPO2 = {"musique": 74.7, "2wiki": 90.4, "hotpotqa": 96.3}
MODEL = "llama3.1:8b"
OLLAMA = "http://localhost:11434/api/chat"
CACHE = Path("/tmp/ngdb_followups"); CACHE.mkdir(exist_ok=True)

DEPTH, WIDTH, NSEED, SCORE_MODE = 3, 4, 5, 0
MAXK, MAXW = 20, 6
N_CTX, CTX_CHARS = 3, 600
BUDGETS = (5, 4, 3, 2)

SYS = ("You are helping a search system answer a multi-hop question. "
       "You are shown the question and the passages retrieved so far. "
       "Some information needed to answer is still missing. "
       "Write ONE short search query that would retrieve the missing information. "
       "Use specific names, titles, or dates found in the passages above -- "
       "that is the whole point, since the original question does not contain them. "
       "Output only the query, nothing else. "
       "If the passages already contain everything needed, output exactly: NONE")

_N = re.compile(r"[^a-z0-9 ]+")
norm = lambda s: " ".join(_N.sub(" ", s.lower()).split())


def load_official(name):
    key = DATASETS[name]
    corpus = json.load(open(hf_hub_download(OFFICIAL, f"{key}_corpus.json",
                                            repo_type="dataset")))
    qs = json.load(open(hf_hub_download(OFFICIAL, f"{key}.json", repo_type="dataset")))
    nz = lambda s: " ".join(s.split())
    by_text, by_title = {}, {}
    for i, c in enumerate(corpus):
        by_text.setdefault(nz(c["text"]), i)
        by_title.setdefault(c["title"], i)
    out = []
    for r in qs:
        if name == "musique":
            ids = sorted({by_text[nz(p["paragraph_text"])] for p in r["paragraphs"]
                          if p.get("is_supporting") and nz(p["paragraph_text"]) in by_text})
        else:
            ids = sorted({by_title[sf[0]] for sf in r["supporting_facts"]
                          if sf[0] in by_title})
        out.append({"q": r["question"], "gold": ids})
    return [c["title"] for c in corpus], [c["text"] for c in corpus], out


def ask(question, ctx):
    body = json.dumps({
        "model": MODEL, "stream": False,
        "options": {"temperature": 0.0, "num_predict": 60},
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user",
                      "content": f"Question: {question}\n\nPassages retrieved so far:\n{ctx}"}],
    }).encode()
    for a in range(3):
        try:
            req = urllib.request.Request(OLLAMA, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return (json.load(r)["message"].get("content") or "").strip()
        except Exception:
            if a == 2:
                return ""
            time.sleep(3)
    return ""


def prepare(ds, t0):
    """색인 → 1홉 → 후속 질의 → 2홉. 데이터셋 하나를 끝내고 결과만 남긴다(메모리 절약)."""
    titles, texts, questions = load_official(ds)
    n = len(titles)
    P = np.load(hf_hub_download(RESULTS_REPO, f"emb/{ds}_{EMBED_TAG}_P.npy",
                                repo_type="dataset")).astype(np.float32)
    Q = np.load(hf_hub_download(RESULTS_REPO, f"emb/{ds}_{EMBED_TAG}_Q.npy",
                                repo_type="dataset")).astype(np.float32)
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
    print(f"[{time.time()-t0:7.1f}s] {ds}: 문단 {n:,} 엣지 {g.n_edges:,}", flush=True)

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
    del P, Q, QSIM, dense

    cf = CACHE / f"{ds}_llama31.json"
    follow = json.loads(cf.read_text())["followups"] if cf.exists() else []
    if len(follow) < len(questions):
        print(f"[{time.time()-t0:7.1f}s] {ds}: 후속 질의 생성 "
              f"({len(follow)}/{len(questions)}부터)", flush=True)
        tg = time.time()
        for i in range(len(follow), len(questions)):
            ctx = "\n\n".join(f"[{titles[d]}] {texts[d][:CTX_CHARS]}"
                              for d in hop1[i][:N_CTX])
            follow.append(ask(questions[i]["q"], ctx))
            if (i + 1) % 100 == 0:
                sp = (time.time() - tg) / (i + 1 - (len(follow) - (i + 1 - len(follow))) or 1)
                done = i + 1
                rate = (time.time() - tg) / max(done - (len(follow) - (i + 1)), 1)
                cf.write_text(json.dumps({"model": MODEL, "followups": follow},
                                         ensure_ascii=False))
                print(f"    {done}/{len(questions)}  "
                      f"{(time.time()-tg)/60:.1f}분 경과", flush=True)
        cf.write_text(json.dumps({"model": MODEL, "followups": follow}, ensure_ascii=False))
    else:
        print(f"[{time.time()-t0:7.1f}s] {ds}: 후속 질의 캐시 사용 ({len(follow)}개)", flush=True)

    nn = sum(1 for f in follow if f.upper().startswith("NONE"))
    nf = sum(1 for f in follow if not f)
    print(f"           NONE {nn}  빈응답 {nf}", flush=True)
    for i in range(3):
        print(f"    Q: {questions[i]['q'][:90]}\n    → {follow[i][:110]!r}", flush=True)

    hop2 = [bm.search(f, MAXK) if f and not f.upper().startswith("NONE") else []
            for f in follow]
    return {"questions": questions, "hop1": hop1,
            "hop2": [[x for x, _ in h] for h in hop2]}


def main():
    t0 = time.time()
    prep = {ds: prepare(ds, t0) for ds in DATASETS}

    def r5(ds, idxs, a):
        D = prep[ds]
        tot = gain = loss = 0.0
        m = 0
        for i in idxs:
            gold = set(D["questions"][i]["gold"])
            if not gold:
                continue
            m += 1
            out = list(D["hop1"][i][:a])
            for x in D["hop2"][i]:
                if len(out) >= 5:
                    break
                if x not in out:
                    out.append(x)
            for x in D["hop1"][i][a:]:
                if len(out) >= 5:
                    break
                if x not in out:
                    out.append(x)
            new, old = set(out[:5]) & gold, set(D["hop1"][i][:5]) & gold
            tot += len(new) / len(gold)
            gain += len(new - old); loss += len(old - new)
        return tot / m * 100, gain, loss

    nq = len(prep["musique"]["questions"])
    dev, test, full = list(range(0, nq, 2)), list(range(1, nq, 2)), list(range(nq))

    print(f"\n{'='*76}\n안전장치 — a=5(2홉 미사용)가 E1 GB 기록값을 재현하는가\n{'='*76}")
    print(f"  {'':<10}{'전체1000':>10}{'기록값':>10}")
    for ds, rec in (("musique", 72.98), ("2wiki", 93.50), ("hotpotqa", 96.15)):
        print(f"  {ds:<10}{r5(ds, full, 5)[0]:>10.2f}{rec:>10.2f}")

    print(f"\n{'='*76}\ndev(짝수 500)에서 예산을 고른다 — 세 데이터셋 공통\n{'='*76}")
    print(f"  {'1홉칸':>6}" + "".join(f"{d[:8]:>10}" for d in DATASETS) + f"{'평균':>9}")
    devs = {}
    for a in BUDGETS:
        vs = {ds: r5(ds, dev, a)[0] for ds in DATASETS}
        devs[a] = sum(vs.values()) / 3
        print(f"  {a:>6}" + "".join(f"{vs[ds]:>10.2f}" for ds in DATASETS) + f"{devs[a]:>9.2f}")
    best = max(devs, key=devs.get)
    print(f"\n  dev 선택: a={best}")

    print(f"\n{'='*76}\ntest(홀수 500) — 예산 전부를 찍는다\n{'='*76}")
    base = {ds: r5(ds, test, 5)[0] for ds in DATASETS}
    print(f"  {'1홉칸':>6}" + "".join(f"{d[:8]:>10}" for d in DATASETS)
          + f"{'평균':>9}{'기준대비':>10}")
    ba = sum(base.values()) / 3
    for a in BUDGETS:
        vs = {ds: r5(ds, test, a)[0] for ds in DATASETS}
        av = sum(vs.values()) / 3
        mk = "  ← dev 선택" if a == best else ""
        print(f"  {a:>6}" + "".join(f"{vs[ds]:>10.2f}" for ds in DATASETS)
              + f"{av:>9.2f}{av-ba:>+10.2f}{mk}")

    print(f"\n{'='*76}\n규칙 A — 세 데이터셋 전부에서 -0.3%p 넘게 떨어지면 기각\n{'='*76}")
    print(f"  {'':<10}{'기준':>8}{'a='+str(best):>8}{'차이':>9}{'회수':>7}{'손실':>7}"
          f"{'Hippo2':>9}{'PropRAG':>9}")
    ok = True
    for ds in DATASETS:
        v, gn, ls = r5(ds, test, best)
        d = v - base[ds]
        if d < -0.3:
            ok = False
        print(f"  {ds:<10}{base[ds]:>8.2f}{v:>8.2f}{d:>+9.2f}{gn:>7.0f}{ls:>7.0f}"
              f"{HIPPO2[ds]:>9}{PROPRAG[ds]:>9}")
    print(f"\n  규칙 A: {'**통과**' if ok else '**실패 — 다른 데이터셋을 해친다**'}")

    print(f"\n{'='*76}\n전체 1000 기준 (경쟁 수치와 같은 기준) — a={best}\n{'='*76}")
    print(f"  {'':<10}{'1홉만':>9}{'후속질의':>10}{'차이':>9}{'Hippo2':>9}{'PropRAG':>9}")
    f1 = f2 = 0.0
    for ds in DATASETS:
        b = r5(ds, full, 5)[0]; v = r5(ds, full, best)[0]
        f1 += b; f2 += v
        print(f"  {ds:<10}{b:>9.2f}{v:>10.2f}{v-b:>+9.2f}{HIPPO2[ds]:>9}{PROPRAG[ds]:>9}")
    print(f"  {'평균':<10}{f1/3:>9.2f}{f2/3:>10.2f}{(f2-f1)/3:>+9.2f}"
          f"{sum(HIPPO2.values())/3:>9.1f}{sum(PROPRAG.values())/3:>9.1f}")
    print(f"\n  ※ 전체 1000 수치는 절반이 예산 선택에 쓰였으므로 낙관적이다.")
    print(f"     모델: {MODEL} (E4는 라우터 Llama-3.1-8B였다 — 기제의 모델 의존성 검증)")

    Path("/tmp/e5_all.json").write_text(json.dumps(
        {"model": MODEL, "dev": devs, "chosen": best, "base_test": base,
         "rule_a_pass": ok}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
