"""E6 — 후속 질의 **라운드 2**. MuSiQue 3~4홉이 막힌 자리를 뚫을 수 있는가.

D-HOPS(§21)가 알려준 것:
    2홉 518문항  +3.86   ← 이득이 전부 여기
    3홉 316문항  -0.63
    4홉 166문항  -1.66   기준 46.99 (2홉은 81.76), 빠진 정답의 56.5%가 후보에도 없음

후속 질의를 **한 번** 쓰면 다리를 **하나** 건넌다. 3~4홉은 다리가 둘·셋이다.
3~4홉도 2홉만큼 오르면 74.51 → 76.85, PropRAG과의 격차가 3.79 → 1.45가 된다.

── 설계 원칙 ────────────────────────────────────────────────────────────────
**홉 수를 보고 분기하지 않는다.** 시험 시점에 모르는 값이고, 쓰면 그것이 과적합이다.
같은 기제를 한 번 더 돌리고 **모델이 멈출 때를 정한다** — 프롬프트에 이미 NONE이 있다.
라운드 1이 NONE이면 라운드 2도 돌리지 않는다.

**프롬프트를 바꾸지 않는다.** 라운드 2도 §18의 SYS를 한 글자도 안 고쳐 쓴다.
맥락도 문단 3개로 같게 유지하되, 그중 하나를 **라운드 1이 찾아온 문단**으로 바꾼다:
    라운드 1 맥락:  hop1[0:3]
    라운드 2 맥락:  hop1[0:2] + hop2[0:1]        ← 크기 동일, 새 발견만 주입
크기를 늘리면 이득이 '맥락이 길어져서'인지 '라운드가 늘어서'인지 갈리지 않는다.

── 핵심 비교 ────────────────────────────────────────────────────────────────
칸 5개를 어떻게 나누는가. (n1, n2, n3) = hop1 / 라운드1 / 라운드2 몫.

    (5,0,0)  안전장치 — 기준선과 정확히 같아야 한다
    (4,1,0)  E5에서 dev로 고른 설정
    (3,2,0)  라운드 1에 두 칸          ← 이것과
    (3,1,1)  라운드 1·2에 한 칸씩       ← 이것의 비교가 실험의 핵심이다
    (2,2,1)

**(3,1,1) > (3,2,0)이어야 "라운드 2가 새 정보를 준다"가 성립한다.**
같으면 그냥 칸을 더 준 효과이고 라운드 2는 필요 없다.

dev(짝수 500)에서 고르고 test(홀수 500)에서 잰다. 세 데이터셋 공통 설정.
규칙 A — 세 데이터셋 전부에서 -0.3%p 넘게 떨어지면 기각.

라운드별 후속 질의는 생성 즉시 Hub에 올린다(/tmp는 청소된다 — 한 번 잃었다).
"""

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, HfApi

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
CACHE = Path(__file__).parent / "cache"; CACHE.mkdir(exist_ok=True)

DEPTH, WIDTH, NSEED, SCORE_MODE = 3, 4, 5, 0
MAXK, MAXW = 20, 6
N_CTX, CTX_CHARS = 3, 600
BUDGETS = ((5, 0, 0), (4, 1, 0), (3, 2, 0), (3, 1, 1), (2, 2, 1))

# §18과 동일. 라운드 1·2 모두 이것을 쓴다. 절대 바꾸지 않는다.
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
is_none = lambda f: (not f) or f.strip().upper().startswith("NONE")


def load_official(name):
    key = DATASETS[name]
    corpus = json.load(open(hf_hub_download(OFFICIAL, f"{key}_corpus.json", repo_type="dataset")))
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


def generate(ds, rnd, questions, titles, texts, ctx_of, t0):
    """라운드 rnd의 후속 질의. 캐시가 있으면 그대로 쓰고, 없으면 만들어 Hub에 올린다."""
    cf = CACHE / (f"{ds}_llama31.json" if rnd == 1 else f"{ds}_llama31_r{rnd}.json")
    follow = json.loads(cf.read_text())["followups"] if cf.exists() else []
    if len(follow) >= len(questions):
        print(f"[{time.time()-t0:7.1f}s] {ds} R{rnd}: 캐시 사용 ({len(follow)})", flush=True)
        return follow
    print(f"[{time.time()-t0:7.1f}s] {ds} R{rnd}: 생성 ({len(follow)}/{len(questions)}부터)", flush=True)
    tg = time.time()
    for i in range(len(follow), len(questions)):
        c = ctx_of(i)
        follow.append("" if c is None else ask(questions[i]["q"], c))
        if (i + 1) % 100 == 0:
            cf.write_text(json.dumps({"model": MODEL, "round": rnd, "followups": follow},
                                     ensure_ascii=False))
            print(f"    {i+1}/{len(questions)}  {(time.time()-tg)/60:.1f}분", flush=True)
    cf.write_text(json.dumps({"model": MODEL, "round": rnd, "followups": follow},
                             ensure_ascii=False))
    try:
        HfApi().upload_file(path_or_fileobj=str(cf),
                            path_in_repo=f"decomp/{ds}_followups_llama31_r{rnd}.json",
                            repo_id=RESULTS_REPO, repo_type="dataset")
        print(f"           Hub 업로드 완료", flush=True)
    except Exception as e:
        print(f"           Hub 업로드 실패(계속): {e}", flush=True)
    return follow


def prepare(ds, t0):
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

    doc = lambda d: f"[{titles[d]}] {texts[d][:CTX_CHARS]}"

    # 라운드 1 — 맥락 hop1[0:3]
    f1 = generate(ds, 1, questions, titles, texts,
                  lambda i: "\n\n".join(doc(d) for d in hop1[i][:N_CTX]), t0)
    hop2 = [[x for x, _ in (bm.search(f, MAXK) if not is_none(f) else [])] for f in f1]

    # 라운드 2 — 맥락 hop1[0:2] + hop2[0:1]. 크기 동일, 새 발견만 주입.
    # 라운드 1이 NONE이거나 아무것도 못 찾았으면 라운드 2를 돌리지 않는다.
    def ctx2(i):
        if is_none(f1[i]) or not hop2[i]:
            return None
        return "\n\n".join(doc(d) for d in list(hop1[i][:N_CTX - 1]) + [hop2[i][0]])
    f2 = generate(ds, 2, questions, titles, texts, ctx2, t0)
    hop3 = [[x for x, _ in (bm.search(f, MAXK) if not is_none(f) else [])] for f in f2]

    skipped = sum(1 for i in range(len(questions)) if ctx2(i) is None)
    print(f"           R1 NONE {sum(1 for f in f1 if is_none(f))}  "
          f"R2 건너뜀 {skipped}  R2 NONE {sum(1 for f in f2 if is_none(f))}", flush=True)
    return {"questions": questions, "hop1": hop1, "hop2": hop2, "hop3": hop3}


def main():
    t0 = time.time()
    prep = {ds: prepare(ds, t0) for ds in DATASETS}

    def merged(D, i, b):
        n1, n2, n3 = b
        out = list(D["hop1"][i][:n1])
        for src, cnt in ((D["hop2"][i], n2), (D["hop3"][i], n3)):
            added = 0
            for x in src:
                if added >= cnt or len(out) >= 5:
                    break
                if x not in out:
                    out.append(x); added += 1
        for x in D["hop1"][i][n1:]:      # 남는 칸은 hop1으로 채운다
            if len(out) >= 5:
                break
            if x not in out:
                out.append(x)
        return out[:5]

    def r5(ds, idxs, b):
        D = prep[ds]
        tot = gain = loss = 0.0
        m = 0
        for i in idxs:
            gold = set(D["questions"][i]["gold"])
            if not gold:
                continue
            m += 1
            new = set(merged(D, i, b)) & gold
            old = set(D["hop1"][i][:5]) & gold
            tot += len(new) / len(gold)
            gain += len(new - old); loss += len(old - new)
        return tot / m * 100, gain, loss

    nq = len(prep["musique"]["questions"])
    dev, test, full = list(range(0, nq, 2)), list(range(1, nq, 2)), list(range(nq))

    print(f"\n{'='*78}\n안전장치 — (5,0,0)이 E1 GB 기록값을 재현하는가\n{'='*78}")
    for ds, rec in (("musique", 72.98), ("2wiki", 93.50), ("hotpotqa", 96.15)):
        print(f"  {ds:<10}{r5(ds, full, (5,0,0))[0]:>9.2f}  기록 {rec}")

    print(f"\n{'='*78}\ndev(짝수 500) — 세 데이터셋 공통 설정\n{'='*78}")
    print(f"  {'배분':>9}" + "".join(f"{d[:8]:>10}" for d in DATASETS) + f"{'평균':>9}")
    devs = {}
    for b in BUDGETS:
        vs = {ds: r5(ds, dev, b)[0] for ds in DATASETS}
        devs[b] = sum(vs.values()) / 3
        print(f"  {str(b):>9}" + "".join(f"{vs[ds]:>10.2f}" for ds in DATASETS)
              + f"{devs[b]:>9.2f}")
    best = max(devs, key=devs.get)
    print(f"\n  dev 선택: {best}")
    print(f"  **핵심 비교** (3,2,0) {devs[(3,2,0)]:.2f} vs (3,1,1) {devs[(3,1,1)]:.2f}"
          f"  → 라운드2 순효과 {devs[(3,1,1)]-devs[(3,2,0)]:+.2f}")

    print(f"\n{'='*78}\ntest(홀수 500) — 배분 전부를 찍는다\n{'='*78}")
    base = {ds: r5(ds, test, (5,0,0))[0] for ds in DATASETS}
    ba = sum(base.values()) / 3
    print(f"  {'배분':>9}" + "".join(f"{d[:8]:>10}" for d in DATASETS)
          + f"{'평균':>9}{'기준대비':>10}")
    for b in BUDGETS:
        vs = {ds: r5(ds, test, b)[0] for ds in DATASETS}
        av = sum(vs.values()) / 3
        mk = "  ← dev 선택" if b == best else ""
        print(f"  {str(b):>9}" + "".join(f"{vs[ds]:>10.2f}" for ds in DATASETS)
              + f"{av:>9.2f}{av-ba:>+10.2f}{mk}")

    print(f"\n{'='*78}\n규칙 A — 세 데이터셋 전부에서 -0.3%p 넘게 떨어지면 기각\n{'='*78}")
    print(f"  {'':<10}{'기준':>8}{'선택':>8}{'차이':>9}{'회수':>7}{'손실':>7}{'Hippo2':>9}{'PropRAG':>9}")
    ok = True
    for ds in DATASETS:
        v, gn, ls = r5(ds, test, best)
        d = v - base[ds]
        if d < -0.3:
            ok = False
        print(f"  {ds:<10}{base[ds]:>8.2f}{v:>8.2f}{d:>+9.2f}{gn:>7.0f}{ls:>7.0f}"
              f"{HIPPO2[ds]:>9}{PROPRAG[ds]:>9}")
    print(f"\n  규칙 A: {'**통과**' if ok else '**실패**'}")

    print(f"\n{'='*78}\n전체 1000 기준 — {best}\n{'='*78}")
    print(f"  {'':<10}{'1홉만':>9}{'선택':>9}{'차이':>9}{'Hippo2':>9}{'PropRAG':>9}")
    s1 = s2 = 0.0
    for ds in DATASETS:
        b0 = r5(ds, full, (5,0,0))[0]; v = r5(ds, full, best)[0]
        s1 += b0; s2 += v
        print(f"  {ds:<10}{b0:>9.2f}{v:>9.2f}{v-b0:>+9.2f}{HIPPO2[ds]:>9}{PROPRAG[ds]:>9}")
    print(f"  {'평균':<10}{s1/3:>9.2f}{s2/3:>9.2f}{(s2-s1)/3:>+9.2f}"
          f"{sum(HIPPO2.values())/3:>9.1f}{sum(PROPRAG.values())/3:>9.1f}")

    # MuSiQue 홉 수별 — 라운드 2가 3~4홉을 뚫었는가 (§21의 그 표)
    D = prep["musique"]
    strata = {}
    for i, r in enumerate(D["questions"]):
        if r["gold"]:
            strata.setdefault(len(r["gold"]), []).append(i)
    print(f"\n{'='*78}\nMuSiQue 홉 수별 — 라운드 2가 3~4홉을 뚫었는가 (§21 대조)\n{'='*78}")
    print(f"  {'정답수':>6}{'질문':>7}{'기준':>9}{'(4,1,0)':>10}{'선택':>9}"
          f"{'R2효과':>9}{'§21':>8}")
    prev = {2: +3.86, 3: -0.63, 4: -1.66}
    for k in sorted(strata):
        idx = strata[k]
        f = lambda b: sum(len(set(merged(D, i, b)) & set(D["questions"][i]["gold"])) / k
                          for i in idx) / len(idx) * 100
        b0, b1, bb = f((5,0,0)), f((4,1,0)), f(best)
        print(f"  {k:>6}{len(idx):>7}{b0:>9.2f}{b1:>10.2f}{bb:>9.2f}"
              f"{bb-b1:>+9.2f}{prev[k]:>+8.2f}")

    Path("/tmp/e6.json").write_text(json.dumps(
        {"dev": {str(k): v for k, v in devs.items()}, "chosen": list(best),
         "base_test": base, "rule_a": ok}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
