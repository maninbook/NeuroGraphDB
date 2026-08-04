# /// script
# requires-python = ">=3.10"
# dependencies = ["pybind11", "numpy", "huggingface_hub>=0.28"]
# ///
"""E5 — E4의 MuSiQue 이득(+1.62)을 **정직하게 검증한다.**

E4가 전체 1000개에서 본 것:
    MuSiQue  1홉 72.98 → 4+1 합치기 **74.60 (+1.62)**, 회수 107 / 손실 76
    천장: BM25 상위20에서 빠진 정답 46.3% 회수, 그중 **24.9%는 원래 상위20에도 없던 것**
    즉 순위 재배열이 아니라 **새 정보**다. 비용 $0.0134.

E4로는 주장할 수 없는 이유 넷 — 여기서 전부 막는다:
  1. 전체 1000개에서 쟀고 **합치기 예산(4+1)을 그걸 보고 골랐다**
     → dev(짝수 500)에서 고르고 test(홀수 500)에서 한 번만 잰다
  2. 작동점이 좁다 (4+1만 되고 3+2부터 손해)
     → 예산 전부를 dev·test 양쪽에 찍어 취약성을 드러낸다
  3. 규칙 A 미검증 (한 데이터셋에서만 듣는 것은 기제가 아니라 그 데이터셋의 특성일 수 있다)
     → **세 데이터셋 전부** 돌리고, 나머지 둘에서 **해롭지 않아야** 채택한다
  4. 선택 잡음
     → dev 상위 예산 여럿을 test에서 재고 편차를 찍는다

**프롬프트는 E4와 한 글자도 안 바꾼다.** LLM이 불리언 연산자를 뱉어 BM25가 손해를
보고 있지만(예: "AND (Poland OR Warsaw)"), 결과를 보고 프롬프트를 고치면 그게 과적합이다.
보수적인 숫자를 낸다.

MuSiQue 후속 질의는 **E4 캐시를 재사용**한다(재생성하면 temperature 0라도 값이
흔들릴 수 있고, 그러면 E4와 비교가 안 된다). 2Wiki·HotpotQA만 새로 생성한다.
추가 비용 약 $0.03.
"""

import json
import os
import re
import subprocess
import sys
import sysconfig
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE_REPO = "goethe0101/neurographdb"
RESULTS_REPO = "goethe0101/neurographdb-results"
OFFICIAL = "osunlp/HippoRAG_v2"
EMBED_TAG = "NV-Embed-v2"
DATASETS = {"musique": "musique", "2wiki": "2wikimultihopqa", "hotpotqa": "hotpotqa"}
PROPRAG = {"musique": 78.3, "2wiki": 94.1, "hotpotqa": 97.4}
HIPPO2 = {"musique": 74.7, "2wiki": 90.4, "hotpotqa": 96.3}
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ROUTER = "https://router.huggingface.co/v1/chat/completions"

DEPTH, WIDTH, NSEED, SCORE_MODE = 3, 4, 5, 0     # E1 GB 설정 그대로
MAXK, MAXW = 20, 6
N_CTX, CTX_CHARS, WORKERS = 3, 600, 8
BUDGETS = (5, 4, 3, 2)      # 1홉에 남길 칸 수. 5면 2홉을 안 쓴다(= 안전장치)

# E4와 동일. 절대 바꾸지 않는다.
SYS = ("You are helping a search system answer a multi-hop question. "
       "You are shown the question and the passages retrieved so far. "
       "Some information needed to answer is still missing. "
       "Write ONE short search query that would retrieve the missing information. "
       "Use specific names, titles, or dates found in the passages above -- "
       "that is the whole point, since the original question does not contain them. "
       "Output only the query, nothing else. "
       "If the passages already contain everything needed, output exactly: NONE")

_N = re.compile(r"[^a-z0-9 ]+")


def norm(s):
    return " ".join(_N.sub(" ", s.lower()).split())


def build_core():
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(CODE_REPO, repo_type="dataset", allow_patterns="src/**"))
    src = root / "src"
    out = Path("/tmp/ngdb"); out.mkdir(exist_ok=True)
    (out / "__init__.py").write_text(
        "from ._ngdb_core import BM25, DenseIndex, Graph\n"
        '__all__ = ["BM25", "DenseIndex", "Graph"]\n')
    inc = subprocess.run([sys.executable, "-m", "pybind11", "--includes"],
                         capture_output=True, text=True, check=True).stdout.split()
    ext = sysconfig.get_config_var("EXT_SUFFIX")
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-fPIC", "-shared",
                    *inc, f"-I{src}", "-o", str(out / f"_ngdb_core{ext}"),
                    str(src / "bm25.cpp"), str(src / "dense.cpp"),
                    str(src / "graph.cpp"), str(src / "bindings.cpp")], check=True)
    sys.path.insert(0, "/tmp")


def load_official(name):
    from huggingface_hub import hf_hub_download
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


def title_edges(titles, texts):
    lookup = {}
    for i, t in enumerate(titles):
        k = norm(t)
        if len(k) >= 5:
            lookup.setdefault(k, i)
    e = set()
    for i, body in enumerate(texts):
        toks = norm(body).split()
        for w in range(1, MAXW + 1):
            for a in range(len(toks) - w + 1):
                j = lookup.get(" ".join(toks[a:a + w]))
                if j is not None and j != i:
                    e.add((i, j))
    return e


def ask(tok, question, ctx):
    body = json.dumps({
        "model": MODEL, "temperature": 0.0, "max_tokens": 60,
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user",
                      "content": f"Question: {question}\n\nPassages retrieved so far:\n{ctx}"}],
    }).encode()
    for a in range(4):
        try:
            req = urllib.request.Request(
                ROUTER, data=body,
                headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.load(r)
            return (d["choices"][0]["message"].get("content") or "").strip(), \
                   d.get("usage", {}).get("total_tokens", 0)
        except Exception:
            if a == 3:
                return "", 0
            time.sleep(2 * (a + 1))
    return "", 0


def main():
    import numpy as np
    from huggingface_hub import get_token, hf_hub_download, HfApi
    t0 = time.time()
    build_core()
    from ngdb import BM25, DenseIndex, Graph
    api, tok = HfApi(), get_token()

    prep, spent = {}, 0.0
    for ds in DATASETS:
        titles, texts, questions = load_official(ds)
        n = len(titles)
        P = np.load(hf_hub_download(RESULTS_REPO, f"emb/{ds}_{EMBED_TAG}_P.npy",
                                    repo_type="dataset")).astype(np.float32)
        Q = np.load(hf_hub_download(RESULTS_REPO, f"emb/{ds}_{EMBED_TAG}_Q.npy",
                                    repo_type="dataset")).astype(np.float32)
        dense = DenseIndex(P.shape[1]); dense.add_batch(P)
        g = Graph(n); g.add_edges([(a, b, 1.0) for a, b in title_edges(titles, texts)])
        bm = BM25()
        for t, b in zip(titles, texts):
            bm.add(t + " " + b)
        bm.finalize()
        QSIM = (P @ Q.T).astype(np.float32)

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
        print(f"[{time.time()-t0:6.1f}s] {ds}: 문단 {n:,} 엣지 {g.n_edges:,}")

        # 후속 질의 — MuSiQue는 E4 캐시 재사용
        cache = f"decomp/{ds}_followups.json"
        follow = None
        try:
            follow = json.load(open(hf_hub_download(RESULTS_REPO, cache,
                                                    repo_type="dataset")))["followups"]
            print(f"           캐시 재사용: {cache} ({len(follow)}개)")
        except Exception:
            pass
        if follow is None or len(follow) != len(questions):
            def one(i):
                ctx = "\n\n".join(f"[{titles[d]}] {texts[d][:CTX_CHARS]}"
                                  for d in hop1[i][:N_CTX])
                return ask(tok, questions[i]["q"], ctx)
            with ThreadPoolExecutor(WORKERS) as ex:
                res = list(ex.map(one, range(len(questions))))
            follow = [r[0] for r in res]
            nt = sum(r[1] for r in res); spent += nt * 2.8e-8
            print(f"[{time.time()-t0:6.1f}s]  {ds} 생성 {nt:,}토큰 (약 ${nt*2.8e-8:.4f}) "
                  f"빈응답 {sum(1 for f in follow if not f)} "
                  f"NONE {sum(1 for f in follow if f.upper().startswith('NONE'))}")
            api.upload_file(path_or_fileobj=json.dumps(
                {"model": MODEL, "system": SYS, "n_ctx": N_CTX, "followups": follow},
                ensure_ascii=False, indent=1).encode(),
                path_in_repo=cache, repo_id=RESULTS_REPO, repo_type="dataset")

        hop2 = [bm.search(f, MAXK) if f and not f.upper().startswith("NONE") else []
                for f in follow]
        prep[ds] = {"questions": questions, "hop1": hop1, "hop2": hop2}

    print(f"\n이번 실행 LLM 비용 합계 약 ${spent:.4f}")

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
            for x, _ in D["hop2"][i]:
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
    dev, test = list(range(0, nq, 2)), list(range(1, nq, 2))
    print(f"\ndev {len(dev)} / test {len(test)} (짝/홀 분할)")

    print(f"\n{'='*78}\ndev에서 예산을 고른다 (a=5는 2홉 미사용 = 안전장치)\n{'='*78}")
    print(f"  {'1홉칸':>6}" + "".join(f"{ds[:8]:>10}" for ds in DATASETS) + f"{'평균':>9}")
    devrows = []
    for a in BUDGETS:
        vs = {ds: r5(ds, dev, a)[0] for ds in DATASETS}
        avg = sum(vs.values()) / 3
        devrows.append((a, vs, avg))
        print(f"  {a:>6}" + "".join(f"{vs[ds]:>10.2f}" for ds in DATASETS) + f"{avg:>9.2f}")

    basedev = next(v for a, v, _ in devrows if a == 5)
    best = max(devrows, key=lambda x: x[2])
    print(f"\n  dev 최고 예산: a={best[0]} (평균 {best[2]:.2f}, "
          f"기준 {next(x[2] for x in devrows if x[0]==5):.2f})")

    print(f"\n{'='*78}\ntest에서 잰다 — 예산 전부를 찍어 취약성을 드러낸다\n{'='*78}")
    basetest = {ds: r5(ds, test, 5)[0] for ds in DATASETS}
    print(f"  기준선(2홉 미사용): " + " ".join(f"{ds} {basetest[ds]:.2f}" for ds in DATASETS)
          + f"  평균 {sum(basetest.values())/3:.2f}")
    print(f"\n  {'1홉칸':>6}" + "".join(f"{ds[:8]:>10}" for ds in DATASETS)
          + f"{'평균':>9}{'기준대비':>10}")
    testrows = {}
    for a in BUDGETS:
        vs = {ds: r5(ds, test, a)[0] for ds in DATASETS}
        avg = sum(vs.values()) / 3
        testrows[a] = vs
        mark = "  ← dev 선택" if a == best[0] else ""
        print(f"  {a:>6}" + "".join(f"{vs[ds]:>10.2f}" for ds in DATASETS)
              + f"{avg:>9.2f}{avg-sum(basetest.values())/3:>+10.2f}{mark}")

    a = best[0]
    print(f"\n{'='*78}\n판정 — 규칙 A: 세 데이터셋 전부에서 해롭지 않아야 한다\n{'='*78}")
    print(f"  {'':<10}{'기준':>8}{'a='+str(a):>8}{'차이':>9}{'회수':>7}{'손실':>7}"
          f"{'HippoRAG2':>11}{'PropRAG':>9}")
    ok = True
    for ds in DATASETS:
        v, gn, ls = r5(ds, test, a)
        d = v - basetest[ds]
        if d < -0.3:
            ok = False
        print(f"  {ds:<10}{basetest[ds]:>8.2f}{v:>8.2f}{d:>+9.2f}{gn:>7.0f}{ls:>7.0f}"
              f"{HIPPO2[ds]:>11}{PROPRAG[ds]:>9}")
    print(f"\n  규칙 A: {'통과 — 어느 데이터셋도 -0.3%p 넘게 떨어지지 않았다' if ok else '**실패 — 다른 데이터셋을 해친다**'}")

    Path("/tmp/decomp3.json").write_text(json.dumps(
        {"dev": [[a, v, g] for a, v, g in devrows], "base_test": basetest,
         "test": {str(k): v for k, v in testrows.items()}, "chosen": a,
         "rule_a_pass": ok}, indent=2, ensure_ascii=False))
    api.upload_file(path_or_fileobj="/tmp/decomp3.json",
                    path_in_repo="runs/decomp_e5.json",
                    repo_id=RESULTS_REPO, repo_type="dataset")


if __name__ == "__main__":
    main()
