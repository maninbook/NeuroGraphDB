# /// script
# requires-python = ">=3.10"
# dependencies = ["pybind11", "numpy", "huggingface_hub>=0.28"]
# ///
"""E4 — 질의 쪽 공략. **후속 질의**를 만들어 MuSiQue의 벽을 넘을 수 있는가.

── 왜 질의 쪽인가 ──────────────────────────────────────────────────────────
MuSiQue의 벽이 숫자로 확정됐다(§17): **제목 엣지 정밀도 1.65%, 회수 12.7%.**
우리가 만들 수 있는 어떤 엣지 집합도 2% 아래다. **코퍼스에 정밀한 엣지 신호가 없다.**
단일홉 질문을 다른 출처에서 합성해 만들었으니 근거 문단이 링크된 적이 없다(§12).
구조가 코퍼스가 아니라 **질문**에 있다.

── 왜 정적 분해로는 안 되는가 (스모크 테스트에서 확인) ──────────────────────
"인셉션 감독의 배우자는?"을 그냥 분해시키면:
    "인셉션 감독이 누구인가?"
    "인셉션 감독의 배우자는?"      ← **여전히 '놀란'이 없다**
다리 개체는 **1홉을 실제로 읽어야** 나온다. 그래서 검색-후-재질의로 간다:

    1홉  기존 GB 파이프라인으로 검색 → 상위 3개 문단
    LLM  질문 + 그 3개 문단을 주고 **아직 빠진 것을 찾을 후속 질의** 하나를 생성
    2홉  후속 질의로 재검색

── 비용 ────────────────────────────────────────────────────────────────────
LLM은 **질문 1,000개**에만 쓴다. 색인에는 여전히 한 번도 안 쓴다.
경쟁자는 문단 11,656개에 LLM을 돌린다 — 11.7배 차이다.
Llama-3.1-8B, 질문당 약 600토큰, 실측 $0.028/M토큰 → **약 $0.02.**
생성 결과를 Hub에 캐시하므로 이후 실험은 전부 무료로 재현된다(임베딩 캐시와 같은 방식).

── 이 스크립트가 재는 것 ───────────────────────────────────────────────────
합치기(merge)는 나중 문제다. 먼저 **천장**부터 본다:

C1. 후속 질의의 BM25 상위 k에 **빠진 정답**이 들어오는가.
    BM25를 쓰는 이유 — 다리 개체는 희소 고유명사이고, 문자열 일치가 그것을
    정확히 분리한다(§15에서 lift 350~1035로 확인). 임베딩은 문단 전체로 희석시킨다(§14).
C2. 그 정답이 **원래 상위 20에도 없던** 것인가.
    이미 후보에 있던 것을 다시 찾아오는 것은 새 정보가 아니다.
C3. 합쳤을 때 Recall@5. 칸이 5개뿐이라 **얻은 만큼 잃을 수 있다** —
    회수와 손실을 나란히 찍는다(all@10에서 세 번 겪은 실패다).

프롬프트는 **하나**만 쓴다. 데이터셋별로 다르게 쓰면 그게 과적합이다.
temperature 0. 실패한 호출은 후속 질의 없음으로 처리한다(안전 쪽).
"""

import json
import os
import re
import subprocess
import sys
import sysconfig
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE_REPO = "goethe0101/neurographdb"
RESULTS_REPO = "goethe0101/neurographdb-results"
OFFICIAL = "osunlp/HippoRAG_v2"
EMBED_TAG = "NV-Embed-v2"
DS = "musique"
DS_KEY = "musique"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ROUTER = "https://router.huggingface.co/v1/chat/completions"

DEPTH, WIDTH, NSEED, SCORE_MODE = 3, 4, 5, 0     # E1 GB 설정 그대로
MAXK, MAXW = 20, 6
N_CTX = 3            # LLM에 보여줄 1홉 문단 수
CTX_CHARS = 600      # 문단당 잘라 넣는 길이
WORKERS = 8

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


def load_official():
    from huggingface_hub import hf_hub_download
    corpus = json.load(open(hf_hub_download(OFFICIAL, f"{DS_KEY}_corpus.json",
                                            repo_type="dataset")))
    qs = json.load(open(hf_hub_download(OFFICIAL, f"{DS_KEY}.json", repo_type="dataset")))
    nz = lambda s: " ".join(s.split())
    by_text = {}
    for i, c in enumerate(corpus):
        by_text.setdefault(nz(c["text"]), i)
    out = []
    for r in qs:
        ids = sorted({by_text[nz(p["paragraph_text"])] for p in r["paragraphs"]
                      if p.get("is_supporting") and nz(p["paragraph_text"]) in by_text})
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
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                ROUTER, data=body,
                headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.load(r)
            c = (d["choices"][0]["message"].get("content") or "").strip()
            return c, d.get("usage", {}).get("total_tokens", 0)
        except Exception:
            if attempt == 3:
                return "", 0
            time.sleep(2 * (attempt + 1))
    return "", 0


def main():
    import numpy as np
    from huggingface_hub import get_token, hf_hub_download, HfApi
    t0 = time.time()
    build_core()
    from ngdb import BM25, DenseIndex, Graph

    titles, texts, questions = load_official()
    n = len(titles)
    P = np.load(hf_hub_download(RESULTS_REPO, f"emb/{DS}_{EMBED_TAG}_P.npy",
                                repo_type="dataset")).astype(np.float32)
    Q = np.load(hf_hub_download(RESULTS_REPO, f"emb/{DS}_{EMBED_TAG}_Q.npy",
                                repo_type="dataset")).astype(np.float32)
    dense = DenseIndex(P.shape[1]); dense.add_batch(P)
    g = Graph(n); g.add_edges([(a, b, 1.0) for a, b in title_edges(titles, texts)])
    bm = BM25()
    for t, b in zip(titles, texts):
        bm.add(t + " " + b)
    bm.finalize()
    QSIM = (P @ Q.T).astype(np.float32)
    print(f"[{time.time()-t0:6.1f}s] 문단 {n:,} 질문 {len(questions):,} 엣지 {g.n_edges:,}")

    # ── 1홉: 기존 GB 파이프라인 ─────────────────────────────────────────────
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

    base5 = sum(len(set(r["gold"]) & set(hop1[i][:5])) / len(r["gold"])
                for i, r in enumerate(questions) if r["gold"]) / len(questions) * 100
    print(f"[{time.time()-t0:6.1f}s] 1홉 Recall@5 {base5:.2f}  (§17 기록값 72.98과 대조)")

    # ── LLM: 후속 질의 생성 ────────────────────────────────────────────────
    tok = get_token()

    def one(i):
        ctx = "\n\n".join(f"[{titles[d]}] {texts[d][:CTX_CHARS]}"
                          for d in hop1[i][:N_CTX])
        return ask(tok, questions[i]["q"], ctx)

    print(f"[{time.time()-t0:6.1f}s] 후속 질의 생성 시작 ({MODEL}, {WORKERS} 동시)")
    with ThreadPoolExecutor(WORKERS) as ex:
        res = list(ex.map(one, range(len(questions))))
    follow = [r[0] for r in res]
    ntok = sum(r[1] for r in res)
    nfail = sum(1 for f in follow if not f)
    nnone = sum(1 for f in follow if f.strip().upper().startswith("NONE"))
    print(f"[{time.time()-t0:6.1f}s] 완료. 총 {ntok:,}토큰 "
          f"(약 ${ntok*2.8e-8:.4f})  빈응답 {nfail}  NONE {nnone}")
    for i in range(5):
        print(f"    Q: {questions[i]['q']}\n    →  {follow[i]!r}")

    HfApi().upload_file(
        path_or_fileobj=json.dumps(
            {"model": MODEL, "system": SYS, "n_ctx": N_CTX,
             "followups": follow}, ensure_ascii=False, indent=1).encode(),
        path_in_repo=f"decomp/{DS}_followups.json",
        repo_id=RESULTS_REPO, repo_type="dataset")
    print("  캐시 업로드: decomp/musique_followups.json (이후 실험은 무료 재현)")

    # ── C1/C2: 천장 ────────────────────────────────────────────────────────
    print(f"\n{'='*78}\n천장 — 후속 질의가 **빠진 정답**을 데려오는가\n{'='*78}")
    hop2 = [bm.search(f, 20) if f and not f.upper().startswith("NONE") else []
            for f in follow]
    for k in (1, 3, 5, 10, 20):
        c1 = c2 = denom = 0
        for i, r in enumerate(questions):
            gold = set(r["gold"])
            if not gold:
                continue
            miss5 = gold - set(hop1[i][:5])
            if not miss5:
                continue
            denom += 1
            got = {x for x, _ in hop2[i][:k]}
            if miss5 & got:
                c1 += 1
            if (miss5 - set(hop1[i][:MAXK])) & got:
                c2 += 1
        print(f"  BM25 상위 {k:>2}: 빠진 정답 회수 {c1:>4}/{denom} ({c1/max(denom,1)*100:5.1f}%)"
              f"   그중 상위20에도 없던 것 {c2:>4} ({c2/max(denom,1)*100:5.1f}%)")

    # ── C3: 실제 합치기 ────────────────────────────────────────────────────
    print(f"\n{'='*78}\n합치기 — 칸이 5개뿐이다. 회수와 손실을 나란히 본다\n{'='*78}")
    print(f"  {'1홉칸':>6}{'2홉칸':>6}{'R@5':>8}{'기준대비':>10}{'회수':>7}{'손실':>7}")
    for a in (5, 4, 3, 2):
        b = 5 - a
        tot = gain = loss = 0.0
        for i, r in enumerate(questions):
            gold = set(r["gold"])
            if not gold:
                continue
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
        r5 = tot / len(questions) * 100
        print(f"  {a:>6}{b:>6}{r5:>8.2f}{r5-base5:>+10.2f}{gain:>7.0f}{loss:>7.0f}")

    print("\n  ※ 전체 1000개 진단이다. 채택하려면 dev/test 분할로 다시 검증한다.")
    print("     PropRAG MuSiQue Recall@5 = 78.3 / HippoRAG 2 = 74.7")


if __name__ == "__main__":
    main()
