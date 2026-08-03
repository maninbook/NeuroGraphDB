# /// script
# requires-python = ">=3.10"
# dependencies = ["pybind11", "numpy", "huggingface_hub>=0.28"]
# ///
"""E2 — 순위 방식을 고쳐 PropRAG을 넘을 수 있는가. **개발/테스트 분할로 정직하게.**

격차 진단이 알려준 것:
    2Wiki    실패의 65.6%가 순위 문제, 오라클 Recall@5 **97.7** (PropRAG 94.1)
    HotpotQA 실패의 75.0%가 순위 문제, 오라클 **99.1** (PropRAG 97.4)
    놓친 정답의 중앙 순위가 7~8위
**정답은 이미 후보에 있다. 엣지를 더 만들 필요 없이 순위만 고치면 된다.**

── 왜 분할하는가 ──────────────────────────────────────────────────────────
파라미터를 쓸어서 **테스트셋에서 제일 좋은 것을 고르면 그건 과적합이지 승리가 아니다.**
이 분야에 별도 개발셋이 없으므로 직접 만든다:

    질문 1000개를 인덱스 짝/홀로 가른다 (deterministic, 500/500)
    dev(짝수)에서만 설정을 고른다
    test(홀수)에서 **한 번만** 재고 그 값을 보고한다
    전체 1000 수치도 같이 내되 **절반이 튜닝에 쓰였음을 명시**한다

선택 기준: dev의 세 데이터셋 평균 Recall@5. **하나의 설정을 세 데이터셋에 공통 적용**한다.
데이터셋마다 다른 설정을 고르면 그건 3배 더 과적합이다.

**선택 잡음도 드러낸다.** 162개 중 dev 1등은 운으로 뽑혔을 수 있다. 그래서
**dev 상위 3개를 전부 test에서 재고 셋 다 보고한다.** 서로 크게 엇갈리면
선택이 불안정했다는 뜻이므로 "이겼다"고 말하지 않는다.
dev→test 하락폭도 함께 낸다. 크면 dev에 과적합된 것이다.

── 쓸어볼 것 (전부 순위 쪽, 엣지는 안 건드린다) ────────────────────────────
    score_mode  0 평균 / 1 최약(사슬은 가장 약한 고리) / 2 도착노드
    beam_width  2 / 4 / 8            (PropRAG은 4)
    depth       2 / 3                (PropRAG은 3)
    n_seeds     3 / 5 / 10           (강한 임베더에선 5가 최적이 아닐 수 있다)
    merge       append   빔 결과 먼저, dense로 채움 (현재)
                scoremax max(dense점수, 빔점수)로 정렬
                         — 빔 점수가 qsim이라 dense 점수와 **같은 척도**다.
                           예전 점수합치기 실패는 활성값이라 척도가 달라서였다.
                rrf      순위 융합 (파라미터 없음)

임베딩은 캐시돼 있다. **GPU 불필요, cpu-upgrade에서 무료.**
"""

import itertools
import json
import os
import re
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

CODE_REPO = "goethe0101/neurographdb"
RESULTS_REPO = "goethe0101/neurographdb-results"
OFFICIAL = "osunlp/HippoRAG_v2"
EMBED_TAG = "NV-Embed-v2"
DATASETS = {"musique": "musique", "2wiki": "2wikimultihopqa", "hotpotqa": "hotpotqa"}
PROPRAG = {"musique": 78.3, "2wiki": 94.1, "hotpotqa": 97.4}
HIPPO2 = {"musique": 74.7, "2wiki": 90.4, "hotpotqa": 96.3}
BASE_GB = {"musique": 73.0, "2wiki": 93.5, "hotpotqa": 96.2}   # E1 정정치
MAXK, MAXW, RRF_K = 20, 6, 60
_N = re.compile(r"[^a-z0-9 ]+")

SCORE_MODES = (0, 1, 2)
WIDTHS = (2, 4, 8)
DEPTHS = (2, 3)
SEEDS = (3, 5, 10)
MERGES = ("append", "scoremax", "rrf")


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
    """정답은 **문단 인덱스**로 확정한다 (MuSiQue 제목 중복 때문. SOTA.md 정정 참조)."""
    from huggingface_hub import hf_hub_download
    key = DATASETS[name]
    corpus = json.load(open(hf_hub_download(OFFICIAL, f"{key}_corpus.json", repo_type="dataset")))
    qs = json.load(open(hf_hub_download(OFFICIAL, f"{key}.json", repo_type="dataset")))
    titles = [c["title"] for c in corpus]
    texts = [c["text"] for c in corpus]
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
    return titles, texts, out


def title_edges(titles, texts):
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
    import numpy as np
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    build_core()
    from ngdb import DenseIndex, Graph

    prep = {}
    for ds in DATASETS:
        titles, texts, questions = load_official(ds)
        n = len(titles)
        P = np.load(hf_hub_download(RESULTS_REPO, f"emb/{ds}_{EMBED_TAG}_P.npy",
                                    repo_type="dataset"))
        Q = np.load(hf_hub_download(RESULTS_REPO, f"emb/{ds}_{EMBED_TAG}_Q.npy",
                                    repo_type="dataset"))
        dense = DenseIndex(P.shape[1]); dense.add_batch(P)
        g = Graph(n); g.add_edges([(a, b, 1.0) for a, b in title_edges(titles, texts)])
        # 질의 유사도는 설정마다 다시 계산할 필요가 없다. 한 번만 만든다.
        QSIM = (P @ Q.T).astype(np.float32)          # (n, nq)
        hits = [dense.search(Q[i], max(MAXK, max(SEEDS))) for i in range(len(questions))]
        prep[ds] = {"g": g, "questions": questions, "hits": hits, "QSIM": QSIM}
        print(f"[{time.time()-t0:6.1f}s] {ds}: 문단 {n:,} 질문 {len(questions):,} "
              f"엣지 {g.n_edges:,}")

    def rank(ds, i, sm, w, d, ns, merge):
        D = prep[ds]
        h = D["hits"][i]
        qs = D["QSIM"][:, i]
        sd = [x for x, _ in h[:ns]]
        bs = D["g"].beam_search(sd, qs.tolist(), d, w, 0.0, MAXK, sm)
        dense_ids = [x for x, _ in h[:MAXK]]
        if merge == "append":
            seen, out = set(), []
            for x, _ in bs:
                if x not in seen:
                    seen.add(x); out.append(x)
            for x in dense_ids:
                if x not in seen:
                    seen.add(x); out.append(x)
            return out[:MAXK]
        if merge == "scoremax":
            sc = {}
            for x, s in h[:MAXK]:
                sc[x] = max(sc.get(x, -9e9), float(s))
            for x, s in bs:
                sc[x] = max(sc.get(x, -9e9), float(s))
            return [x for x, _ in sorted(sc.items(), key=lambda y: -y[1])][:MAXK]
        rr = {}
        for r, x in enumerate(dense_ids):
            rr[x] = rr.get(x, 0.0) + 1.0 / (RRF_K + r + 1)
        for r, (x, _) in enumerate(bs):
            rr[x] = rr.get(x, 0.0) + 1.0 / (RRF_K + r + 1)
        return [x for x, _ in sorted(rr.items(), key=lambda y: -y[1])][:MAXK]

    def r5(ds, idxs, cfg):
        D = prep[ds]
        tot = 0.0
        for i in idxs:
            gold = D["questions"][i]["gold"]
            if not gold:
                continue
            R = rank(ds, i, *cfg)[:5]
            tot += sum(1 for gv in gold if gv in R) / len(gold)
        return tot / max(len(idxs), 1) * 100

    nq = len(prep["musique"]["questions"])
    dev = list(range(0, nq, 2))
    test = list(range(1, nq, 2))
    print(f"\ndev {len(dev)} / test {len(test)} (짝/홀 분할)")

    configs = list(itertools.product(SCORE_MODES, WIDTHS, DEPTHS, SEEDS, MERGES))
    print(f"설정 {len(configs)}개를 dev에서만 평가한다\n")

    best, rows = None, []
    for k, cfg in enumerate(configs, 1):
        vals = {ds: r5(ds, dev, cfg) for ds in DATASETS}
        avg = sum(vals.values()) / 3
        rows.append({"cfg": cfg, "dev": vals, "dev_avg": avg})
        if best is None or avg > best["dev_avg"]:
            best = rows[-1]
        if k % 20 == 0 or k == len(configs):
            print(f"  [{time.time()-t0:6.1f}s] {k}/{len(configs)} · 현재 최고 "
                  f"{best['cfg']} dev평균 {best['dev_avg']:.2f}")

    top3 = sorted(rows, key=lambda x: -x["dev_avg"])[:3]
    print(f"\n{'='*78}\ndev 상위 3개 — **셋 다 test에서 재서 선택 잡음을 드러낸다**\n{'='*78}")

    out = {"test_top3": [], "full_top3": []}
    for rk, r in enumerate(top3, 1):
        cfg = r["cfg"]
        sm, w, d, ns, mg = cfg
        te = {ds: r5(ds, test, cfg) for ds in DATASETS}
        fu = {ds: r5(ds, list(range(nq)), cfg) for ds in DATASETS}
        ta, fa = sum(te.values()) / 3, sum(fu.values()) / 3
        out["test_top3"].append({"rank": rk, "cfg": list(cfg), "dev_avg": r["dev_avg"],
                                 "test": te, "test_avg": ta, "full": fu, "full_avg": fa})
        print(f"\n[dev {rk}위] score_mode={sm} width={w} depth={d} seeds={ns} merge={mg}")
        print(f"  dev 평균 {r['dev_avg']:.2f} → test 평균 {ta:.2f} "
              f"(하락 {r['dev_avg']-ta:+.2f})")
        print(f"  {'':<10}{'test':>8}{'HippoRAG2':>11}{'PropRAG':>9}")
        for ds in DATASETS:
            mark = "  승" if te[ds] > PROPRAG[ds] else ("  동" if abs(te[ds]-PROPRAG[ds]) < 0.3 else "")
            print(f"  {ds:<10}{te[ds]:>8.1f}{HIPPO2[ds]:>11}{PROPRAG[ds]:>9}{mark}")
        print(f"  {'평균':<10}{ta:>8.1f}{sum(HIPPO2.values())/3:>11.1f}"
              f"{sum(PROPRAG.values())/3:>9.1f}")

    spread = max(x["test_avg"] for x in out["test_top3"]) - \
             min(x["test_avg"] for x in out["test_top3"])
    print(f"\n상위 3개의 test 평균 편차: {spread:.2f}%p")
    if spread > 1.0:
        print("  ! 선택이 불안정하다. dev 1등을 '우리 설정'이라고 부르면 안 된다")
    else:
        print("  선택이 안정적이다 — 상위권이 test에서도 비슷하게 나온다")

    print(f"\n전체 1000 수치는 참고용이다 — **절반이 튜닝에 쓰였으므로 낙관적이다.**")
    print(f"보고할 값은 test(홀수 500)이고, 공개 수치는 1000 기준이라 표본이 절반임을 밝힌다.")
    print(f"(PropRAG·HippoRAG 2도 빔폭·damping 등을 이 데이터셋들에서 골랐다.")
    print(f" 우리 dev/test 분할은 그쪽보다 오히려 보수적이다.)")

    p = Path("/tmp/tune_e2.json"); p.write_text(json.dumps(
        {**out, "top3_test_spread": spread,
         "all_configs": [{"cfg": list(r["cfg"]), "dev_avg": r["dev_avg"]}
                         for r in rows]}, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=str(p), path_in_repo=f"runs/{p.name}",
                        repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"\n업로드: runs/{p.name}  ({(time.time()-t0)/60:.1f}분, GPU 없음)")


if __name__ == "__main__":
    main()
