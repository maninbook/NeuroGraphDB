# /// script
# requires-python = ">=3.10"
# dependencies = ["pybind11", "numpy", "huggingface_hub>=0.28"]
# ///
"""E3 — 희소 고유명사 **다리 엣지**를 그래프에 넣으면 실제로 오르는가.

D-BRIDGE가 알려준 것 (발판이 진짜 정답일 때 = 오라클 성격):
    MuSiQue  다리 존재 96.1% vs 제목 언급 32.8%
             정답쌍의 21.0%가 코퍼스 5회 이하 문자열을 공유, 무작위쌍 0.06% → **lift 350**
    2Wiki    99.8% vs 88.8%, lift 273
    HotpotQA 92.5% vs 70.1%, lift 699

**빈도로 거른 것이 결정적이다.** 논항 공유는 빈도를 안 걸러 lift가 무너졌다(31~74).
'New York'은 다리가 아니다. 'Christopher Nolan'은 다리다.

── 왜 단서 갱신과 달리 해볼 만한가 ──────────────────────────────────────────
단서 갱신은 질의 시점에 "어느 것이 발판인가"를 골라야 했고, MuSiQue는 1홉이 가장
약해서 그 도박이 자멸했다(130 회수 / 360 손실).
문자열 다리는 **색인 시점에 엣지로 넣는다.** 발판 선택이 아예 없다.
그래프에 엣지가 늘고 기존 희소 seed + 확산이 알아서 쓴다.

── 진짜 위험: 엣지 폭발 ─────────────────────────────────────────────────────
빈도 d인 문자열은 d(d-1)/2개의 엣지를 만든다. 임계값을 올리면 폭발한다.
현재 제목 그래프는 2Wiki 2,743개, HotpotQA 5,958개로 **아주 희소**하고,
이 희소성이 우리 방법의 근간이다. 엣지가 뭉개지면 확산이 무의미해진다
— 이 프로젝트에서 여러 번 본 실패다.

그래서 임계값 T를 쓸되 **엣지 수와 실제 Recall@5를 같이 낸다.**
T는 dev(짝수 500)에서 고르고 test(홀수 500)에서 한 번만 잰다. 세 데이터셋 공통.

절제 실험도 같이 돌린다:
    title      제목 엣지만 (= E1 기준선, 안전장치)
    bridge     다리 엣지만
    both       둘 다
어느 쪽이 일하는지 갈라 봐야 한다. both만 재면 무엇이 이득의 원인인지 모른다.

임베딩은 캐시돼 있다. **GPU 불필요, cpu-upgrade에서 무료.**
"""

import json
import os
import re
import subprocess
import sys
import sysconfig
import time
from collections import Counter, defaultdict
from pathlib import Path

CODE_REPO = "goethe0101/neurographdb"
RESULTS_REPO = "goethe0101/neurographdb-results"
OFFICIAL = "osunlp/HippoRAG_v2"
EMBED_TAG = "NV-Embed-v2"
DATASETS = {"musique": "musique", "2wiki": "2wikimultihopqa", "hotpotqa": "hotpotqa"}
PROPRAG = {"musique": 78.3, "2wiki": 94.1, "hotpotqa": 97.4}
HIPPO2 = {"musique": 74.7, "2wiki": 90.4, "hotpotqa": 96.3}

# E1 GB 설정을 그대로 쓴다. 엣지만 바꾼다 — 그래야 이득의 원인이 엣지임이 분명하다.
DEPTH, WIDTH, NSEED, SCORE_MODE, MERGE = 3, 4, 5, 0, "append"
MAXK, MAXW = 20, 6
THRESHOLDS = (2, 3, 5, 10, 20)
EDGE_CAP = 8_000_000        # 이보다 많으면 만들지 않고 건너뛴다(메모리 보호)

_N = re.compile(r"[^a-z0-9 ]+")
_TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9.'\-]*")
_CAP = re.compile(r"^[A-Z][A-Za-z0-9.'\-]*$")
_YEAR = re.compile(r"^(1[0-9]{3}|20[0-9]{2})$")
_STOP = {"The", "A", "An", "He", "She", "It", "They", "This", "That", "In", "On",
         "At", "For", "As", "But", "And", "Or", "His", "Her", "Its", "Their",
         "After", "Before", "When", "While", "During", "However", "There",
         "These", "Those", "Some", "Both", "Also", "Then", "Since", "By", "Of",
         "From", "With", "To", "Was", "Is", "Are", "Were", "Has", "Have", "Had"}
BSPAN = 4


def norm(s):
    return " ".join(_N.sub(" ", s.lower()).split())


def spans(text):
    """D-BRIDGE와 **같은 추출기**여야 한다. 다르면 진단이 이 실험을 설명하지 못한다."""
    toks = _TOK.findall(text)
    out, i, n = set(), 0, len(toks)
    while i < n:
        if _YEAR.match(toks[i]):
            out.add(toks[i]); i += 1; continue
        if _CAP.match(toks[i]) and toks[i] not in _STOP:
            j = i
            while j < n and _CAP.match(toks[j]) and toks[j] not in _STOP and j - i < BSPAN:
                j += 1
            run = toks[i:j]
            for a in range(len(run)):
                for b in range(a + 1, min(a + BSPAN, len(run)) + 1):
                    s = " ".join(run[a:b])
                    if len(s) >= 3:
                        out.add(s)
            i = j
        else:
            i += 1
    return out


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
    edges = set()
    for i, body in enumerate(texts):
        toks = norm(body).split()
        for n in range(1, MAXW + 1):
            for a in range(len(toks) - n + 1):
                j = lookup.get(" ".join(toks[a:a + n]))
                if j is not None and j != i:
                    edges.add((i, j))
    return edges


def bridge_sets(titles, texts, T):
    """빈도 T 이하인 문자열만 다리로 쓴다. 반환은 (문자열 -> 문단 목록)."""
    S = [spans(t + " " + b) for t, b in zip(titles, texts)]
    post = defaultdict(list)
    for i, s in enumerate(S):
        for x in s:
            post[x].append(i)
    return {x: v for x, v in post.items() if 2 <= len(v) <= T}


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
                                    repo_type="dataset")).astype(np.float32)
        Q = np.load(hf_hub_download(RESULTS_REPO, f"emb/{ds}_{EMBED_TAG}_Q.npy",
                                    repo_type="dataset")).astype(np.float32)
        dense = DenseIndex(P.shape[1]); dense.add_batch(P)
        te = title_edges(titles, texts)
        post_all = bridge_sets(titles, texts, max(THRESHOLDS))
        QSIM = (P @ Q.T).astype(np.float32)
        hits = [dense.search(Q[i], MAXK) for i in range(len(questions))]
        prep[ds] = {"n": n, "questions": questions, "hits": hits, "QSIM": QSIM,
                    "title": te, "post": post_all}
        print(f"[{time.time()-t0:6.1f}s] {ds}: 문단 {n:,} 제목엣지 {len(te):,} "
              f"다리문자열(df<={max(THRESHOLDS)}) {len(post_all):,}")

    def make_graph(ds, mode, T):
        D = prep[ds]
        g = Graph(D["n"])
        es = set()
        if mode in ("title", "both"):
            es |= D["title"]
        if mode in ("bridge", "both"):
            for x, v in D["post"].items():
                if len(v) > T:
                    continue
                for a in range(len(v)):
                    for b in range(a + 1, len(v)):
                        es.add((v[a], v[b])); es.add((v[b], v[a]))
                if len(es) > EDGE_CAP:
                    return None, len(es)
        g.add_edges([(a, b, 1.0) for a, b in es])
        return g, len(es)

    def r5(ds, g, idxs):
        D = prep[ds]
        tot = m = 0.0
        for i in idxs:
            gold = D["questions"][i]["gold"]
            if not gold:
                continue
            h = D["hits"][i]
            qs = D["QSIM"][:, i]
            bs = g.beam_search([x for x, _ in h[:NSEED]], qs.tolist(),
                               DEPTH, WIDTH, 0.0, MAXK, SCORE_MODE)
            seen, out = set(), []
            for x, _ in bs:
                if x not in seen:
                    seen.add(x); out.append(x)
            for x, _ in h[:MAXK]:
                if x not in seen:
                    seen.add(x); out.append(x)
            R = set(out[:5])
            tot += sum(1 for v in gold if v in R) / len(gold); m += 1
        return tot / m * 100

    nq = len(prep["musique"]["questions"])
    dev = list(range(0, nq, 2))
    test = list(range(1, nq, 2))
    print(f"\ndev {len(dev)} / test {len(test)} (짝/홀 분할)")

    # ── 안전장치: title-only가 E1 기준선을 재현하는가 ──────────────────────
    print(f"\n{'='*78}\n안전장치 — title 전용이 E1 GB를 재현하는가 (전체 1000)\n{'='*78}")
    base_full = {}
    for ds in DATASETS:
        g, ne = make_graph(ds, "title", 0)
        base_full[ds] = r5(ds, g, list(range(nq)))
        print(f"  {ds:<10} 엣지 {ne:>9,}  R@5 {base_full[ds]:.2f}")

    # ── dev에서 T와 mode를 고른다 ─────────────────────────────────────────
    print(f"\n{'='*78}\ndev에서 탐색 — 엣지 수도 함께 본다 (희소성이 근간이다)\n{'='*78}")
    rows = []
    for mode in ("title", "bridge", "both"):
        for T in ((0,) if mode == "title" else THRESHOLDS):
            devs, edges, skip = {}, {}, False
            for ds in DATASETS:
                g, ne = make_graph(ds, mode, T)
                if g is None:
                    print(f"  {mode:<7} T={T:<3} {ds}: 엣지 {ne:,} > 상한 — 건너뜀")
                    skip = True; break
                edges[ds] = ne
                devs[ds] = r5(ds, g, dev)
            if skip:
                continue
            avg = sum(devs.values()) / 3
            rows.append({"mode": mode, "T": T, "dev": devs, "dev_avg": avg, "edges": edges})
            print(f"  {mode:<7} T={T:<3} 엣지 " +
                  " ".join(f"{ds[:4]}{edges[ds]:>8,}" for ds in DATASETS) +
                  f"  dev " + " ".join(f"{devs[ds]:5.1f}" for ds in DATASETS) +
                  f"  평균 {avg:.2f}")

    base_row = next(r for r in rows if r["mode"] == "title")
    top3 = sorted(rows, key=lambda x: -x["dev_avg"])[:3]

    print(f"\n{'='*78}\ndev 상위 3개를 test에서 잰다 — 선택 잡음을 드러낸다\n{'='*78}")
    out = {"base_full": base_full, "dev_rows": rows, "test": []}
    for rk, r in enumerate(top3, 1):
        te = {}
        for ds in DATASETS:
            g, _ = make_graph(ds, r["mode"], r["T"])
            te[ds] = r5(ds, g, test)
        ta = sum(te.values()) / 3
        out["test"].append({"rank": rk, "mode": r["mode"], "T": r["T"],
                            "test": te, "test_avg": ta, "edges": r["edges"]})
        print(f"\n[dev {rk}위] mode={r['mode']} T={r['T']}  "
              f"dev {r['dev_avg']:.2f} → test {ta:.2f} (하락 {r['dev_avg']-ta:+.2f})")
        print(f"  {'':<10}{'test':>8}{'HippoRAG2':>11}{'PropRAG':>9}{'엣지':>11}")
        for ds in DATASETS:
            mk = "  승" if te[ds] > PROPRAG[ds] else ""
            print(f"  {ds:<10}{te[ds]:>8.2f}{HIPPO2[ds]:>11}{PROPRAG[ds]:>9}"
                  f"{r['edges'][ds]:>11,}{mk}")
        print(f"  {'평균':<10}{ta:>8.2f}{sum(HIPPO2.values())/3:>11.1f}"
              f"{sum(PROPRAG.values())/3:>9.1f}")

    # title 기준선도 test에서 재야 비교가 성립한다
    bt = {}
    for ds in DATASETS:
        g, _ = make_graph(ds, "title", 0)
        bt[ds] = r5(ds, g, test)
    out["base_test"] = bt
    ba = sum(bt.values()) / 3
    print(f"\n기준선(title 전용) test: " +
          " ".join(f"{ds} {bt[ds]:.2f}" for ds in DATASETS) + f"  평균 {ba:.2f}")
    best = out["test"][0]
    print(f"**순수 이득: {best['test_avg']-ba:+.2f}%p** (test 기준, 같은 분할)")

    spread = max(x["test_avg"] for x in out["test"]) - min(x["test_avg"] for x in out["test"])
    print(f"상위 3개 test 편차 {spread:.2f}%p — " +
          ("선택 불안정, '이겼다'고 쓰지 않는다" if spread > 1.0 else "선택 안정"))

    Path("/tmp/bedge.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj="/tmp/bedge.json", path_in_repo="runs/bedge_e3.json",
                        repo_id=RESULTS_REPO, repo_type="dataset")


if __name__ == "__main__":
    main()
