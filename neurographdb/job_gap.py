# /// script
# requires-python = ">=3.10"
# dependencies = ["pybind11", "numpy", "huggingface_hub>=0.28"]
# ///
"""PropRAG과의 격차 진단 — 엣지 문제인가 순위 문제인가.

E1 결과 (Recall@5, 공식 코퍼스, NV-Embed-v2):
    우리 GB   75.1 / 93.5 / 96.2   (평균 88.3, LLM 0회)
    HippoRAG2 74.7 / 90.4 / 96.3   (평균 87.1, 문단당 LLM 2회)
    PropRAG   78.3 / 94.1 / 97.4   (평균 89.9, 문단당 LLM 2회)

격차: MuSiQue **−3.2** / HotpotQA −1.2 / 2Wiki −0.6.

**질문은 하나다.** 우리가 놓친 정답이
  (a) 그래프에 애초에 도달 경로가 없어서 못 찾았나  → 엣지 문제. LLM 명제가 이기는 지점.
  (b) 후보에는 있는데 5위 안에 못 올렸나            → 순위 문제. **LLM 없이 고칠 수 있다.**

이 비율이 격차를 닫을 수 있는지를 결정한다.

실패 질문을 이렇게 가른다:
    도달불가   정답이 dense 상위20에도 없고 그래프로도 3홉 안에 안 닿는다
    후보에있음  후보에는 들어왔는데 5위 밖 (= 순위 문제)
그리고 **오라클 상한**을 낸다: 후보 집합을 완벽히 재배열하면 R@5가 얼마인가.
그 값이 PropRAG의 78.3/94.1/97.4를 넘으면 **탐색만 고쳐도 이길 수 있다.**

임베딩은 E1이 캐시해 뒀다. **GPU 불필요, cpu-upgrade에서 무료로 돈다.**
"""

import json
import os
import re
import subprocess
import sys
import sysconfig
import time
from collections import deque
from pathlib import Path

CODE_REPO = "goethe0101/neurographdb"
RESULTS_REPO = "goethe0101/neurographdb-results"
OFFICIAL = "osunlp/HippoRAG_v2"
EMBED_TAG = "NV-Embed-v2"
DATASETS = {"musique": "musique", "2wiki": "2wikimultihopqa", "hotpotqa": "hotpotqa"}
PROPRAG = {"musique": 78.3, "2wiki": 94.1, "hotpotqa": 97.4}
OURS_GB = {"musique": 75.1, "2wiki": 93.5, "hotpotqa": 96.2}
N_SEEDS, MAXK, MAXW = 5, 20, 6
BEAM_D, BEAM_W = 3, 4
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
    corpus = json.load(open(hf_hub_download(OFFICIAL, f"{key}_corpus.json", repo_type="dataset")))
    qs = json.load(open(hf_hub_download(OFFICIAL, f"{key}.json", repo_type="dataset")))
    titles = [c["title"] for c in corpus]
    texts = [c["text"] for c in corpus]
    out = []
    for r in qs:
        if name == "musique":
            gold = sorted({p["title"] for p in r["paragraphs"] if p.get("is_supporting")})
        else:
            gold = sorted({sf[0] for sf in r["supporting_facts"]})
        out.append({"q": r["question"], "gold": gold})
    return titles, texts, out


def title_edges(titles, texts):
    lookup = {}
    for i, t in enumerate(titles):
        k = norm(t)
        if len(k) >= 5:
            lookup.setdefault(k, i)
    adj = {}
    for i, body in enumerate(texts):
        toks = norm(body).split()
        hit = set()
        for n in range(1, MAXW + 1):
            for a in range(len(toks) - n + 1):
                j = lookup.get(" ".join(toks[a:a + n]))
                if j is not None and j != i:
                    hit.add(j)
        if hit:
            adj[i] = sorted(hit)
    return adj


def reachable(adj, seeds, depth=BEAM_D):
    """seed에서 depth홉 안에 닿는 노드 전부. 빔 폭 제한 없이 — 도달 가능성의 상한이다."""
    seen = set(seeds)
    frontier = deque((s, 0) for s in seeds)
    while frontier:
        u, d = frontier.popleft()
        if d >= depth:
            continue
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                frontier.append((v, d + 1))
    return seen


def main():
    import numpy as np
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    build_core()
    from ngdb import DenseIndex, Graph

    report = {}
    for ds in DATASETS:
        titles, texts, questions = load_official(ds)
        tidx = {t: i for i, t in enumerate(titles)}
        n = len(titles)
        adj = title_edges(titles, texts)

        tag = f"emb/{ds}_{EMBED_TAG}"
        P = np.load(hf_hub_download(RESULTS_REPO, f"{tag}_P.npy", repo_type="dataset"))
        Q = np.load(hf_hub_download(RESULTS_REPO, f"{tag}_Q.npy", repo_type="dataset"))
        dense = DenseIndex(P.shape[1]); dense.add_batch(P)
        g = Graph(n)
        g.add_edges([(a, b, 1.0) for a, vs in adj.items() for b in vs])
        print(f"[{time.time()-t0:6.1f}s] {ds}: 문단 {n:,} 질문 {len(questions):,} "
              f"엣지 {sum(len(v) for v in adj.values()):,}")

        stat = {k: 0 for k in ("total", "ok", "fail", "unreachable", "in_cand")}
        miss_ranks = []
        r5_list, all5_list, oracle_r5_list = [], [], []
        for i, q in enumerate(questions):
            gold = [tidx[t] for t in q["gold"] if t in tidx]
            if not gold:
                continue
            stat["total"] += 1
            hits = dense.search(Q[i], MAXK)
            sd = [d for d, _ in hits[:N_SEEDS]]
            qsim = (P @ Q[i]).astype(np.float32)
            bs = g.beam_search(sd, qsim.tolist(), BEAM_D, BEAM_W, 0.0, MAXK)
            seen, order = set(), []
            for d, _ in bs:
                if d not in seen:
                    seen.add(d); order.append(d)
            for d, _ in hits:
                if d not in seen:
                    seen.add(d); order.append(d)
            order = order[:MAXK]

            # E1과 **같은 방식**으로 지표를 낸다. 통제군이 어긋나면 나머지를 못 믿는다.
            r5_list.append(sum(1 for gv in gold if gv in order[:5]) / len(gold))
            all5 = all(gv in order[:5] for gv in gold)
            all5_list.append(float(all5))

            # 후보 집합 = dense 상위20 ∪ 그래프 3홉 도달(빔 폭 제한 없음)
            cand = {d for d, _ in hits} | reachable(adj, sd)
            # **오라클도 Recall@5로 낸다.** 후보에서 최선의 5개를 고를 수 있다면
            # 정답 중 몇 개를 담을 수 있나. PropRAG의 78.3과 같은 지표여야 비교가 된다.
            oracle_r5_list.append(min(len(set(gold) & cand), 5) / len(gold))

            if all5:
                stat["ok"] += 1
                continue
            stat["fail"] += 1
            if all(gv in cand for gv in gold):
                stat["in_cand"] += 1
            else:
                stat["unreachable"] += 1
            for gv in gold:
                if gv not in order[:5]:
                    miss_ranks.append(order.index(gv) + 1 if gv in order else 99)

        t = max(stat["total"], 1)
        f = max(stat["fail"], 1)
        r5 = float(np.mean(r5_list)) * 100
        all5 = float(np.mean(all5_list)) * 100
        oracle_r5 = float(np.mean(oracle_r5_list)) * 100
        print(f"\n{'='*74}\n{ds} — GB 실패 층화 (격차 {OURS_GB[ds]} vs PropRAG {PROPRAG[ds]})\n{'='*74}")
        gap = abs(r5 - OURS_GB[ds])
        print(f"  통제군 대조: R@5 = {r5:.1f} (E1 기록 {OURS_GB[ds]}, 차이 {gap:.1f}%p)")
        if gap > 0.5:
            print(f"    ! E1과 어긋난다. 아래 층화를 신뢰할 수 없다 — 원인부터 찾을 것")
        print(f"  근거전부@5 = {all5:.1f}")
        print(f"  성공(정답 전부 5위 안)   {stat['ok']:>5} ({stat['ok']/t:>6.1%})")
        print(f"  실패                    {stat['fail']:>5} ({stat['fail']/t:>6.1%})")
        print(f"    ├ **후보엔 있음(순위 문제)** {stat['in_cand']:>5} "
              f"({stat['in_cand']/f:>6.1%} of 실패)")
        print(f"    └ 도달 불가(엣지 문제)      {stat['unreachable']:>5} "
              f"({stat['unreachable']/f:>6.1%} of 실패)")
        mr = np.array([r for r in miss_ranks if r < 99])
        n_out = sum(1 for r in miss_ranks if r >= 99)
        if len(mr):
            print(f"  놓친 정답이 상위20 안에 있을 때의 순위: 중앙 {np.median(mr):.0f}, "
                  f"10위 이내 {100*np.mean(mr <= 10):.0f}%")
        print(f"  놓친 정답이 상위20 밖: {n_out}건")
        print(f"\n  **오라클 상한 Recall@5 = {oracle_r5:.1f}** (후보에서 최선의 5개를 고를 때)")
        print(f"    우리 GB {r5:.1f} · PropRAG {PROPRAG[ds]} — 같은 지표로 비교")
        verdict = ("탐색만 고쳐도 PropRAG을 넘을 수 있다"
                   if oracle_r5 > PROPRAG[ds] else
                   "탐색을 완벽히 해도 PropRAG에 못 미친다 — 엣지가 부족하다")
        print(f"    → {verdict}")

        report[ds] = {**stat, "r5": r5, "all5": all5, "oracle_r5": oracle_r5,
                      "ours_gb": OURS_GB[ds], "proprag": PROPRAG[ds],
                      "verdict": verdict}

    out = Path("/tmp/gap_analysis.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=str(out), path_in_repo=f"runs/{out.name}",
                        repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"\n업로드: runs/{out.name}  ({(time.time()-t0)/60:.1f}분, GPU 없음)")


if __name__ == "__main__":
    main()
