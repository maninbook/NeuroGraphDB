# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11", "numpy", "torch",
#     # NV-Embed-v2의 원격 코드는 구 transformers API(_tied_weights_keys)를 쓴다.
#     # 최신 버전에서는 all_tied_weights_keys를 찾다가 AttributeError로 죽는다.
#     # 모델 카드가 지정한 버전으로 고정한다. 임베더를 바꾸면 재현 관문이 무의미해진다.
#     "transformers==4.42.4",
#     "sentencepiece", "protobuf", "einops", "huggingface_hub>=0.28",
#     # NV-Embed-v2의 원격 코드가 datasets를 import한다. 4.x부터 torchcodec을
#     # 끌어오는데 컨테이너에 FFmpeg 공유 라이브러리가 없어 죽는다(세션 초반에 겪음).
#     # 그 앞 버전으로 고정한다.
#     "datasets>=2.14,<4",
# ]
# ///
"""E1 — 공식 표준 환경에서 SOTA와 붙는다. 사전등록은 ../SOTA.md.

표준 환경(Gutiérrez et al. 2025 = HippoRAG 2 팀 배포, PropRAG·SAG가 그대로 씀):
    코퍼스 osunlp/HippoRAG_v2 — MuSiQue 11,656 / 2Wiki 6,119 / HotpotQA 9,811
    임베더 nvidia/NV-Embed-v2 (7B) · 지표 문단 Recall@2 / Recall@5

이겨야 할 Recall@5:
    NV-Embed-v2 단독  69.7 / 76.5 / 94.5
    HippoRAG 2       74.7 / 90.4 / 96.3   (색인에 문단당 LLM 2회)
    PropRAG          78.3 / 94.1 / 97.4   (색인에 문단당 LLM 2회, 40분)

우리는 **색인에도 질의에도 LLM을 쓰지 않는다.**

조건 (전부 같은 임베딩 위에서, 검색 방식만 다름)
    D    NV-Embed-v2 단독            — 그들의 69.7/76.5/94.5를 재현해야 한다
    G    D + 제목언급 그래프 확산      — 우리 기존 방식
    GK   G + 임베딩 kNN 엣지          — 연결률 23.4→53.1%(MuSiQue)로 올린 그것
    GKB  GK + 경로 빔서치             — PropRAG 이득의 큰 쪽(+2.7%p), LLM 불필요
    GKBQ GKB + 질의 게이팅            — 밀도가 올라가면 필요해진다고 예측했던 것

**첫 관문 — 재현 확인.** D가 공개값에서 ±2.0%p를 벗어나면 설정이 다른 것이므로
거기서 멈춘다. 결과 해석으로 넘어가지 않는다.

파라미터는 전부 **실행 전 고정**. 게이팅 lo/hi/floor는 이전 사전등록 값 그대로.
kNN은 k=10 문턱없음(연결률 측정에서 세 데이터셋 모두 관문을 넘긴 지점).
빔서치는 PropRAG과 같은 깊이 3·폭 4.

인자: DATASET
"""

import json
import os
import subprocess
import sys
import sysconfig
import time
from itertools import permutations
from pathlib import Path

CODE_REPO = "goethe0101/neurographdb"
RESULTS_REPO = "goethe0101/neurographdb-results"
OFFICIAL = "osunlp/HippoRAG_v2"
EMBED_MODEL = "nvidia/NV-Embed-v2"
DATASETS = {"musique": "musique", "2wiki": "2wikimultihopqa", "hotpotqa": "hotpotqa"}
PUBLISHED_D = {"musique": 69.7, "2wiki": 76.5, "hotpotqa": 94.5}   # NV-Embed-v2 단독 R@5
REPRO_TOL = 2.0

N_SEEDS = 5
MAXK = 20
KS = (2, 5, 10, 20)
MAXW = 6
KNN_K, KNN_TH = 10, 0.0          # 사전등록
BEAM_D, BEAM_W = 3, 4            # PropRAG과 동일
LO, HI, FLOOR = 0.30, 0.75, 0.25 # 이전 사전등록 값

DATASET = sys.argv[1] if len(sys.argv) > 1 else "musique"
import re
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
        out.append({"q": r["question"], "gold": gold, "answer": str(r.get("answer", ""))})
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


def mcnemar(pairs):
    from math import comb
    b = sum(1 for a, x in pairs if not a and x)
    c = sum(1 for a, x in pairs if a and not x)
    n = b + c
    if n == 0:
        return {"gain": 0, "loss": 0, "n_discordant": 0, "p_value": 1.0}
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return {"gain": b, "loss": c, "n_discordant": n, "p_value": min(1.0, 2 * tail)}


def load_or_encode(dataset, titles, texts, questions, encode, instr, t0):
    """임베딩을 Hub에 캐시한다. **한 번만 GPU를 쓰고 이후 실험은 CPU로 돈다.**

    이 프로젝트 GPU 비용의 대부분(l4x1 9.4시간)이 같은 임베딩을 잡마다 다시
    계산한 값이었다. 벡터가 있으면 내적도 그래프 순회도 CPU로 충분하다.
    """
    import numpy as np
    from huggingface_hub import HfApi, hf_hub_download

    tag = f"emb/{dataset}_{EMBED_MODEL.split('/')[-1]}"
    try:
        pp = hf_hub_download(RESULTS_REPO, f"{tag}_P.npy", repo_type="dataset")
        qp = hf_hub_download(RESULTS_REPO, f"{tag}_Q.npy", repo_type="dataset")
        P, Q = np.load(pp), np.load(qp)
        if P.shape[0] == len(titles) and Q.shape[0] == len(questions):
            print(f"[{time.time()-t0:6.1f}s] 임베딩 캐시 적중 — GPU 인코딩 건너뜀 "
                  f"(P {P.shape}, Q {Q.shape})")
            return P, Q
        print(f"  캐시 크기 불일치 — 다시 인코딩한다")
    except Exception:
        print(f"[{time.time()-t0:6.1f}s] 임베딩 캐시 없음 — 이번 한 번만 인코딩한다")

    P = encode([f"{t}. {x}" for t, x in zip(titles, texts)])
    Q = encode([q["q"] for q in questions], instruction=instr)
    api = HfApi()
    for arr, suffix in ((P, "_P.npy"), (Q, "_Q.npy")):
        f = Path(f"/tmp/{dataset}{suffix}")
        np.save(f, arr)
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f"{tag}{suffix}",
                        repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"[{time.time()-t0:6.1f}s] 임베딩 캐시 업로드 — 다음부터 CPU로 돈다")
    return P, Q


def main():
    import numpy as np
    import torch
    t0 = time.time()
    build_core()
    from ngdb import DenseIndex, Graph

    titles, texts, questions = load_official(DATASET)
    tidx = {t: i for i, t in enumerate(titles)}
    n = len(titles)
    te = title_edges(titles, texts)
    print(f"[{time.time()-t0:6.1f}s] {DATASET} 공식: 문단 {n:,} 질문 {len(questions):,} "
          f"제목엣지 {len(te):,}")

    # NV-Embed-v2는 trust_remote_code가 필요하고 질의에 instruction을 붙인다.
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = AutoModel.from_pretrained(EMBED_MODEL, trust_remote_code=True,
                                    torch_dtype=torch.float16).to(dev).eval()
    INSTR = "Instruct: Given a question, retrieve documents that best answer the question\nQuery: "

    def encode(items, instruction="", batch=8):
        out = []
        for a in range(0, len(items), batch):
            with torch.no_grad():
                e = mdl.encode(items[a:a + batch], instruction=instruction,
                               max_length=512)
                e = torch.nn.functional.normalize(e, p=2, dim=1)
            out.append(e.float().cpu().numpy())
            if (a // batch) % 50 == 0:
                print(f"    [{time.time()-t0:6.1f}s] 인코딩 {a}/{len(items)}")
        return np.concatenate(out).astype(np.float32)

    P, Q = load_or_encode(DATASET, titles, texts, questions, encode, INSTR, t0)
    dense = DenseIndex(P.shape[1]); dense.add_batch(P)
    print(f"[{time.time()-t0:6.1f}s] 벡터 색인 {dense.size} ({P.shape[1]}차원)")

    hits = [dense.search(Q[i], MAXK) for i in range(len(questions))]

    # kNN 엣지 — 이미 있는 임베딩의 내적이라 추가 비용이 없다
    knn = set()
    B = 512
    for a in range(0, n, B):
        sim = P[a:a+B] @ P.T
        for r in range(sim.shape[0]):
            sim[r, a+r] = -1.0
        ix = np.argpartition(-sim, KNN_K, axis=1)[:, :KNN_K]
        for r in range(sim.shape[0]):
            for c in ix[r]:
                if sim[r, c] >= KNN_TH:
                    knn.add((a + r, int(c)))
    print(f"[{time.time()-t0:6.1f}s] kNN 엣지 {len(knn):,} (k={KNN_K}, th={KNN_TH})")

    g_t = Graph(n); g_t.add_edges([(a, b, 1.0) for a, b in te])
    g_tk = Graph(n); g_tk.add_edges([(a, b, 1.0) for a, b in (te | knn)])

    def merge(order, i):
        seen, out = set(), []
        for d in order:
            if d not in seen:
                seen.add(d); out.append(d)
        for d, _ in hits[i]:
            if d not in seen:
                seen.add(d); out.append(d)
        return out[:MAXK]

    def rank(cond, i):
        h = hits[i]
        if cond == "D":
            return [d for d, _ in h][:MAXK]
        sh = h[:N_SEEDS]
        sd = [d for d, _ in sh]; sa = [float(s) for _, s in sh]
        if cond == "G":
            return merge([d for d, _ in g_t.spread(sd, sa, 3, 0.65, 0.02, MAXK)], i)
        qsim = (P @ Q[i]).astype(np.float32)
        if cond == "GK":
            return merge([d for d, _ in g_tk.spread(sd, sa, 3, 0.65, 0.02, MAXK)], i)
        if cond == "GKB":
            return merge([d for d, _ in g_tk.beam_search(sd, qsim.tolist(),
                                                         BEAM_D, BEAM_W, 0.0, MAXK)], i)
        if cond == "GKBQ":
            gt = np.clip((qsim - LO) / (HI - LO), FLOOR, 1.0)
            gq = Graph(n); gq.add_edges([(a, b, float(gt[b])) for a, b in (te | knn)])
            return merge([d for d, _ in gq.beam_search(sd, qsim.tolist(),
                                                       BEAM_D, BEAM_W, 0.0, MAXK)], i)
        raise ValueError(cond)

    conds = ["D", "G", "GK", "GKB", "GKBQ"]
    rows, per_q = {}, {}
    for c in conds:
        rs = []
        for i, q in enumerate(questions):
            R = [titles[d] for d in rank(c, i)]
            gold = q["gold"]
            r = {f"r@{k}": sum(1 for gv in gold if gv in R[:k]) / len(gold) for k in KS}
            r["all@5"] = float(all(gv in R[:5] for gv in gold))
            rs.append(r)
        rows[c] = {k: float(np.mean([x[k] for x in rs])) for k in rs[0]}
        per_q[c] = [x["r@5"] for x in rs]
        print(f"[{time.time()-t0:6.1f}s] {c} 완료  R@5 {rows[c]['r@5']*100:.1f}")

    d5 = rows["D"]["r@5"] * 100
    pub = PUBLISHED_D[DATASET]
    gap = abs(d5 - pub)
    print(f"\n{'='*76}\nE1 {DATASET} — 공식 코퍼스 + {EMBED_MODEL}\n{'='*76}")
    print(f"재현 관문: D의 R@5 = {d5:.1f} (공개값 {pub}, 차이 {gap:.1f}%p, 허용 {REPRO_TOL})")
    if gap > REPRO_TOL:
        print("  !! 관문 실패 — 설정이 다르다. 여기서 멈추고 원인을 찾는다.")
    else:
        print("  통과 — 아래 비교를 해석해도 된다.")

    print(f"\n{'조건':<8}" + "".join(f"{'R@'+str(k):>9}" for k in KS) + f"{'근거전부@5':>12}")
    for c in conds:
        print(f"{c:<8}" + "".join(f"{rows[c][f'r@{k}']*100:>9.1f}" for k in KS)
              + f"{rows[c]['all@5']*100:>12.1f}")

    print(f"\n공개 수치 (같은 코퍼스·같은 임베더)")
    for nm, v in [("NV-Embed-v2 단독", PUBLISHED_D[DATASET]),
                  ("HippoRAG 2", {"musique":74.7,"2wiki":90.4,"hotpotqa":96.3}[DATASET]),
                  ("PropRAG", {"musique":78.3,"2wiki":94.1,"hotpotqa":97.4}[DATASET])]:
        print(f"  {nm:<20} R@5 {v}")

    tests = {}
    print()
    for c in conds[1:]:
        mc = mcnemar([(a >= 1.0, b >= 1.0) for a, b in zip(per_q["D"], per_q[c])])
        tests[f"D vs {c}"] = mc
        v = ("앞섬" if mc["gain"] > mc["loss"] and mc["p_value"] < 0.05 else
             "뒤짐" if mc["loss"] > mc["gain"] and mc["p_value"] < 0.05 else "차이 없음")
        print(f"D vs {c}: 이득 {mc['gain']} 손실 {mc['loss']} p={mc['p_value']:.4f} → {v}")

    payload = {"dataset": DATASET, "corpus": OFFICIAL, "embed_model": EMBED_MODEL,
               "n_passages": n, "n_questions": len(questions),
               "title_edges": len(te), "knn_edges": len(knn),
               "knn_k": KNN_K, "knn_th": KNN_TH, "beam": [BEAM_D, BEAM_W],
               "gate": [LO, HI, FLOOR],
               "reproduction": {"ours_r5": d5, "published_r5": pub, "gap": gap,
                                "pass": bool(gap <= REPRO_TOL)},
               "results": rows, "mcnemar": tests, "runtime_sec": time.time() - t0}
    out = Path(f"/tmp/e1_{DATASET}_nvembed.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=str(out), path_in_repo=f"runs/{out.name}",
                        repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"\n업로드: runs/{out.name}")


if __name__ == "__main__":
    main()
