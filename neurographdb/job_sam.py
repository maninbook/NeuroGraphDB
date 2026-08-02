# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11", "numpy", "pyarrow",
#     "transformers>=4.40", "torch", "huggingface_hub>=0.28",
# ]
# ///
"""S — SAM식 단서 갱신. 회상이 다음 회상의 단서를 바꾼다. LLM 없이.

김기인 방향: 시각계의 측면 억제 말고 **논리 회로·생각의 흐름**.

형식은 SAM(Raaijmakers & Shiffrin 1981, 자유회상의 고전 모델):
    단서 = 맥락 + 방금 회상해낸 항목
    회상에 성공하면 그 항목이 단서에 합류해 다음 표집이 달라진다

우리 인출은 한 방이었다. "X 감독의 어머니는?"에서 어머니 문서는 X의 제목을 담지 않고
질문은 어머니 이름을 담지 않아 **어느 경로로도 못 갔다.** 감독 문단을 먼저 찾으면
그 내용이 단서가 된다.

반복 인출 자체는 새롭지 않다(IRCoT, Self-Ask, FLARE). **전부 LLM으로 다음 질의를
생성한다.** 우리 것은 LLM을 쓰지 않는다 — 단서는 벡터 합이다.

사전 측정(job_cue.py)에서 상한이 나왔다. 놓친 정답을 되찾는 비율:
    HotpotQA +1.9%p / 2Wiki +4.2%p / **MuSiQue +13.4%p** (실제 조건)
오라클(+13.5%p)과 실제 조건이 같다 — 어느 게 진짜 발판인지 알 필요가 없다.

**단 그 숫자는 2라운드 단독 회수다.** 실제로는 두 라운드를 한 목록으로 합쳐야 하고
칸을 두고 경쟁한다. 별칭 실험을 죽인 게 그 밀어내기였다. 그래서 합치기를 못 박는다.

── 사전등록 ────────────────────────────────────────────────────────────────
단서       cue = unit(q + p_top1). 1라운드 1위 문단을 질의 접두어와 함께 인코딩.
           오라클을 쓰지 않는다.
2라운드    cue로 dense 상위 20 → 그중 상위 5를 seed로 확산 (1라운드와 동일 규칙)
합치기     **1라운드 1~5위는 손대지 않는다.** 그 구간은 dense의 것이고 우리 이득은
           6~20위에서 나온다는 게 이미 확인됐다. 6위 이하만 두 라운드의 순위를
           RRF(k=60)로 융합한다. RRF는 튜닝 파라미터가 없어 사후 조정 여지가 없다.
주 가설    MuSiQue 근거2개@10에서 2라운드 > 1라운드. McNemar, α=0.05, n=1000.
부 가설    HotpotQA·2Wiki 동일 지표. 유의하지 않아도 보고한다.
안전장치   **어느 데이터셋이든 R@10이 1.0%p 넘게 떨어지면 실패로 기록한다.**
기전 예측  이득은 **1라운드에서 정답을 일부만 찾은 질문**에 몰려야 한다.
           발판이 없던 질문(MuSiQue 4.3%)에서는 이득이 없어야 한다.
실패 조건  유의하지 않으면 그대로 적고, 단서 가중치나 RRF k를 바꿔 재시도하지 않는다.
────────────────────────────────────────────────────────────────────────────

인자: DATASET N_QUESTIONS SEED
"""

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
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
N_SEEDS = 5
MAXK = 20
KS = (1, 2, 5, 10, 20)
MAXW = 6
KEEP = 5              # 1라운드 상위 5위는 고정 (사전등록)
RRF_K = 60            # 표준값, 튜닝하지 않음

DATASET = sys.argv[1] if len(sys.argv) > 1 else "musique"
N_Q = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0

PARQUET_REV = "refs/convert/parquet"
DATASETS = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor/validation/0000.parquet"),
    "2wiki":    ("framolfese/2WikiMultihopQA", "default/validation/0000.parquet"),
    "musique":  ("dgslibisey/MuSiQue", "default/validation/0000.parquet"),
}
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


def load_pool(dataset, n_questions, seed):
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    name, path = DATASETS[dataset]
    f = hf_hub_download(name, path, repo_type="dataset", revision=PARQUET_REV)
    ds = pq.read_table(f).to_pylist()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:n_questions]
    pool, questions = {}, []
    for i in idx:
        row = ds[int(i)]
        if dataset == "musique":
            passages = {p["title"]: p["paragraph_text"] for p in row["paragraphs"]}
            gold = sorted({p["title"] for p in row["paragraphs"] if p.get("is_supporting")})
        else:
            passages = {t: " ".join(s) for t, s in
                        zip(row["context"]["title"], row["context"]["sentences"])}
            gold = sorted(set(row["supporting_facts"]["title"]))
        if not gold:
            continue
        for t, x in passages.items():
            pool.setdefault(t, x)
        questions.append({"q": row["question"], "gold": gold})
    titles = list(pool)
    return titles, [pool[t] for t in titles], questions


def build_edges(titles, texts):
    lookup = {}
    for i, t in enumerate(titles):
        k = norm(t)
        if len(k) >= 5:
            lookup.setdefault(k, i)
    edges = []
    for i, body in enumerate(texts):
        toks = norm(body).split()
        hit = set()
        for n in range(1, MAXW + 1):
            for a in range(len(toks) - n + 1):
                j = lookup.get(" ".join(toks[a:a + n]))
                if j is not None and j != i:
                    hit.add(j)
        edges.extend((i, j, 1.0) for j in hit)
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


def main():
    import numpy as np
    t0 = time.time()
    build_core()
    from ngdb import DenseIndex, Graph

    titles, texts, questions = load_pool(DATASET, N_Q, SEED)
    tidx = {t: i for i, t in enumerate(titles)}
    g = Graph(len(titles)); g.add_edges(build_edges(titles, texts))
    print(f"[{time.time()-t0:6.1f}s] {DATASET}: 질문 {len(questions)} 풀 {len(titles)}")

    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    etok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    emodel = AutoModel.from_pretrained(EMBED_MODEL).to(dev).eval()

    def encode(items, batch=128):
        out = []
        for a in range(0, len(items), batch):
            b = etok(items[a:a + batch], padding=True, truncation=True,
                     max_length=512, return_tensors="pt").to(dev)
            with torch.no_grad():
                h = emodel(**b).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    docs = [f"{t}. {x}" for t, x in zip(titles, texts)]
    P = encode(docs)
    dense = DenseIndex(P.shape[1]); dense.add_batch(P)
    Q = encode([QUERY_PREFIX + q["q"] for q in questions])
    Pq = encode([QUERY_PREFIX + d for d in docs])     # 문단을 단서로 쓸 때의 인코딩
    print(f"[{time.time()-t0:6.1f}s] 벡터 색인 {dense.size}")

    def one_round(vec):
        hits = dense.search(vec, MAXK)
        sh = hits[:N_SEEDS]
        acts = g.spread([d for d, _ in sh], [float(s) for _, s in sh],
                        3, 0.65, 0.02, MAXK)
        seen, out = set(), []
        for d, _ in acts:
            if d not in seen:
                seen.add(d); out.append(d)
        for d, _ in hits:
            if d not in seen:
                seen.add(d); out.append(d)
        return out[:MAXK]

    def unit(v):
        n = np.linalg.norm(v)
        return (v / n if n > 0 else v).astype(np.float32)

    r1_all, r2_all = [], []
    for i in range(len(questions)):
        r1 = one_round(Q[i])
        r1_all.append(r1)
        cue = unit(Q[i] + Pq[r1[0]])          # SAM — 회상한 항목이 단서에 합류
        r2 = one_round(cue)
        # 합치기: 1라운드 1~5위 고정, 나머지는 RRF 융합
        head = r1[:KEEP]
        rr = {}
        for rank, d in enumerate(r1[KEEP:]):
            rr[d] = rr.get(d, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, d in enumerate(r2):
            if d in head:
                continue
            rr[d] = rr.get(d, 0.0) + 1.0 / (RRF_K + rank + 1)
        tail = [d for d, _ in sorted(rr.items(), key=lambda x: -x[1])]
        r2_all.append((head + tail)[:MAXK])

    def score(ranks):
        rows = []
        for i, q in enumerate(questions):
            R = [titles[d] for d in ranks[i]]
            gold = q["gold"]
            r = {f"r@{k}": sum(1 for gv in gold if gv in R[:k]) / len(gold) for k in KS}
            r["all@10"] = float(all(gv in R[:10] for gv in gold))
            rows.append(r)
        return rows

    rows1, rows2 = score(r1_all), score(r2_all)
    agg = lambda rs: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
    a1, a2 = agg(rows1), agg(rows2)

    print(f"\n{'='*78}\nSAM 단서 갱신 {DATASET} — n={len(questions)} seed={SEED}\n{'='*78}")
    print(f"{'조건':<14}" + "".join(f"{'R@'+str(k):>9}" for k in KS) + f"{'근거2개@10':>12}")
    for nm, v in (("1라운드", a1), ("2라운드", a2)):
        print(f"{nm:<14}" + "".join(f"{v[f'r@{k}']:>9.3f}" for k in KS)
              + f"{v['all@10']:>12.3f}")

    expected = {"hotpotqa": 0.940, "2wiki": 0.907, "musique": 0.350}
    print(f"\n통제군 대조: {a1['all@10']:.3f} (G1 기록 {expected[DATASET]:.3f})")

    mc = mcnemar([(bool(a["all@10"]), bool(b["all@10"])) for a, b in zip(rows1, rows2)])
    v = ("2라운드가 앞섬" if mc["gain"] > mc["loss"] and mc["p_value"] < 0.05 else
         "2라운드가 뒤짐" if mc["loss"] > mc["gain"] and mc["p_value"] < 0.05 else
         "유의하지 않음")
    print(f"\nMcNemar: 이득 {mc['gain']} 손실 {mc['loss']} "
          f"(불일치 {mc['n_discordant']}, p={mc['p_value']:.4f}) → {v}")

    # 기전 — 1라운드에서 정답을 일부만 찾은 질문에 이득이 몰렸는가
    part, none_, full = [], [], []
    for i, q in enumerate(questions):
        gold = [tidx[t] for t in q["gold"] if t in tidx]
        found = sum(1 for gv in gold if gv in r1_all[i])
        (full if found == len(gold) else none_ if found == 0 else part).append(i)
    print(f"\n기전 확인 (사전등록: 이득은 '일부만 찾은' 질문에 몰려야 한다)")
    for label, ids in (("일부만 찾음", part), ("발판 없음", none_), ("이미 전부", full)):
        if not ids:
            continue
        m = mcnemar([(bool(rows1[i]["all@10"]), bool(rows2[i]["all@10"])) for i in ids])
        print(f"  {label:<12} n={len(ids):>4}  이득 {m['gain']:>3} 손실 {m['loss']:>3}")

    drop = a1["r@10"] - a2["r@10"]
    if drop > 0.01:
        print(f"\n! 안전장치 위반 — R@10이 {drop*100:.1f}%p 떨어졌다. 실패로 기록한다")

    payload = {"dataset": DATASET, "n_questions": len(questions),
               "pool_size": len(titles), "keep": KEEP, "rrf_k": RRF_K, "seed": SEED,
               "round1": a1, "round2": a2, "mcnemar": mc, "verdict": v,
               "n_partial": len(part), "n_noanchor": len(none_),
               "runtime_sec": time.time() - t0}
    out = Path(f"/tmp/sam_{DATASET}_s{SEED}.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=str(out), path_in_repo=f"runs/{out.name}",
                        repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"\n업로드: runs/{out.name}")


if __name__ == "__main__":
    main()
