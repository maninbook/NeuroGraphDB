# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11", "numpy", "pyarrow",
#     "transformers>=4.40", "torch", "huggingface_hub>=0.28",
# ]
# ///
"""F — ACT-R fan 할인. 검증된 확산 형식을 검색에 얹는다.

사전 측정에서 두 항이 갈렸다:
  합산     정답/잡음 수렴도가 1.06 / 1.03 / 0.98 — 변별력 없음. **쓰지 않는다.**
  fan 할인 정답 경로 출처의 fan이 잡음 경로보다 0.78 / 0.78 / 0.76배 — 세 번 다 일관.

ACT-R(Anderson): S_ji = S − ln(fan_j). 출처 j가 여러 곳에 뻗을수록 각각에 덜 준다.
Anderson & Reder의 fan effect 실험에서 나온 형태이고, 임의 상수가 아니다.

우리 확산은 곱셈이므로 정규화한 형태로 얹는다:

    F1  ACT-R형    w' = w × (S − ln(1+fan_u)) / S           S=3.0 고정
    F2  예산 보존  w' = w / fan_u                            (순수 나눗셈)
    F3  F1 + 게이팅 (게이팅은 이미 4회 재현된 개선이다)

fan은 **출처 u**의 출차수다. 목적지 v의 입차수(허브 억제)와는 다른 양이며,
그쪽은 데이터셋마다 부호가 뒤집혔지만 이쪽은 세 번 다 같은 방향이었다.

── 사전등록 ────────────────────────────────────────────────────────────────
주 가설   MuSiQue 근거2개@10에서 F1 > B0. 여지가 가장 크고 HippoRAG에 지는 곳이다.
          McNemar 대응 이분, α=0.05, n=1000.
부 가설   HotpotQA·2Wiki 동일 지표, F2·F3도 함께. 유의하지 않아도 보고한다.
안전장치  어느 데이터셋이든 R@10이 1.0%p 넘게 떨어지면 실패로 기록한다.
기전 예측 이득은 **출처 fan이 큰 경로로 잡음이 들어오던 질문**에 몰려야 한다.
파라미터  S=3.0으로 **미리 고정.** 게이팅은 job_gate.py 사전등록 값 그대로.
          돌린 뒤 재탐색하지 않는다.
실패 조건 유의하지 않으면 그대로 적고 S를 바꿔가며 재시도하지 않는다.
────────────────────────────────────────────────────────────────────────────

인자: DATASET N_QUESTIONS SEED
"""

import json
import math
import os
import re
import subprocess
import sys
import sysconfig
import time
from collections import defaultdict
from pathlib import Path

CODE_REPO = "goethe0101/neurographdb"
RESULTS_REPO = "goethe0101/neurographdb-results"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
N_SEEDS = 5
MAXK = 20
KS = (1, 2, 5, 10, 20)
MAXW = 6
S_ACTR = 3.0                              # 사전등록 값
LO, HI, FLOOR = 0.30, 0.75, 0.25          # job_gate.py 사전등록 값

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
    out = defaultdict(set)
    for i, body in enumerate(texts):
        toks = norm(body).split()
        for n in range(1, MAXW + 1):
            for a in range(len(toks) - n + 1):
                j = lookup.get(" ".join(toks[a:a + n]))
                if j is not None and j != i:
                    out[i].add(j)
    return [(a, b) for a, vs in out.items() for b in vs], {a: len(v) for a, v in out.items()}


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
    edges, fan = build_edges(titles, texts)
    print(f"[{time.time()-t0:6.1f}s] {DATASET}: 질문 {len(questions)} 풀 {len(titles)} "
          f"엣지 {len(edges):,}")

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

    P = encode([f"{t}. {x}" for t, x in zip(titles, texts)])
    dense = DenseIndex(P.shape[1]); dense.add_batch(P)
    Q = encode([QUERY_PREFIX + q["q"] for q in questions])
    hits = [dense.search(Q[i], MAXK) for i in range(len(questions))]
    print(f"[{time.time()-t0:6.1f}s] 벡터 색인 {dense.size}")

    # 출처별 fan 할인. 정규화해서 최대 1.0을 넘지 않게 둔다 — 통제군 대비 강화가
    # 아니라 **약화만** 하도록. 그래야 fan 할인 단독 효과가 섞이지 않는다.
    disc_actr = {u: (S_ACTR - math.log(1 + f)) / S_ACTR for u, f in fan.items()}
    disc_budget = {u: 1.0 / f for u, f in fan.items()}

    def rank(i, disc, gated):
        gt = None
        if gated:
            gt = np.clip((P @ Q[i] - LO) / (HI - LO), FLOOR, 1.0).astype(np.float32)
        w = []
        for a, b in edges:
            x = 1.0
            if disc is not None:
                x *= disc.get(a, 1.0)
            if gt is not None:
                x *= float(gt[b])
            w.append((a, b, x))
        g = Graph(len(titles)); g.add_edges(w)
        sh = hits[i][:N_SEEDS]
        acts = g.spread([d for d, _ in sh], [float(s) for _, s in sh],
                        3, 0.65, 0.02, MAXK)
        seen, out = set(), []
        for d, _ in acts:
            if d not in seen:
                seen.add(d); out.append(d)
        for d, _ in hits[i]:
            if d not in seen:
                seen.add(d); out.append(d)
        return [titles[d] for d in out]

    conds = {"B0_통제군": (None, False), "F1_ACTR": (disc_actr, False),
             "F2_예산": (disc_budget, False), "F3_ACTR+게이팅": (disc_actr, True)}
    per_q, agg = {}, {}
    for name, (disc, ga) in conds.items():
        rows = []
        for i, q in enumerate(questions):
            R = rank(i, disc, ga)[:MAXK]
            gold = q["gold"]
            r = {f"r@{k}": sum(1 for g in gold if g in R[:k]) / len(gold) for k in KS}
            r["all@10"] = float(all(g in R[:10] for g in gold))
            rows.append(r)
        per_q[name] = rows
        agg[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print(f"[{time.time()-t0:6.1f}s] {name} 완료")

    print(f"\n{'='*80}\nACT-R fan 할인 {DATASET} — S={S_ACTR} n={len(questions)} seed={SEED}"
          f"\n{'='*80}")
    print(f"{'조건':<16}" + "".join(f"{'R@'+str(k):>9}" for k in KS) + f"{'근거2개@10':>12}")
    for name, v in agg.items():
        print(f"{name:<16}" + "".join(f"{v[f'r@{k}']:>9.3f}" for k in KS)
              + f"{v['all@10']:>12.3f}")

    base = per_q["B0_통제군"]
    tests = {}
    print()
    for name in ("F1_ACTR", "F2_예산", "F3_ACTR+게이팅"):
        mc = mcnemar([(bool(a["all@10"]), bool(b["all@10"]))
                      for a, b in zip(base, per_q[name])])
        tests[name] = mc
        v = ("앞섬" if mc["gain"] > mc["loss"] and mc["p_value"] < 0.05 else
             "뒤짐" if mc["loss"] > mc["gain"] and mc["p_value"] < 0.05 else "차이 없음")
        print(f"{name} vs 통제군: 이득 {mc['gain']} 손실 {mc['loss']} "
              f"(불일치 {mc['n_discordant']}, p={mc['p_value']:.4f}) → {v}")

    drop = agg["B0_통제군"]["r@10"] - agg["F1_ACTR"]["r@10"]
    if drop > 0.01:
        print(f"\n! 안전장치 위반 — F1의 R@10이 {drop*100:.1f}%p 떨어졌다. 실패로 기록한다")

    payload = {"dataset": DATASET, "n_questions": len(questions),
               "pool_size": len(titles), "edges": len(edges), "S": S_ACTR,
               "gate": {"lo": LO, "hi": HI, "floor": FLOOR}, "seed": SEED,
               "results": agg, "mcnemar": tests, "runtime_sec": time.time() - t0}
    out = Path(f"/tmp/fan_{DATASET}_s{SEED}.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=str(out), path_in_repo=f"runs/{out.name}",
                        repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"\n업로드: runs/{out.name}")


if __name__ == "__main__":
    main()
