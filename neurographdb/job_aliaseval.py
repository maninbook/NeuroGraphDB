# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11", "numpy", "pyarrow",
#     "transformers>=4.40", "torch", "huggingface_hub>=0.28",
# ]
# ///
"""E — 제목 별칭 키가 검색을 실제로 개선하는가. 사전등록 포함.

진단 경로: 근거쌍의 39/31/68%가 어떤 홉으로도 안 닿았다 → 왜 안 닿나 표본을 읽었다
→ 위키의 **동음이의 괄호**와 **쉼표 접미사** 때문에 개체가 본문에 있는데도
   매칭이 안 되고 있었다 ('Test (wrestler)'는 본문에 "Test"로만 나온다)
→ 별칭 키를 추가하면 정답쌍 연결이 61.0→69.2 / 68.8→73.6 / 32.4→35.2%로 오르고
   lift는 3587→1628 / 4437→1599 / 952→774로 떨어진다(그래도 압도적)

**연결이 늘었다고 검색이 좋아지는 건 아니다.** 이웃이 늘면 확산이 분산된다.
이 프로젝트에서 "구조를 늘리면 좋아질 것"이라는 예상이 세 번 틀렸다
(헤비안, 문단 술어, 논항 공유). 그래서 돌리기 전에 적는다.

── 사전등록 ────────────────────────────────────────────────────────────────
주 가설   HotpotQA 근거2개@10에서 별칭엣지 > 현재엣지. 연결 이득이 가장 크다(+8.2%p).
          McNemar 대응 이분, α=0.05, n=1000. 검정을 바꾸지 않는다.
부 가설   2Wiki·MuSiQue 동일 지표. 유의하지 않아도 보고한다.
안전장치  **어느 데이터셋이든 R@10이 1.0%p 넘게 떨어지면 실패로 기록한다.**
          한 곳을 위해 다른 곳을 망가뜨리는 건 개선이 아니다.
기전 예측 이득은 **별칭 덕분에 새로 연결된 질문**에 몰려야 한다.
          거기가 아니면 설명이 틀린 것이고 숫자가 좋아도 그렇게 적는다.
실패 조건 유의하지 않으면 "제목 별칭은 연결을 늘리지만 검색을 개선하지 않는다"로
          적고, 별칭 규칙을 바꿔가며 재시도하지 않는다.
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
from itertools import permutations
from pathlib import Path

CODE_REPO = "goethe0101/neurographdb"
RESULTS_REPO = "goethe0101/neurographdb-results"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
N_SEEDS = 5
MAXK = 20
KS = (1, 2, 5, 10, 20)
MAXW = 6

DATASET = sys.argv[1] if len(sys.argv) > 1 else "hotpotqa"
N_Q = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0

PARQUET_REV = "refs/convert/parquet"
DATASETS = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor/validation/0000.parquet"),
    "2wiki":    ("framolfese/2WikiMultihopQA", "default/validation/0000.parquet"),
    "musique":  ("dgslibisey/MuSiQue", "default/validation/0000.parquet"),
}
_N = re.compile(r"[^a-z0-9 ]+")
_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_COMMA = re.compile(r",\s*[^,]+$")


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


def keys_base(t):
    k = norm(t)
    return [k] if len(k) >= 5 else []


def keys_alias(t):
    """원 제목 키는 유지하고 별칭을 **추가**한다. 기존 매칭을 잃지 않기 위해서다."""
    out = keys_base(t)
    s = _COMMA.sub("", _PAREN.sub("", t))
    if s != t:
        k = norm(s)
        if len(k) >= 5 and k not in out:
            out.append(k)
    return out


def build_edges(titles, texts, keyfn):
    lookup = {}
    for i, t in enumerate(titles):
        for k in keyfn(t):
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


def main():
    import numpy as np
    t0 = time.time()
    build_core()
    from ngdb import DenseIndex, Graph

    titles, texts, questions = load_pool(DATASET, N_Q, SEED)
    tidx = {t: i for i, t in enumerate(titles)}
    e_base = build_edges(titles, texts, keys_base)
    e_alias = build_edges(titles, texts, keys_alias)
    print(f"[{time.time()-t0:6.1f}s] {DATASET}: 질문 {len(questions)} 풀 {len(titles)} "
          f"| 엣지 {len(e_base):,} → {len(e_alias):,}")

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

    def make_graph(edges):
        g = Graph(len(titles))
        g.add_edges([(a, b, 1.0) for a, b in edges])
        return g

    def evaluate(g):
        rows = []
        for i in range(len(questions)):
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
            ranked = [titles[d] for d in out[:MAXK]]
            gold = questions[i]["gold"]
            r = {f"r@{k}": sum(1 for gg in gold if gg in ranked[:k]) / len(gold) for k in KS}
            r["all@10"] = float(all(gg in ranked[:10] for gg in gold))
            rows.append(r)
        return rows

    rows_base = evaluate(make_graph(e_base))
    rows_alias = evaluate(make_graph(e_alias))
    agg = lambda rs: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
    a_base, a_alias = agg(rows_base), agg(rows_alias)

    print(f"\n{'='*78}\n제목 별칭 {DATASET} — n={len(questions)}\n{'='*78}")
    print(f"{'조건':<12}" + "".join(f"{'R@'+str(k):>9}" for k in KS) + f"{'근거2개@10':>12}")
    for nm, v in (("현재 엣지", a_base), ("별칭 엣지", a_alias)):
        print(f"{nm:<12}" + "".join(f"{v[f'r@{k}']:>9.3f}" for k in KS)
              + f"{v['all@10']:>12.3f}")

    mc = mcnemar([(bool(a["all@10"]), bool(b["all@10"]))
                  for a, b in zip(rows_base, rows_alias)])
    verdict = ("별칭이 앞섬" if mc["gain"] > mc["loss"] and mc["p_value"] < 0.05 else
               "별칭이 뒤짐" if mc["loss"] > mc["gain"] and mc["p_value"] < 0.05 else
               "유의하지 않음")
    print(f"\nMcNemar: 이득 {mc['gain']} 손실 {mc['loss']} "
          f"(불일치 {mc['n_discordant']}, p={mc['p_value']:.4f}) → {verdict}")

    # ── 기전 예측 — 이득이 새로 연결된 질문에 몰렸는가 ──────────────────────
    newly, already = [], []
    for i, q in enumerate(questions):
        g = [tidx[t] for t in q["gold"] if t in tidx]
        if len(g) < 2:
            continue
        was = any((a, b) in e_base for a, b in permutations(g, 2))
        now = any((a, b) in e_alias for a, b in permutations(g, 2))
        (newly if (now and not was) else already).append(i)
    print(f"\n기전 확인 (사전등록: 이득은 새로 연결된 질문에 몰려야 한다)")
    for label, ids in (("별칭으로 새로 연결됨", newly), ("그 외", already)):
        if not ids:
            continue
        m = mcnemar([(bool(rows_base[i]["all@10"]), bool(rows_alias[i]["all@10"]))
                     for i in ids])
        print(f"  {label:<22} n={len(ids):>4}  이득 {m['gain']:>3} 손실 {m['loss']:>3}")

    drop = a_base["r@10"] - a_alias["r@10"]
    if drop > 0.01:
        print(f"\n! 안전장치 위반 — R@10이 {drop*100:.1f}%p 떨어졌다. 실패로 기록한다")

    payload = {"dataset": DATASET, "n_questions": len(questions),
               "pool_size": len(titles), "seed": SEED,
               "edges_base": len(e_base), "edges_alias": len(e_alias),
               "base": a_base, "alias": a_alias, "mcnemar": mc, "verdict": verdict,
               "n_newly_linked": len(newly), "runtime_sec": time.time() - t0}
    out = Path(f"/tmp/alias_{DATASET}_n{len(questions)}.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True, private=True)
    api.upload_file(path_or_fileobj=str(out), path_in_repo=f"runs/{out.name}",
                    repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"\n업로드: runs/{out.name}")


if __name__ == "__main__":
    main()
