# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11", "numpy", "datasets>=3.2",
#     "sentence-transformers", "torch", "huggingface_hub>=0.28",
# ]
# ///
"""G1 — 그래프 채널이 조밀 벡터를 넘는가. 그리고 seed 희소성이 그 열쇠인가.

인자: DATASET(hotpotqa|2wiki|musique) N_QUESTIONS SEED

B0에서 목표선을 확정했다 (질문 1000, 풀 9798):
    BM25      근거2개@10 = 0.694
    조밀 벡터  근거2개@10 = 0.857   ← 넘어야 할 선
    하이브리드                0.864

진단(NEURODB_DIAGNOSIS.md)이 말한 것: NeuroDB는 **전부를 seed로 잡아** 그래프가
일할 자리가 없었다. 그래서 여기서 **seed 개수를 쓸어본다.** 이게 패턴 분리 축이다.

엣지는 LLM 없이 만든다 — **문단 A의 본문이 문단 B의 제목을 언급하면 A→B.**
HotpotQA의 bridge 질문이 정확히 이 구조다(1번 문단에서 언급된 개체를 따라 2번으로).
비용이 0이고, 그래서 이득이 나온다면 그건 그래프 구조 자체의 이득이다.

사전등록
  - 주 판정: graph 또는 hybrid의 **근거2개@10**이 dense를 넘는가
  - 같은 질문 집합을 여러 방법이 푸는 **대응 설계** → McNemar
  - seed 개수는 탐색 변수다. 최적 seed를 고른 뒤 그 값으로 판정하면 과적합이므로,
    **판정은 seed를 고정(=5)한 사전 지정 조건에서 한다.** 나머지는 탐색 결과로만 보고
  - 헤비안은 여기서 켜지 않는다. 1회성 QA라 강화할 사용 이력이 없다(설계문서 참조)
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
KS = (1, 2, 5, 10, 20)
SEED_SWEEP = (1, 2, 3, 5, 10, 20, 50, 200)   # 200 ≈ "거의 전부 seed" (NeuroDB의 상태)
PREREG_SEEDS = 5                              # 판정에 쓸 사전 지정값

DATASET = sys.argv[1] if len(sys.argv) > 1 else "hotpotqa"
N_Q = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0


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


# 세 벤치마크가 스키마가 다르다. 정답 단위는 셋 다 **문단 제목**으로 통일한다.
DATASETS = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", "validation"),
    "2wiki":    ("framolfese/2WikiMultihopQA", "default", "validation"),
    "musique":  ("dgslibisey/MuSiQue", "default", "validation"),
}


def _rows_hotpot_style(row):
    """HotpotQA·2Wiki 공통 — context{title,sentences} + supporting_facts{title}"""
    passages = {t: " ".join(s) for t, s in
                zip(row["context"]["title"], row["context"]["sentences"])}
    gold = sorted(set(row["supporting_facts"]["title"]))
    return passages, gold


def _rows_musique(row):
    """MuSiQue — paragraphs[{title, paragraph_text, is_supporting}]"""
    passages, gold = {}, []
    for p in row["paragraphs"]:
        passages.setdefault(p["title"], p["paragraph_text"])
        if p.get("is_supporting"):
            gold.append(p["title"])
    return passages, sorted(set(gold))


def load_pool(dataset, n_questions, seed):
    from datasets import load_dataset
    import numpy as np

    name, config, split = DATASETS[dataset]
    ds = load_dataset(name, config, split=split)
    parse = _rows_musique if dataset == "musique" else _rows_hotpot_style

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:n_questions]
    pool, questions = {}, []
    for i in idx:
        row = ds[int(i)]
        passages, gold = parse(row)
        if not gold:                     # 근거가 없는 항목은 평가 불가 — 건너뛴다
            continue
        for t, x in passages.items():
            pool.setdefault(t, x)
        questions.append({"q": row["question"], "gold": gold})
    titles = list(pool)
    return titles, [pool[t] for t in titles], questions


def build_mention_edges(titles, texts, max_title_words=6):
    """문단 본문이 다른 문단의 제목을 언급하면 엣지. LLM 비용 0.

    제목마다 정규식을 돌리면 문단 × 제목 = 9,600만 번 탐색이라 매우 느리다.
    대신 본문에서 n-gram 창을 뽑아 제목 사전에 조회한다. 문단 길이에만 비례한다.
    """
    norm = lambda s: re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    lookup = {}
    for i, t in enumerate(titles):
        key = " ".join(norm(t).split())
        if len(key) >= 5:            # 아주 짧은 제목은 오탐이 많다
            lookup.setdefault(key, i)

    edges = []
    for i, body in enumerate(texts):
        toks = norm(body).split()
        hit = set()
        for n in range(1, max_title_words + 1):
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
    print(f"[{time.time()-t0:6.1f}s] C++ 코어 빌드 완료")

    titles, texts, questions = load_pool(DATASET, N_Q, SEED)
    print(f"[{time.time()-t0:6.1f}s] {DATASET}: 질문 {len(questions)} | 풀 {len(titles)}")

    edges = build_mention_edges(titles, texts)
    g = Graph(len(titles))
    g.add_edges(edges)
    print(f"[{time.time()-t0:6.1f}s] 제목 언급 엣지 {g.n_edges:,}개 "
          f"(노드당 평균 {g.n_edges/max(len(titles),1):.1f})")

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=dev)
    dim = model.get_sentence_embedding_dimension()
    P = model.encode([f"{t}. {x}" for t, x in zip(titles, texts)], batch_size=128,
                     convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    dense = DenseIndex(dim); dense.add_batch(P)
    Q = model.encode([QUERY_PREFIX + q["q"] for q in questions], batch_size=128,
                     convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    print(f"[{time.time()-t0:6.1f}s] 벡터 색인 {dense.size} ({dim}차원, {dev})")

    maxk = max(KS)

    def evaluate(rank_fn):
        rows = []
        for i, q in enumerate(questions):
            ranked = [titles[d] for d in rank_fn(i)[:maxk]]
            r = {f"r@{k}": (sum(1 for gg in q["gold"] if gg in ranked[:k]) / len(q["gold"]))
                 for k in KS}
            r["all@10"] = float(all(gg in ranked[:10] for gg in q["gold"]))
            rows.append(r)
        return rows

    def agg(rows):
        out = {f"r@{k}": float(np.mean([r[f"r@{k}"] for r in rows])) for k in KS}
        out["all@10"] = float(np.mean([r["all@10"] for r in rows]))
        return out

    # ── dense 단독 (재현: B0의 0.857이 나와야 한다) ──────────────────────────
    dense_hits = [dense.search(Q[i], maxk) for i in range(len(questions))]
    dense_rows = evaluate(lambda i: [d for d, _ in dense_hits[i]])
    print(f"\n[{time.time()-t0:6.1f}s] dense 재현: 근거2개@10 = {agg(dense_rows)['all@10']:.3f}")

    # ── seed 개수 쓸기 — 패턴 분리 축 ───────────────────────────────────────
    print(f"\n{'='*78}\nseed 희소성 쓸기 (그래프 단독 / 벡터+그래프 혼합)\n{'='*78}")
    print(f"{'seeds':>6}{'graph R@10':>13}{'graph 2개@10':>15}{'hybrid R@10':>13}{'hybrid 2개@10':>15}")
    sweep, keep_rows = {}, {}
    for ns in SEED_SWEEP:
        def rank_graph(i, ns=ns):
            hits = dense_hits[i][:ns]
            acts = g.spread([d for d, _ in hits], [float(s) for _, s in hits],
                            3, 0.65, 0.02, maxk)
            seen, out = set(), []
            for d, _ in acts:
                if d not in seen:
                    seen.add(d); out.append(d)
            for d, _ in dense_hits[i]:          # 확산이 부족하면 벡터 순위로 채운다
                if d not in seen:
                    seen.add(d); out.append(d)
            return out

        def rank_hybrid(i, ns=ns):
            hits = dense_hits[i][:ns]
            acts = dict(g.spread([d for d, _ in hits], [float(s) for _, s in hits],
                                 3, 0.65, 0.02, 0))
            comb = {}
            for d, s in dense_hits[i]:
                comb[d] = 0.7 * float(s) + 0.3 * acts.get(d, 0.0)
            for d, a in acts.items():
                comb.setdefault(d, 0.3 * a)
            return [d for d, _ in sorted(comb.items(), key=lambda x: -x[1])]

        gr, hr = evaluate(rank_graph), evaluate(rank_hybrid)
        ga, ha = agg(gr), agg(hr)
        sweep[ns] = {"graph": ga, "hybrid": ha}
        keep_rows[ns] = (gr, hr)
        print(f"{ns:>6}{ga['r@10']:>13.3f}{ga['all@10']:>15.3f}"
              f"{ha['r@10']:>13.3f}{ha['all@10']:>15.3f}")

    # ── 사전 지정 조건에서 판정 ─────────────────────────────────────────────
    gr, hr = keep_rows[PREREG_SEEDS]
    d_ok = [bool(r["all@10"]) for r in dense_rows]
    print(f"\n{'='*78}\n판정 (사전 지정 seeds={PREREG_SEEDS}, 근거2개@10)\n{'='*78}")
    verdicts = {}
    for name, rows in (("graph", gr), ("hybrid", hr)):
        ok = [bool(r["all@10"]) for r in rows]
        mc = mcnemar(list(zip(d_ok, ok)))
        base, new = agg(dense_rows)["all@10"], agg(rows)["all@10"]
        if mc["gain"] > mc["loss"] and mc["p_value"] < 0.05:
            v = f"{name}가 dense를 넘음"
        elif mc["loss"] > mc["gain"] and mc["p_value"] < 0.05:
            v = f"{name}가 dense보다 나쁨"
        elif mc["n_discordant"] < 10:
            v = f"판정 불가 — 불일치 {mc['n_discordant']}개"
        else:
            v = "유의하지 않음"
        verdicts[name] = {"mcnemar": mc, "verdict": v, "dense": base, name: new}
        print(f"  dense {base:.3f} → {name} {new:.3f} | 이득 {mc['gain']} 손실 {mc['loss']} "
              f"(불일치 {mc['n_discordant']}, p={mc['p_value']:.4f}) → {v}")

    payload = {"dataset": DATASET, "n_questions": len(questions), "pool_size": len(titles),
               "n_edges": g.n_edges, "seed": SEED, "embed_model": EMBED_MODEL,
               "prereg_seeds": PREREG_SEEDS, "dense": agg(dense_rows),
               "sweep": {str(k): v for k, v in sweep.items()},
               "verdicts": verdicts, "runtime_sec": time.time() - t0}
    out = Path(f"/tmp/graph_{DATASET}_n{len(questions)}.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True, private=True)
    api.upload_file(path_or_fileobj=str(out), path_in_repo=f"runs/{out.name}",
                    repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"\n업로드: https://huggingface.co/datasets/{RESULTS_REPO}")


if __name__ == "__main__":
    main()
