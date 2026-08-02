# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11", "numpy", "pyarrow",
#     "transformers>=4.40", "torch", "vllm>=0.10",
#     "huggingface_hub>=0.28",
# ]
# ///
"""R — 질의 게이팅을 켜고 MuSiQue에서 HippoRAG과 다시 붙는다.

MuSiQue는 우리가 유일하게 유의하게 지던 데이터셋이고(검색 0.394 vs 0.454,
QA EM 0.238 vs 0.272, p=0.019), 게이팅이 정확히 거기서만 이득을 냈다
(3시드 재현, 이득 56 / 손실 13). 그럼 격차가 줄었는지 본다.

조건 — 풀·질문·질문순서·임베더·LLM·프롬프트·top-k 전부 고정. 검색만 다르다.
    dense        조밀 벡터 단독
    graph        현재(게이팅 없음)
    graph+gate   질의 게이팅 켬 (lo=0.30 hi=0.75 floor=0.25, job_gate.py와 동일)
    hipporag     앞선 잡이 올려둔 순위를 그대로 읽는다

**이건 새 가설 검정이 아니라 이미 재현된 것의 위치 확인이다.**
게이팅 파라미터는 job_gate.py에서 사전등록한 값 그대로 쓰고 손대지 않는다.

인자: N_QUESTIONS TOP_K SEED
"""

import json
import os
import re
import string
import subprocess
import sys
import sysconfig
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

CODE_REPO = "goethe0101/neurographdb"
RESULTS_REPO = "goethe0101/neurographdb-results"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DATASET = "musique"
N_SEEDS = 5
MAXK = 20
KS = (1, 2, 5, 10, 20)
MAXW = 6
LO, HI, FLOOR = 0.30, 0.75, 0.25          # job_gate.py 사전등록 값

N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 500
TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0

PARQUET_REV = "refs/convert/parquet"
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


def load_pool(n_questions, seed):
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    f = hf_hub_download("dgslibisey/MuSiQue", "default/validation/0000.parquet",
                        repo_type="dataset", revision=PARQUET_REV)
    ds = pq.read_table(f).to_pylist()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:n_questions]
    pool, questions = {}, []
    for i in idx:
        row = ds[int(i)]
        passages = {p["title"]: p["paragraph_text"] for p in row["paragraphs"]}
        gold = sorted({p["title"] for p in row["paragraphs"] if p.get("is_supporting")})
        if not gold:
            continue
        for t, x in passages.items():
            pool.setdefault(t, x)
        questions.append({"q": row["question"], "gold": gold,
                          "answer": str(row["answer"])})
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
        edges.extend((i, j) for j in hit)
    return edges


def normalize_ans(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred, gold):
    return float(normalize_ans(pred) == normalize_ans(gold))


def f1(pred, gold):
    p, g = normalize_ans(pred).split(), normalize_ans(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


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

    titles, texts, questions = load_pool(N_Q, SEED)
    edges = build_edges(titles, texts)
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

    g_plain = Graph(len(titles))
    g_plain.add_edges([(a, b, 1.0) for a, b in edges])

    def spread_rank(i, gated):
        sh = hits[i][:N_SEEDS]
        if gated:
            gt = np.clip((P @ Q[i] - LO) / (HI - LO), FLOOR, 1.0).astype(np.float32)
            g = Graph(len(titles))
            g.add_edges([(a, b, float(gt[b])) for a, b in edges])
        else:
            g = g_plain
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

    rank = {"dense": [[titles[d] for d, _ in h] for h in hits],
            "graph": [spread_rank(i, False) for i in range(len(questions))],
            "graph+gate": [spread_rank(i, True) for i in range(len(questions))]}

    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(RESULTS_REPO, f"runs/hipporag_{DATASET}_n{N_Q}.json",
                            repo_type="dataset")
        hp = json.loads(Path(p).read_text())
        assert hp["pool_size"] == len(titles), "풀 크기 불일치 — 비교 불가"
        assert hp["questions"] == [q["q"] for q in questions], "질문 순서 불일치 — 비교 불가"
        rank["hipporag"] = hp["ranked_titles"]
        print(f"[{time.time()-t0:6.1f}s] HippoRAG 순위 로드 — 풀·질문 일치 확인")
    except Exception as e:
        print(f"  ! HippoRAG 순위 없음 ({type(e).__name__}) — 우리 조건끼리만 비교")

    print(f"\n{'='*76}\nMuSiQue 검색 — 풀 {len(titles)}, 질문 {len(questions)}\n{'='*76}")
    print(f"{'조건':<14}" + "".join(f"{'R@'+str(k):>9}" for k in KS) + f"{'근거2개@10':>12}")
    retr, per_all10 = {}, {}
    for name, R in rank.items():
        rows = []
        for i, q in enumerate(questions):
            gold = q["gold"]
            r = {f"r@{k}": sum(1 for g in gold if g in R[i][:k]) / len(gold) for k in KS}
            r["all@10"] = float(all(g in R[i][:10] for g in gold))
            rows.append(r)
        retr[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        per_all10[name] = [r["all@10"] for r in rows]
        print(f"{name:<14}" + "".join(f"{retr[name][f'r@{k}']:>9.3f}" for k in KS)
              + f"{retr[name]['all@10']:>12.3f}")

    if "hipporag" in rank:
        for a in ("graph", "graph+gate"):
            mc = mcnemar(list(zip([bool(x) for x in per_all10["hipporag"]],
                                  [bool(x) for x in per_all10[a]])))
            v = ("우리가 앞섬" if mc["gain"] > mc["loss"] and mc["p_value"] < 0.05 else
                 "HippoRAG이 앞섬" if mc["loss"] > mc["gain"] and mc["p_value"] < 0.05 else
                 "유의하지 않음")
            print(f"  {a} vs hipporag: 이득 {mc['gain']} 손실 {mc['loss']} "
                  f"p={mc['p_value']:.4f} → {v}")

    # 임베딩 모델을 내리고 vLLM을 올린다. 안 내리면 L4 24GB에서 EngineCore가 죽는다.
    # (검색 순위는 위에서 이미 다 뽑았고 게이트 계산도 numpy라 GPU가 필요 없다)
    del emodel
    torch.cuda.empty_cache()

    from vllm import LLM, SamplingParams
    llm = LLM(model=LLM_MODEL, dtype="float16", gpu_memory_utilization=0.80,
              max_model_len=8192, enforce_eager=True, seed=SEED)
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    body = {t: x for t, x in zip(titles, texts)}
    print(f"[{time.time()-t0:6.1f}s] LLM 로드 {LLM_MODEL}")

    def make_prompt(q, ctx_titles):
        ctx = "\n\n".join(f"[{n+1}] {t}: {body[t]}" for n, t in enumerate(ctx_titles))
        msg = ("Answer the question using ONLY the passages below.\n"
               "Reply with the short answer span only — no explanation, no sentence.\n\n"
               f"{ctx}\n\nQuestion: {q}\nAnswer:")
        return tok.apply_chat_template([{"role": "user", "content": msg}],
                                       tokenize=False, add_generation_prompt=True)

    qa, per_em = {}, {}
    for name, R in rank.items():
        prompts = [make_prompt(q["q"], R[i][:TOP_K]) for i, q in enumerate(questions)]
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=48))
        preds = [o.outputs[0].text.strip().split("\n")[0] for o in outs]
        ems = [em(p, q["answer"]) for p, q in zip(preds, questions)]
        qa[name] = {"EM": float(np.mean(ems)),
                    "F1": float(np.mean([f1(p, q["answer"])
                                         for p, q in zip(preds, questions)]))}
        per_em[name] = ems
        print(f"[{time.time()-t0:6.1f}s] {name}: EM {qa[name]['EM']:.3f} "
              f"F1 {qa[name]['F1']:.3f}")

    print(f"\n{'='*76}\nMuSiQue 종단 QA — {LLM_MODEL}, top-{TOP_K}\n{'='*76}")
    print(f"{'조건':<14}{'EM':>9}{'F1':>9}")
    for name, v in qa.items():
        print(f"{name:<14}{v['EM']:>9.3f}{v['F1']:>9.3f}")
    tests = {}
    names = list(qa)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            mc = mcnemar(list(zip([bool(x) for x in per_em[a]],
                                  [bool(x) for x in per_em[b]])))
            tests[f"{a} vs {b}"] = mc
            print(f"  {a} vs {b}: 이득 {mc['gain']} 손실 {mc['loss']} p={mc['p_value']:.4f}")

    payload = {"dataset": DATASET, "n_questions": len(questions),
               "pool_size": len(titles), "top_k": TOP_K, "seed": SEED,
               "gate": {"lo": LO, "hi": HI, "floor": FLOOR},
               "retrieval": retr, "qa": qa, "mcnemar": tests,
               "runtime_sec": time.time() - t0}
    out = Path(f"/tmp/rematch_{DATASET}_n{len(questions)}.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=str(out), path_in_repo=f"runs/{out.name}",
                        repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"\n업로드: runs/{out.name}")


if __name__ == "__main__":
    main()
