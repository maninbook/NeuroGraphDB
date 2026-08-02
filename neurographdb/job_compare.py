# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11", "numpy", "pyarrow",
#     "transformers>=4.40", "torch", "vllm>=0.10",
#     "huggingface_hub>=0.28",
# ]
# ///
"""A3 — dense / 우리그래프 / HippoRAG2 3자 비교. 생성 LLM을 키운다.

이전 QA는 Qwen2.5-**7B**였다. 공개 SOTA(HippoRAG2 62.6 EM 등)는 70B급을 쓴다.
생성 모델 크기가 우리 EM을 눌러온 게 맞는지 확인하려고 **72B**로 올린다.

모델은 Qwen2.5-72B-Instruct를 쓴다. Llama-3.3-70B(HippoRAG 논문과 동일)는 gated이고
접근 권한이 없다. 그리고 Qwen 계열을 유지하면 **7B 실행과 계열·토크나이저·채팅
템플릿이 같아** 바뀐 변수가 크기 하나뿐이다. 논문 대조보다 이쪽이 해석이 깨끗하다.

세 검색 방식은 같은 프롬프트·같은 top-k·같은 seed로 채점한다. job_qa.py의
프롬프트와 EM/F1 정의를 **글자 그대로** 옮겼다. 다르면 이전 수치와 이어지지 않는다.

세 데이터셋을 한 잡에서 돈다. 72B 로드가 비싸서 세 번 하면 그만큼 버린다.

인자: N_QUESTIONS TOP_K
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
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
N_SEEDS = 5
MAXK = 20
TP = 4

N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 500
TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SEED = 0

PARQUET_REV = "refs/convert/parquet"
DATASETS = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor/validation/0000.parquet"),
    "2wiki":    ("framolfese/2WikiMultihopQA", "default/validation/0000.parquet"),
    "musique":  ("dgslibisey/MuSiQue", "default/validation/0000.parquet"),
}


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
        questions.append({"q": row["question"], "gold": gold,
                          "answer": str(row["answer"])})
    titles = list(pool)
    return titles, [pool[t] for t in titles], questions


def build_mention_edges(titles, texts, max_title_words=6):
    norm = lambda s: re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    lookup = {}
    for i, t in enumerate(titles):
        key = " ".join(norm(t).split())
        if len(key) >= 5:
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


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
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


def load_hipporag(dataset, n_questions):
    """앞선 잡이 올려둔 HippoRAG2 순위를 가져온다. 없으면 그 조건만 건너뛴다."""
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(RESULTS_REPO, f"runs/hipporag_{dataset}_n{n_questions}.json",
                            repo_type="dataset")
        return json.loads(Path(p).read_text())
    except Exception as e:
        print(f"  ! HippoRAG 순위 없음 ({type(e).__name__}) — 2자 비교로 진행")
        return None


def main():
    import numpy as np
    t0 = time.time()
    build_core()
    from ngdb import DenseIndex, Graph

    # ── 임베딩을 먼저 전부 끝낸다. vLLM이 GPU를 잡으면 bge를 올릴 자리가 없다 ──
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

    prepared = {}
    for ds in DATASETS:
        titles, texts, questions = load_pool(ds, N_Q, SEED)
        edges = build_mention_edges(titles, texts)
        P = encode([f"{t}. {x}" for t, x in zip(titles, texts)])
        dense = DenseIndex(P.shape[1]); dense.add_batch(P)
        Q = encode([QUERY_PREFIX + q["q"] for q in questions])
        hits = [dense.search(Q[i], MAXK) for i in range(len(questions))]
        g = Graph(len(titles)); g.add_edges(edges)

        rank = {}
        rank["dense"] = [[titles[d] for d, _ in h] for h in hits]
        graph_rank = []
        for i in range(len(questions)):
            seed_hits = hits[i][:N_SEEDS]
            acts = g.spread([d for d, _ in seed_hits], [float(s) for _, s in seed_hits],
                            3, 0.65, 0.02, MAXK)
            seen, out = set(), []
            for d, _ in acts:
                if d not in seen:
                    seen.add(d); out.append(d)
            for d, _ in hits[i]:
                if d not in seen:
                    seen.add(d); out.append(d)
            graph_rank.append([titles[d] for d in out])
        rank["graph"] = graph_rank

        hp = load_hipporag(ds, N_Q)
        if hp:
            # 같은 풀·같은 질문 순서인지 확인한다. 어긋나면 비교가 아니다.
            assert hp["pool_size"] == len(titles), \
                f"{ds}: 풀 크기 불일치 {hp['pool_size']} vs {len(titles)}"
            assert hp["questions"] == [q["q"] for q in questions], \
                f"{ds}: 질문 순서 불일치 — 비교 불가"
            rank["hipporag"] = hp["ranked_titles"]

        prepared[ds] = {"titles": titles, "texts": texts, "questions": questions,
                        "rank": rank, "pool": len(titles), "edges": len(edges)}
        print(f"[{time.time()-t0:6.1f}s] {ds}: 질문 {len(questions)} 풀 {len(titles)} "
              f"엣지 {len(edges):,} 조건 {list(rank)}")

    del emodel
    torch.cuda.empty_cache()

    from vllm import LLM, SamplingParams
    llm = LLM(model=LLM_MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=8192, tensor_parallel_size=TP, seed=SEED)
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    print(f"[{time.time()-t0:6.1f}s] LLM 로드 {LLM_MODEL} (TP={TP})")

    report = {}
    for ds, D in prepared.items():
        titles, texts = D["titles"], D["texts"]
        body = {t: x for t, x in zip(titles, texts)}
        questions = D["questions"]

        def make_prompt(q, ctx_titles):
            ctx = "\n\n".join(f"[{n+1}] {t}: {body[t]}" for n, t in enumerate(ctx_titles))
            msg = ("Answer the question using ONLY the passages below.\n"
                   "Reply with the short answer span only — no explanation, no sentence.\n\n"
                   f"{ctx}\n\nQuestion: {q}\nAnswer:")
            return tok.apply_chat_template([{"role": "user", "content": msg}],
                                           tokenize=False, add_generation_prompt=True)

        res, per_q = {}, {}
        for name, ranked in D["rank"].items():
            prompts = [make_prompt(q["q"], ranked[i][:TOP_K])
                       for i, q in enumerate(questions)]
            outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=48))
            preds = [o.outputs[0].text.strip().split("\n")[0] for o in outs]
            ems = [em(p, q["answer"]) for p, q in zip(preds, questions)]
            f1s = [f1(p, q["answer"]) for p, q in zip(preds, questions)]
            res[name] = {"EM": float(np.mean(ems)), "F1": float(np.mean(f1s))}
            per_q[name] = ems
            # 검색 품질도 같이 낸다. QA와 함께 봐야 원인이 보인다
            res[name]["근거전부@k"] = float(np.mean([
                float(all(g in ranked[i][:TOP_K] for g in q["gold"]))
                for i, q in enumerate(questions)]))
            print(f"[{time.time()-t0:6.1f}s] {ds}/{name}: EM {res[name]['EM']:.3f} "
                  f"F1 {res[name]['F1']:.3f} 근거전부@{TOP_K} {res[name]['근거전부@k']:.3f}")

        tests = {}
        names = list(res)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                tests[f"{a} vs {b}"] = mcnemar(list(zip(
                    [bool(x) for x in per_q[a]], [bool(x) for x in per_q[b]])))
        report[ds] = {"results": res, "mcnemar": tests,
                      "pool": D["pool"], "edges": D["edges"], "n": len(questions)}

    print(f"\n{'='*80}\n3자 비교 — {LLM_MODEL}, top-{TOP_K}, n={N_Q}\n{'='*80}")
    for ds, R in report.items():
        print(f"\n{ds}  (풀 {R['pool']}, 질문 {R['n']})")
        print(f"  {'조건':<12}{'EM':>8}{'F1':>8}{'근거전부@'+str(TOP_K):>14}")
        for name, v in R["results"].items():
            print(f"  {name:<12}{v['EM']:>8.3f}{v['F1']:>8.3f}{v['근거전부@k']:>14.3f}")
        for k, mc in R["mcnemar"].items():
            print(f"    {k:<26} 이득 {mc['gain']:>3} 손실 {mc['loss']:>3}  p={mc['p_value']:.4f}")

    payload = {"llm": LLM_MODEL, "embed_model": EMBED_MODEL, "top_k": TOP_K,
               "n_questions": N_Q, "seed": SEED, "report": report,
               "runtime_sec": time.time() - t0}
    out = Path(f"/tmp/compare3_{LLM_MODEL.split('/')[-1]}_top{TOP_K}_n{N_Q}.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True, private=True)
    api.upload_file(path_or_fileobj=str(out), path_in_repo=f"runs/{out.name}",
                    repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"\n업로드: runs/{out.name}")


if __name__ == "__main__":
    main()
