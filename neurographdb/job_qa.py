# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11", "numpy", "pyarrow",
#     # `datasets`를 쓰지 않는다. 5.x가 torchcodec을 끌어오는데 그게 FFmpeg 공유
#     # 라이브러리(libavutil.so.56)를 찾다 실패한다 — 컨테이너에 없다.
#     # 우리는 오디오·비디오가 필요 없으므로 parquet를 직접 읽는다.
#     # sentence-transformers는 쓰지 않는다. import 시점에 멀티모달용 torchcodec을
#     # 끌어오는데(AudioDecoder/VideoDecoder) 컨테이너에 FFmpeg가 없어 죽는다.
#     # bge는 CLS 풀링 + L2 정규화이므로 transformers로 직접 계산한다
#     # (로컬에서 sentence-transformers 출력과 최대 절대차 0.000000 확인).
#     "transformers>=4.40", "torch",
#     # vllm에 하한을 걸지 않으면 sentence-transformers와의 해석 과정에서
#     # 0.2.5(2023년)로 후퇴한다. 그 버전은 휠이 없어 소스 빌드로 가고
#     # 컨테이너에 CUDA_HOME이 없어 실패한다.
#     "vllm>=0.10",
#     "huggingface_hub>=0.28",
# ]
# ///
"""Q1 — 종단 QA 정확도. 남들과 **같은 단위**로 숫자를 낸다.

지금까지 우리 숫자는 검색 재현율이고, HippoRAG2·LogicRAG의 공개 수치는 답변 정확도다.
단위가 달라 비교가 안 된다. 인정받으려면 같은 단위로 놓아야 한다.

핵심 설계 — **LLM과 프롬프트와 k를 완전히 고정하고 검색만 바꾼다.**
그래야 차이가 검색에서 왔다고 말할 수 있다. 모델을 잡 안에서 vLLM으로 서빙하는 이유도
같다. 외부 API는 언제 모델이 바뀔지 몰라 재현이 안 된다.

지표
  EM   정규화 후 완전일치 (SQuAD 관례: 소문자·관사·구두점 제거)
  F1   토큰 F1 (부분 정답 인정)
  둘 다 HippoRAG 계열이 보고하는 지표다

판정: 같은 문제를 두 검색이 푸는 대응 설계 → McNemar (EM 기준)

인자: DATASET(hotpotqa|2wiki|musique) N_QUESTIONS TOP_K SEED
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

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # 컨테이너에 nvcc가 없다

CODE_REPO = "goethe0101/neurographdb"
RESULTS_REPO = "goethe0101/neurographdb-results"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_SEEDS = 5          # G1에서 사전 지정한 값

DATASET = sys.argv[1] if len(sys.argv) > 1 else "hotpotqa"
N_Q = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
TOP_K = int(sys.argv[3]) if len(sys.argv) > 3 else 5   # HippoRAG 계열 관례
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0

# HF가 자동 변환해 두는 parquet 브랜치를 직접 읽는다. 로딩 스크립트도 datasets도 안 탄다.
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


# ── 채점 (SQuAD 관례) ────────────────────────────────────────────────────────

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


def main():
    import numpy as np
    t0 = time.time()
    build_core()
    from ngdb import DenseIndex, Graph

    titles, texts, questions = load_pool(DATASET, N_Q, SEED)
    print(f"[{time.time()-t0:6.1f}s] {DATASET}: 질문 {len(questions)} | 풀 {len(titles)}")

    g = Graph(len(titles))
    g.add_edges(build_mention_edges(titles, texts))
    print(f"[{time.time()-t0:6.1f}s] 엣지 {g.n_edges:,}")

    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    etok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    emodel = AutoModel.from_pretrained(EMBED_MODEL).to(dev).eval()

    def encode(items, batch=128):
        """bge 계열은 CLS 풀링 + L2 정규화. sentence-transformers와 수치가 같다."""
        out = []
        for a in range(0, len(items), batch):
            b = etok(items[a:a + batch], padding=True, truncation=True,
                     max_length=512, return_tensors="pt").to(dev)
            with torch.no_grad():
                h = emodel(**b).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    P = encode([f"{t}. {x}" for t, x in zip(titles, texts)])
    dim = P.shape[1]
    dense = DenseIndex(dim); dense.add_batch(P)
    Q = encode([QUERY_PREFIX + q["q"] for q in questions])
    del emodel, P
    torch.cuda.empty_cache()
    print(f"[{time.time()-t0:6.1f}s] 벡터 색인 {dense.size} ({dim}차원, {dev})")

    # ── 두 검색 방식 ────────────────────────────────────────────────────────
    dense_hits = [dense.search(Q[i], 20) for i in range(len(questions))]

    def rank_dense(i):
        return [d for d, _ in dense_hits[i]]

    def rank_graph(i):
        hits = dense_hits[i][:N_SEEDS]
        acts = g.spread([d for d, _ in hits], [float(s) for _, s in hits],
                        3, 0.65, 0.02, 20)
        seen, out = set(), []
        for d, _ in acts:
            if d not in seen:
                seen.add(d); out.append(d)
        for d, _ in dense_hits[i]:
            if d not in seen:
                seen.add(d); out.append(d)
        return out

    # ── LLM은 잡 안에서 서빙한다. 외부 API는 모델이 바뀌면 재현이 깨진다 ────
    from vllm import LLM, SamplingParams
    llm = LLM(model=LLM_MODEL, dtype="float16", gpu_memory_utilization=0.80,
              max_model_len=8192, enforce_eager=True, seed=SEED)
    tok = llm.get_tokenizer()
    print(f"[{time.time()-t0:6.1f}s] LLM 로드 {LLM_MODEL}")

    def make_prompt(q, ctx_ids):
        ctx = "\n\n".join(f"[{n+1}] {titles[d]}: {texts[d]}" for n, d in enumerate(ctx_ids))
        msg = ("Answer the question using ONLY the passages below.\n"
               "Reply with the short answer span only — no explanation, no sentence.\n\n"
               f"{ctx}\n\nQuestion: {q}\nAnswer:")
        return tok.apply_chat_template([{"role": "user", "content": msg}],
                                       tokenize=False, add_generation_prompt=True)

    results = {}
    per_q = {}
    for name, rank_fn in (("dense", rank_dense), ("graph", rank_graph)):
        prompts = [make_prompt(q["q"], rank_fn(i)[:TOP_K])
                   for i, q in enumerate(questions)]
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=48))
        preds = [o.outputs[0].text.strip().split("\n")[0] for o in outs]
        ems = [em(p, q["answer"]) for p, q in zip(preds, questions)]
        f1s = [f1(p, q["answer"]) for p, q in zip(preds, questions)]
        results[name] = {"EM": float(np.mean(ems)), "F1": float(np.mean(f1s))}
        per_q[name] = {"preds": preds, "em": ems}
        print(f"[{time.time()-t0:6.1f}s] {name}: EM {results[name]['EM']:.3f} "
              f"F1 {results[name]['F1']:.3f}")

    mc = mcnemar(list(zip([bool(x) for x in per_q["dense"]["em"]],
                          [bool(x) for x in per_q["graph"]["em"]])))
    verdict = ("graph가 dense를 넘음" if mc["gain"] > mc["loss"] and mc["p_value"] < 0.05
               else "graph가 더 나쁨" if mc["loss"] > mc["gain"] and mc["p_value"] < 0.05
               else f"판정 불가 — 불일치 {mc['n_discordant']}" if mc["n_discordant"] < 10
               else "유의하지 않음")

    print(f"\n{'='*70}\n{DATASET} 종단 QA — {LLM_MODEL}, top-{TOP_K}, n={len(questions)}\n{'='*70}")
    print(f"{'검색':<10}{'EM':>9}{'F1':>9}")
    for name in ("dense", "graph"):
        print(f"{name:<10}{results[name]['EM']:>9.3f}{results[name]['F1']:>9.3f}")
    print(f"\nMcNemar(EM): 이득 {mc['gain']} 손실 {mc['loss']} "
          f"(불일치 {mc['n_discordant']}, p={mc['p_value']:.4f}) → {verdict}")

    payload = {"dataset": DATASET, "n_questions": len(questions), "top_k": TOP_K,
               "llm": LLM_MODEL, "embed_model": EMBED_MODEL, "n_seeds": N_SEEDS,
               "seed": SEED, "results": results, "mcnemar_em": mc, "verdict": verdict,
               "runtime_sec": time.time() - t0,
               "per_question": [{"q": q["q"], "gold": q["answer"],
                                 "dense": per_q["dense"]["preds"][i],
                                 "graph": per_q["graph"]["preds"][i]}
                                for i, q in enumerate(questions)]}
    out = Path(f"/tmp/qa_{DATASET}_n{len(questions)}.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True, private=True)
    api.upload_file(path_or_fileobj=str(out), path_in_repo=f"runs/{out.name}",
                    repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"\n업로드: https://huggingface.co/datasets/{RESULTS_REPO}")


if __name__ == "__main__":
    main()
