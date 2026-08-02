# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11", "numpy", "pyarrow",
#     "transformers>=4.40", "torch", "vllm>=0.10",
#     "huggingface_hub>=0.28",
# ]
# ///
"""M — 작은 리더로 큰 리더 효과 내기. 검색은 완전히 고정하고 리더만 바꾼다.

목표(김기인): **4~7B를 추론에 쓰면서 70B급 효과, 그리고 빠르게.**

우리가 잰 격차 — MuSiQue n=500 top-10 동일 검색에서 7B EM 0.190 vs 72B 0.238.

그리고 **우리가 지금 7B를 불리하게 쓰고 있다.** 현재 프롬프트는
"Reply with the short answer span only — no explanation"으로 **추론을 금지한다.**
큰 모델은 멀티홉을 속으로 처리하지만 작은 모델은 밖으로 꺼내야 한다.

조건 (검색·문단·질문·top-k 전부 고정, 리더 쪽만 변경)
    S0  현재 프롬프트                     — 기존 기준선과 같아야 한다
    S1  CoT — 단계를 쓰고 마지막 줄에 답
    S2  자기일관성 — S0 프롬프트로 8회 샘플 후 다수결
    S3  CoT + 자기일관성

7B 8회는 72B 1회보다 **메모리가 10배 작다.** 로컬에서 진짜 제약은 메모리이므로
이쪽이 실질적으로 빠르고 돌릴 수 있다.

72B 기준값은 이미 있다(runs/compare3_Qwen2.5-72B-Instruct_top10_n500.json).
여기서는 **격차를 몇 % 메우는지**를 낸다.

── 사전등록 ────────────────────────────────────────────────────────────────
주 가설  MuSiQue EM에서 S3 > S0. McNemar 대응 이분, α=0.05, n=500.
부 가설  HotpotQA·2Wiki 동일. S1·S2 단독 효과도 보고한다.
지표     격차 메움률 = (S* − S0) / (72B − S0). 음수면 그대로 적는다.
파라미터 샘플 8회, temperature 0.7, top_p 0.95로 **미리 고정.** 사후 조정 금지.
실패 조건 유의하지 않으면 그대로 적고 샘플 수·온도를 바꿔 재시도하지 않는다.
────────────────────────────────────────────────────────────────────────────

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
LLM_MODEL = os.environ.get("READER", "Qwen/Qwen2.5-7B-Instruct")
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
N_SEEDS = 5
MAXK = 20
MAXW = 6
N_SAMPLE = 8          # 사전등록
# 폐쇄형(검색 없음) 통제. RAG가 실제로 얼마를 더하는지 재려면 이게 있어야 한다.
CLOSED = os.environ.get("CLOSEDBOOK", "") == "1"
TEMP = 0.7            # 사전등록
TOP_P = 0.95          # 사전등록

N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 500
TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
ONLY = sys.argv[3] if len(sys.argv) > 3 else None
SEED = 0

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
        edges.extend((i, j, 1.0) for j in hit)
    return edges


def normalize_ans(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(p, g):
    return float(normalize_ans(p) == normalize_ans(g))


def f1(p, g):
    a, b = normalize_ans(p).split(), normalize_ans(g).split()
    if not a or not b:
        return float(a == b)
    c = Counter(a) & Counter(b)
    n = sum(c.values())
    if n == 0:
        return 0.0
    pr, rc = n / len(a), n / len(b)
    return 2 * pr * rc / (pr + rc)


def mcnemar(pairs):
    from math import comb
    b = sum(1 for x, y in pairs if not x and y)
    c = sum(1 for x, y in pairs if x and not y)
    n = b + c
    if n == 0:
        return {"gain": 0, "loss": 0, "n_discordant": 0, "p_value": 1.0}
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return {"gain": b, "loss": c, "n_discordant": n, "p_value": min(1.0, 2 * tail)}


def main():
    import numpy as np
    import torch
    t0 = time.time()
    build_core()
    from ngdb import DenseIndex, Graph

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
    for ds in ([ONLY] if ONLY else list(DATASETS)):
        titles, texts, questions = load_pool(ds, N_Q, SEED)
        P = encode([f"{t}. {x}" for t, x in zip(titles, texts)])
        dense = DenseIndex(P.shape[1]); dense.add_batch(P)
        Q = encode([QUERY_PREFIX + q["q"] for q in questions])
        g = Graph(len(titles)); g.add_edges(build_edges(titles, texts))
        ranks = []
        for i in range(len(questions)):
            hits = dense.search(Q[i], MAXK)
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
            ranks.append([titles[d] for d in out[:MAXK]])
        prepared[ds] = {"titles": titles, "body": dict(zip(titles, texts)),
                        "questions": questions, "ranks": ranks}
        print(f"[{time.time()-t0:6.1f}s] {ds}: 질문 {len(questions)} 풀 {len(titles)}")

    del emodel
    torch.cuda.empty_cache()

    from vllm import LLM, SamplingParams
    llm = LLM(model=LLM_MODEL, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=8192, enforce_eager=True, seed=SEED)
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    print(f"[{time.time()-t0:6.1f}s] LLM 로드 {LLM_MODEL}")

    PLAIN = ("Answer the question using ONLY the passages below.\n"
             "Reply with the short answer span only — no explanation, no sentence.\n\n")
    COT = ("Answer the question using ONLY the passages below.\n"
           "Work through the passages step by step. Multi-hop questions need you to find one "
           "fact first, then use it to find the next.\n"
           "End your reply with a final line of exactly this form:\n"
           "ANSWER: <the short answer span>\n\n")

    PLAIN_CB = ("Answer the question from your own knowledge.\n"
                "Reply with the short answer span only — no explanation, no sentence.\n\n")
    COT_CB = ("Answer the question from your own knowledge.\n"
              "Work through it step by step. Multi-hop questions need you to recall one fact "
              "first, then use it to recall the next.\n"
              "End your reply with a final line of exactly this form:\n"
              "ANSWER: <the short answer span>\n\n")

    def extract(text, cot):
        t = text.strip()
        if not cot:
            return t.split("\n")[0].strip()
        m = re.findall(r"ANSWER:\s*(.+)", t)
        if m:
            return m[-1].strip()
        return t.split("\n")[-1].strip()      # 형식을 안 지키면 마지막 줄

    report = {}
    for ds, D in prepared.items():
        body, questions, ranks = D["body"], D["questions"], D["ranks"]

        def prompts(cot):
            out = []
            for i, q in enumerate(questions):
                if CLOSED:
                    head = (COT_CB if cot else PLAIN_CB)
                    msg = f"{head}Question: {q['q']}\nAnswer:"
                else:
                    ctx = "\n\n".join(f"[{n+1}] {t}: {body[t]}"
                                      for n, t in enumerate(ranks[i][:TOP_K]))
                    head = COT if cot else PLAIN
                    msg = f"{head}{ctx}\n\nQuestion: {q['q']}\nAnswer:"
                out.append(tok.apply_chat_template([{"role": "user", "content": msg}],
                                                   tokenize=False, add_generation_prompt=True))
            return out

        def run(cot, n_sample):
            ps = prompts(cot)
            sp = (SamplingParams(temperature=0.0, max_tokens=384 if cot else 48)
                  if n_sample == 1 else
                  SamplingParams(temperature=TEMP, top_p=TOP_P, n=n_sample,
                                 max_tokens=384 if cot else 48, seed=SEED))
            outs = llm.generate(ps, sp)
            preds = []
            for o in outs:
                cands = [extract(c.text, cot) for c in o.outputs]
                cands = [c for c in cands if normalize_ans(c)]
                if not cands:
                    preds.append("")
                elif n_sample == 1:
                    preds.append(cands[0])
                else:
                    # 정규화 형태로 다수결, 대표 표기는 최빈 원문
                    key = Counter(normalize_ans(c) for c in cands).most_common(1)[0][0]
                    preds.append(next(c for c in cands if normalize_ans(c) == key))
            return preds

        conds = {"S0_현재": (False, 1), "S1_CoT": (True, 1),
                 "S2_자기일관성": (False, N_SAMPLE), "S3_CoT+자기일관성": (True, N_SAMPLE)}
        res, per_em = {}, {}
        for name, (cot, ns) in conds.items():
            preds = run(cot, ns)
            ems = [em(p, q["answer"]) for p, q in zip(preds, questions)]
            res[name] = {"EM": float(np.mean(ems)),
                         "F1": float(np.mean([f1(p, q["answer"])
                                              for p, q in zip(preds, questions)]))}
            per_em[name] = ems
            print(f"[{time.time()-t0:6.1f}s] {ds}/{name}: EM {res[name]['EM']:.3f} "
                  f"F1 {res[name]['F1']:.3f}")
        report[ds] = {"results": res, "per_em": per_em}

    ref = {"hotpotqa": 0.612, "2wiki": 0.532, "musique": 0.238}   # 72B graph top-10 n=500
    print(f"\n{'='*80}\n작은 리더 강화 — {LLM_MODEL}, top-{TOP_K}, n={N_Q}\n{'='*80}")
    out_tests = {}
    for ds, R in report.items():
        s0 = R["results"]["S0_현재"]["EM"]
        gap = ref[ds] - s0
        print(f"\n{ds}   72B 기준 {ref[ds]:.3f} · 7B 현재 {s0:.3f} · 격차 {gap:+.3f}")
        print(f"  {'조건':<18}{'EM':>8}{'F1':>8}{'격차 메움':>11}{'McNemar vs S0':>22}")
        tests = {}
        for name, v in R["results"].items():
            if name == "S0_현재":
                print(f"  {name:<18}{v['EM']:>8.3f}{v['F1']:>8.3f}{'—':>11}{'—':>22}")
                continue
            mc = mcnemar(list(zip([bool(x) for x in R["per_em"]["S0_현재"]],
                                  [bool(x) for x in R["per_em"][name]])))
            tests[name] = mc
            close = (v["EM"] - s0) / gap if abs(gap) > 1e-9 else float("nan")
            verdict = f"{mc['gain']}/{mc['loss']} p={mc['p_value']:.4f}"
            print(f"  {name:<18}{v['EM']:>8.3f}{v['F1']:>8.3f}{close:>10.0%}{verdict:>22}")
        out_tests[ds] = tests

    payload = {"llm": LLM_MODEL, "reference_72b": ref, "top_k": TOP_K, "n": N_Q,
               "n_sample": N_SAMPLE, "temperature": TEMP,
               "results": {k: v["results"] for k, v in report.items()},
               "mcnemar": out_tests, "runtime_sec": time.time() - t0}
    tag = LLM_MODEL.split("/")[-1] + ("_closedbook" if CLOSED else "")
    out = Path(f"/tmp/smallreader_{tag}_top{TOP_K}_n{N_Q}.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=str(out), path_in_repo=f"runs/{out.name}",
                        repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"\n업로드: runs/{out.name}")


if __name__ == "__main__":
    main()
