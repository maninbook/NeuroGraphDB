# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11",
#     "numpy",
#     "datasets>=3.2",
#     "sentence-transformers",
#     "torch",
#     "huggingface_hub>=0.28",
# ]
# ///
"""B0 — HF Jobs에서 baseline 측정. C++ 코어를 잡 안에서 컴파일해 쓴다.

로컬은 디스크가 부족하다. 연산은 전부 HF에서 돈다.
소스는 Hub 데이터셋 repo에서 받아 컨테이너 안에서 빌드한다.

    hf jobs uv run --flavor l4x1 --timeout 2h --secrets HF_TOKEN \
        job_baseline.py 1000

인자: N_QUESTIONS
"""

import json
import os
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

N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0


def build_core() -> Path:
    """C++ 검색 코어를 컨테이너 안에서 빌드한다.

    macOS의 `-undefined dynamic_lookup`은 리눅스에서 안 통한다. 여기서는 순수 -shared.
    """
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(CODE_REPO, repo_type="dataset",
                                  allow_patterns="src/**"))
    src = root / "src"
    out = Path("/tmp/ngdb")
    out.mkdir(exist_ok=True)
    (out / "__init__.py").write_text(
        "from ._ngdb_core import BM25, DenseIndex\n"
        '__all__ = ["BM25", "DenseIndex"]\n')

    inc = subprocess.run([sys.executable, "-m", "pybind11", "--includes"],
                         capture_output=True, text=True, check=True).stdout.split()
    ext = sysconfig.get_config_var("EXT_SUFFIX")
    cxx = os.environ.get("CXX", "c++")
    cmd = [cxx, "-std=c++17", "-O2", "-fPIC", "-shared", *inc, f"-I{src}",
           "-o", str(out / f"_ngdb_core{ext}"),
           str(src / "bm25.cpp"), str(src / "dense.cpp"), str(src / "bindings.cpp")]
    print("빌드:", " ".join(cmd[:6]), "...")
    subprocess.run(cmd, check=True)
    sys.path.insert(0, "/tmp")
    return out


def load_pool(n_questions: int, seed: int):
    from datasets import load_dataset
    import numpy as np

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:n_questions]

    passages: dict[str, str] = {}
    questions = []
    for i in idx:
        row = ds[int(i)]
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
            passages.setdefault(title, " ".join(sents))
        questions.append({"q": row["question"],
                          "gold": sorted(set(row["supporting_facts"]["title"])),
                          "answer": row["answer"]})
    titles = list(passages)
    return titles, [passages[t] for t in titles], questions


def main() -> None:
    import numpy as np

    t0 = time.time()
    build_core()
    from ngdb import BM25, DenseIndex          # noqa: E402
    print(f"[{time.time()-t0:6.1f}s] C++ 코어 빌드 완료")

    titles, texts, questions = load_pool(N_Q, SEED)
    print(f"[{time.time()-t0:6.1f}s] 질문 {len(questions)} | 문단 풀 {len(titles)}")

    bm = BM25()
    for t, x in zip(titles, texts):
        bm.add(f"{t} {x}")
    bm.finalize()
    print(f"[{time.time()-t0:6.1f}s] BM25 색인 {bm.size}")

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=dev)
    dim = model.get_sentence_embedding_dimension()
    P = model.encode([f"{t}. {x}" for t, x in zip(titles, texts)],
                     batch_size=128, convert_to_numpy=True,
                     show_progress_bar=False).astype(np.float32)
    dense = DenseIndex(dim)
    dense.add_batch(P)
    Q = model.encode([QUERY_PREFIX + q["q"] for q in questions],
                     batch_size=128, convert_to_numpy=True,
                     show_progress_bar=False).astype(np.float32)
    print(f"[{time.time()-t0:6.1f}s] 벡터 색인 {dense.size} ({dim}차원, {dev})")

    maxk = max(KS)
    methods: dict[str, list] = {"bm25": [], "dense": [], "hybrid": []}
    per_q = []

    def recall_at(ranked, gold, k):
        return sum(1 for g in gold if g in ranked[:k]) / len(gold) if gold else 0.0

    for i, q in enumerate(questions):
        b_hits = bm.search(q["q"], maxk)
        d_hits = dense.search(Q[i], maxk)
        b_titles = [titles[d] for d, _ in b_hits]
        d_titles = [titles[d] for d, _ in d_hits]
        # RRF — 튜닝 파라미터가 없어 baseline으로 공정하다
        rr: dict[int, float] = {}
        for rank, (doc, _) in enumerate(b_hits):
            rr[doc] = rr.get(doc, 0.0) + 1.0 / (60 + rank + 1)
        for rank, (doc, _) in enumerate(d_hits):
            rr[doc] = rr.get(doc, 0.0) + 1.0 / (60 + rank + 1)
        h_titles = [titles[d] for d, _ in sorted(rr.items(), key=lambda x: -x[1])[:maxk]]

        row = {"gold": q["gold"]}
        for name, ranked in (("bm25", b_titles), ("dense", d_titles), ("hybrid", h_titles)):
            r = {f"r@{k}": recall_at(ranked, q["gold"], k) for k in KS}
            # 멀티홉의 실질 성공 조건 — 근거 2개를 모두 담았는가
            r["all@10"] = float(all(g in ranked[:10] for g in q["gold"]))
            methods[name].append(r)
            row[name] = r
        per_q.append(row)
        if (i + 1) % 200 == 0:
            print(f"  [{time.time()-t0:6.1f}s] 평가 {i+1}/{len(questions)}")

    print(f"\n{'='*72}\nHotpotQA distractor — 질문 {len(questions)} / 풀 {len(titles)}\n{'='*72}")
    print("방법        " + "".join(f"{'R@'+str(k):>9}" for k in KS) + f"{'근거2개@10':>12}")
    summary = {}
    for name, rows in methods.items():
        vals = {f"r@{k}": float(np.mean([r[f"r@{k}"] for r in rows])) for k in KS}
        vals["all@10"] = float(np.mean([r["all@10"] for r in rows]))
        summary[name] = vals
        print(f"{name:<12}" + "".join(f"{vals[f'r@{k}']:>9.3f}" for k in KS)
              + f"{vals['all@10']:>12.3f}")

    payload = {"dataset": "hotpotqa distractor validation", "n_questions": len(questions),
               "pool_size": len(titles), "seed": SEED, "embed_model": EMBED_MODEL,
               "device": dev, "summary": summary, "per_question": per_q,
               "runtime_sec": time.time() - t0}
    out = Path(f"/tmp/baseline_hotpotqa_n{len(questions)}.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True, private=True)
    api.upload_file(path_or_fileobj=str(out), path_in_repo=f"runs/{out.name}",
                    repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"\n업로드: https://huggingface.co/datasets/{RESULTS_REPO}")


if __name__ == "__main__":
    main()
