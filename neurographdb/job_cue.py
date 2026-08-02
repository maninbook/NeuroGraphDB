# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pybind11", "numpy", "pyarrow",
#     "transformers>=4.40", "torch", "huggingface_hub>=0.28",
# ]
# ///
"""사전 측정 — 단서 갱신(생각의 흐름)으로 회수를 늘릴 수 있는가.

측면 억제·재순위 계열은 상한이 1~5%p로 소진됐다(job_redundancy.py).
남은 병목은 **회수**다 — MuSiQue 질문의 61.2%는 상위 20에 정답이 다 안 들어온다.

김기인 교정: 시각계의 측면 억제 말고 **논리 회로·생각의 흐름**을 봐야 한다.

형식은 SAM(Raaijmakers & Shiffrin 1981, 자유회상의 고전 모델):
    단서 = 맥락 + 방금 회상해낸 항목
    표집확률 ∝ 단서와의 연합강도
    회상에 성공하면 그 항목이 단서에 합류해 다음 표집이 달라진다

우리 인출은 한 방이다. 질문을 한 번 인코딩하고 한 번 퍼뜨리고 끝난다.
"X 감독의 어머니는?"에서 어머니 문서는 X의 제목을 담지 않고 질문은 어머니 이름을
담지 않으므로 **어느 경로로도 못 간다.** 감독 문단을 먼저 찾으면 그 이름이 단서가 된다.

**상한을 먼저 잰다.** 실패 질문에서, 이미 찾은 정답 문단을 단서로 쓰면
못 찾던 정답 문단이 잡히는가. 오라클 상한이 낮으면 어떤 반복 인출도 소용없다.

  단서 형태 (전부 LLM 없이 만들 수 있는 것)
    q          현재 방식 (질문만)
    p_found    찾은 문단을 그대로 단서로 (오라클 상한)
    q + p      질문과 문단을 합친 단서 (Rocchio식, alpha=0.5 / 1.0)
    q + top1   **오라클이 아닌 실제 조건** — 우리 1위 결과를 단서로

마지막 줄이 핵심이다. 오라클만 높고 실제 조건이 낮으면 구현해도 못 쓴다.

LLM 불필요.
"""

import os
import re
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

CODE_REPO = "goethe0101/neurographdb"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
N_SEEDS = 5
MAXK = 20
MAXW = 6

N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
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


def main():
    import numpy as np
    t0 = time.time()
    build_core()
    from ngdb import DenseIndex, Graph

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

    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    for ds in DATASETS:
        titles, texts, questions = load_pool(ds, N_Q, SEED)
        tidx = {t: i for i, t in enumerate(titles)}
        P = encode([f"{t}. {x}" for t, x in zip(titles, texts)])
        dense = DenseIndex(P.shape[1]); dense.add_batch(P)
        Q = encode([QUERY_PREFIX + q["q"] for q in questions])
        # 문단을 질의처럼 쓰려면 bge 질의 접두어를 붙여야 한다
        Pq = encode([QUERY_PREFIX + f"{t}. {x}" for t, x in zip(titles, texts)])
        g = Graph(len(titles)); g.add_edges(build_edges(titles, texts))

        def current_rank(i):
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
            return out[:MAXK]

        stat = {k: 0 for k in ("total", "already", "no_anchor", "cand",
                               "oracle_p", "oracle_q_p05", "oracle_q_p10",
                               "real_top1", "real_q_top1")}
        for i, q in enumerate(questions):
            gold = [tidx[t] for t in q["gold"] if t in tidx]
            if len(gold) < 2:
                continue
            stat["total"] += 1
            R = current_rank(i)
            missing = [gv for gv in gold if gv not in R]
            if not missing:
                stat["already"] += 1
                continue
            found = [gv for gv in gold if gv in R]
            if not found:
                stat["no_anchor"] += 1     # 발판이 없다 — 단서 갱신 자체가 불가능
                continue
            stat["cand"] += 1              # 발판이 있는 질문 = 단서 갱신이 노릴 대상

            anchor = found[0]
            top1 = R[0]
            tests = {
                "oracle_p":      Pq[anchor],
                "oracle_q_p05":  unit(Q[i] + 0.5 * Pq[anchor]),
                "oracle_q_p10":  unit(Q[i] + 1.0 * Pq[anchor]),
                "real_top1":     Pq[top1],
                "real_q_top1":   unit(Q[i] + 1.0 * Pq[top1]),
            }
            for name, vec in tests.items():
                got = {d for d, _ in dense.search(vec.astype(np.float32), MAXK)}
                if all(m in got for m in missing):
                    stat[name] += 1

        c = max(stat["cand"], 1)
        n = max(stat["total"], 1)
        print(f"\n{'='*76}\n{ds}  근거2개 이상 질문 {stat['total']}\n{'='*76}")
        print(f"  이미 전부 회수      {stat['already']:>5} ({stat['already']/n:>5.1%})")
        print(f"  발판이 없음         {stat['no_anchor']:>5} ({stat['no_anchor']/n:>5.1%})"
              f"   ← 단서 갱신 불가")
        print(f"  **단서 갱신 대상**  {stat['cand']:>5} ({stat['cand']/n:>5.1%})"
              f"   ← 여기서만 얻을 수 있다")
        print(f"\n  대상 질문에서 놓친 정답을 되찾은 비율")
        for name, label in (("oracle_p", "오라클: 정답문단 단독"),
                            ("oracle_q_p05", "오라클: 질문+0.5×문단"),
                            ("oracle_q_p10", "오라클: 질문+1.0×문단"),
                            ("real_top1", "실제: 1위결과 단독"),
                            ("real_q_top1", "실제: 질문+1위결과")):
            print(f"    {label:<22} {stat[name]:>4}/{stat['cand']:<4} "
                  f"({stat[name]/c:>5.1%})   전체 대비 +{stat[name]/n:>4.1%}p")
        print(f"\n  → 전체 상한(오라클 최선) = +{max(stat['oracle_p'], stat['oracle_q_p05'], stat['oracle_q_p10'])/n:.1%}p, "
              f"실제 조건 최선 = +{max(stat['real_top1'], stat['real_q_top1'])/n:.1%}p")
    print(f"\n({time.time()-t0:.0f}초)")


if __name__ == "__main__":
    main()
