# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "hipporag==2.0.0a4",
#     "pyarrow",
#     "huggingface_hub>=0.28",
# ]
#
# [tool.uv]
# override-dependencies = ["openai==1.91.0"]
# ///
# NOTE: hipporag 2.0.0a4는 `openai==1.91.1`을 핀으로 거는데 그 버전이 PyPI에 없다
#       (1.91.0과 1.92.0은 존재). 공개된 그대로는 설치가 안 된다.
#       비교의 공정성을 위해 **그 핀 하나만** 최소로 덮어쓴다. 나머지 핀은 손대지 않는다.
"""A — HippoRAG 정면 비교. 그쪽 **공개 구현**을 우리 조건에서 직접 돌린다.

지금까지 우리는 공개 논문 수치와 나란히 놓기만 했다. 그건 비교가 아니다.
LLM도 프롬프트도 지표 정의도 다르기 때문이다. 여기서 그걸 없앤다.

고정하는 것 — 문단 풀, 질문과 그 순서, 임베딩 모델, LLM, 검색 개수.
    임베더는 `Transformers/BAAI/bge-base-en-v1.5`로 강제한다.
    (기본값 NV-Embed-v2는 7B짜리라 그대로 두면 **임베더 차이**를 그래프 차이로 착각한다)
    LLM은 우리 QA와 같은 Qwen2.5-7B-Instruct를 잡 안에서 서빙해 붙인다.

바뀌는 것 — 검색 방식 하나뿐이다.

이 스크립트는 **순위만 뽑아 올린다.** QA는 별도 잡에서 세 순위(dense / 우리 / HippoRAG)를
같은 프롬프트에 넣어 한 번에 돌린다. LLM을 한 군데로 몰아야 조건이 흔들리지 않는다.

인자: DATASET N_QUESTIONS SEED
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RESULTS_REPO = "goethe0101/neurographdb-results"
LLM = "Qwen/Qwen2.5-7B-Instruct"
EMBED = "BAAI/bge-base-en-v1.5"               # 우리와 동일한 임베더로 강제
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAXK = 20

DATASET = sys.argv[1] if len(sys.argv) > 1 else "hotpotqa"
N_Q = int(sys.argv[2]) if len(sys.argv) > 2 else 500
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0

PARQUET_REV = "refs/convert/parquet"
DATASETS = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor/validation/0000.parquet"),
    "2wiki":    ("framolfese/2WikiMultihopQA", "default/validation/0000.parquet"),
    "musique":  ("dgslibisey/MuSiQue", "default/validation/0000.parquet"),
}
HR_DATASET = {"hotpotqa": "hotpotqa", "2wiki": "2wikimultihopqa", "musique": "musique"}


def load_pool(dataset, n_questions, seed):
    """job_qa.py / job_hebbian.py와 **글자 단위로 같은** 풀을 만든다.

    여기가 틀어지면 비교가 아니라 다른 실험이 된다. 복사해 두고 손대지 않는다.
    """
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


def serve_vllm(model, port=8000, timeout=1800):
    """잡 안에서 OpenAI 호환 서버를 띄운다. OpenIE와 재순위가 같은 경로를 쓰게 된다."""
    env = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0")
    p = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", model, "--port", str(port),
         "--max-model-len", "8192", "--gpu-memory-utilization", "0.80",
         "--disable-log-requests"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if p.poll() is not None:
            raise RuntimeError(f"vLLM 서버가 죽었다 (exit {p.returncode})")
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as r:
                if r.status == 200:
                    print(f"[vLLM] 준비됨 ({time.time()-t0:.0f}초)")
                    return p
        except Exception:
            time.sleep(5)
    raise RuntimeError("vLLM 서버 기동 시간 초과")


def install_bge_embedder():
    """우리와 **같은** bge 인코딩을 HippoRAG의 임베딩 자리에 꽂는다.

    릴리스된 2.0.0a4는 임베딩 모델 주입을 받지 않고, 팩토리가 아는 이름도
    GritLM/NV-Embed-v2/contriever/OpenAI/Cohere뿐이다(기본값은 7B짜리 NV-Embed-v2).
    기본값을 그대로 두면 **임베더 차이(110M vs 7B)를 그래프 차이로 착각**하게 된다.

    그래서 그들이 열어둔 `BaseEmbeddingModel` 인터페이스에 우리 인코딩을 구현해 넣는다.
    OpenIE·PPR·사실 재순위 등 **알고리즘은 한 줄도 건드리지 않는다.**
    """
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    import hipporag  # noqa: F401  (패키지를 먼저 로드해야 하위 모듈이 등록된다)
    from hipporag.embedding_model import BaseEmbeddingModel, EmbeddingConfig

    # `hipporag/__init__.py`가 이름 `HippoRAG`를 **클래스**로 덮어쓴다.
    # 그래서 `import hipporag.HippoRAG as m`은 모듈이 아니라 클래스를 준다.
    # 모듈 전역을 갈아끼워야 하므로 sys.modules에서 직접 집는다.
    hr_mod = sys.modules["hipporag.HippoRAG"]

    class BGEEmbeddingModel(BaseEmbeddingModel):
        def __init__(self, global_config=None, embedding_model_name=None):
            super().__init__(global_config=global_config)
            self.embedding_model_name = embedding_model_name or EMBED
            self.embedding_config = EmbeddingConfig()
            self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.tok = AutoTokenizer.from_pretrained(EMBED)
            self.model = AutoModel.from_pretrained(EMBED).to(self.dev).eval()
            self.embedding_dim = self.model.config.hidden_size

        def batch_encode(self, texts, **kw):
            if isinstance(texts, str):
                texts = [texts]
            # instruction이 있으면 질의다. bge는 질의에 접두어를 붙여야 성능이 난다.
            # 우리 실험에서 쓴 접두어와 **같은 문자열**을 쓴다.
            if kw.get("instruction"):
                texts = [QUERY_PREFIX + t for t in texts]
            bs = getattr(self.global_config, "embedding_batch_size", 128) or 128
            out = []
            for a in range(0, len(texts), bs):
                b = self.tok(texts[a:a + bs], padding=True, truncation=True,
                             max_length=512, return_tensors="pt").to(self.dev)
                with torch.no_grad():
                    h = self.model(**b).last_hidden_state[:, 0]   # CLS 풀링
                out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
            return np.concatenate(out).astype(np.float32)

    hr_mod._get_embedding_model_class = lambda embedding_model_name=None: BGEEmbeddingModel
    # 패치가 먹었는지 **여기서** 확인한다. 조용히 실패하면 기본 NV-Embed-v2가 쓰여
    # 비교가 통째로 무효가 되는데, 결과만 봐서는 눈치채기 어렵다.
    got = hr_mod._get_embedding_model_class(embedding_model_name=EMBED)
    assert got is BGEEmbeddingModel, f"임베더 패치 실패: {got}"
    print(f"[임베더] {EMBED} 어댑터 설치 확인 — 우리 실험과 동일한 인코딩")


def main():
    import numpy as np
    t0 = time.time()

    titles, texts, questions = load_pool(DATASET, N_Q, SEED)
    docs = [f"{t}. {x}" for t, x in zip(titles, texts)]
    doc2title = {d: t for d, t in zip(docs, titles)}
    print(f"[{time.time()-t0:6.1f}s] {DATASET}: 질문 {len(questions)} | 풀 {len(titles)}")

    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

    # 임베더 패치를 vLLM 기동(약 2분) **앞에서** 검증한다.
    # 뒤에 두면 실패할 때마다 서버 기동 시간을 그냥 버린다.
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig
    install_bge_embedder()

    server = serve_vllm(LLM)

    try:
        cfg = BaseConfig(
            llm_name=LLM,
            llm_base_url="http://localhost:8000/v1",
            embedding_model_name=EMBED,
            embedding_batch_size=128,
            retrieval_top_k=MAXK,
            openie_mode="online",
            dataset=HR_DATASET[DATASET],
            save_dir=f"/tmp/hipporag_{DATASET}",
            force_index_from_scratch=True,
            force_openie_from_scratch=True,
            save_openie=True,
        )
        hr = HippoRAG(global_config=cfg)

        print(f"[{time.time()-t0:6.1f}s] 색인 시작 — OpenIE {len(docs)}문단")
        hr.index(docs=docs)
        print(f"[{time.time()-t0:6.1f}s] 색인 완료")

        sols = hr.retrieve(queries=[q["q"] for q in questions], num_to_retrieve=MAXK)
        print(f"[{time.time()-t0:6.1f}s] 검색 완료 {len(sols)}건")
    finally:
        server.terminate()

    # ── 반환된 문서를 제목으로 되돌린다 ──────────────────────────────────────
    unmatched = 0
    ranked_all = []
    for sol in sols:
        r = []
        for d in sol.docs[:MAXK]:
            t = doc2title.get(d)
            if t is None:                       # 공백 정규화 등으로 어긋난 경우
                t = doc2title.get(d.strip())
            if t is None:
                unmatched += 1
                t = d.split(".")[0].strip()     # 마지막 수단
            r.append(t)
        ranked_all.append(r)
    if unmatched:
        print(f"  ! 제목 역매핑 실패 {unmatched}건 — 접두어로 대체함")

    KS = (1, 2, 5, 10, 20)
    rows = []
    for q, ranked in zip(questions, ranked_all):
        gold = q["gold"]
        r = {f"r@{k}": sum(1 for g in gold if g in ranked[:k]) / len(gold) for k in KS}
        r["all@10"] = float(all(g in ranked[:10] for g in gold))
        rows.append(r)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in list(rows[0])}

    print(f"\n{'='*72}\nHippoRAG2 {DATASET} — 질문 {len(questions)} / 풀 {len(titles)}\n{'='*72}")
    print("".join(f"{'R@'+str(k):>9}" for k in KS) + f"{'근거2개@10':>12}")
    print("".join(f"{agg[f'r@{k}']:>9.3f}" for k in KS) + f"{agg['all@10']:>12.3f}")
    print(f"\n임베더 {EMBED} | LLM {LLM} — 우리 실험과 동일")

    payload = {"method": "hipporag2", "dataset": DATASET, "n_questions": len(questions),
               "pool_size": len(titles), "seed": SEED, "embed_model": EMBED, "llm": LLM,
               "summary": agg, "ranked_titles": ranked_all,
               "gold": [q["gold"] for q in questions],
               "questions": [q["q"] for q in questions],
               "answers": [q["answer"] for q in questions],
               "unmatched_docs": unmatched, "runtime_sec": time.time() - t0}
    out = Path(f"/tmp/hipporag_{DATASET}_n{len(questions)}.json")
    out.write_text(json.dumps(payload, ensure_ascii=False))
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True, private=True)
    api.upload_file(path_or_fileobj=str(out), path_in_repo=f"runs/{out.name}",
                    repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"업로드: runs/{out.name}")


if __name__ == "__main__":
    main()
