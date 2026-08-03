# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy", "torch", "transformers>=4.40", "huggingface_hub>=0.28",
# ]
# ///
"""임베딩 kNN 엣지 — 연결률을 올리되 lift를 지킬 수 있는가.

공식 코퍼스에서 제목 언급 엣지만으로는 MuSiQue 연결률이 **23.4%**다(관문 40% 미달).
2Wiki 66.6% · HotpotQA 59.7%는 통과. **MuSiQue를 살리려면 엣지를 더 잇는 수밖에 없다.**

PropRAG도 같은 걸 쓴다 — 개체 임베딩 유사도 ≥0.8을 동의어 엣지로 잇는다. **LLM이 필요 없다.**
우리는 문단 임베딩을 이미 갖고 있으니 문단-문단 kNN 엣지가 공짜다.

**과거의 교훈을 그대로 적용한다.** 논항 공유 엣지는 연결률은 올렸지만
lift가 952 → 78로 무너져 기각됐다. 연결률만 보면 안 되고 **lift를 같이 봐야 한다.**
기준선: 제목 언급 엣지의 lift가 공식 코퍼스에서 1267 / 5327 / 4592다.

측정: k와 유사도 문턱을 쓸어가며 (연결률, 무작위쌍 연결률, lift, 노드당 엣지).
**연결률이 올라도 lift가 한 자릿수 배로 떨어지면 쓸 수 없다.**

임베더는 bge-base로 먼저 본다 — 싸고, 여기서 lift가 무너지면 NV-Embed-v2에서도
쓸 이유가 없다. 통과하면 그때 큰 임베더로 다시 잰다.
"""

import json
import random
import re
import sys
import time
from itertools import permutations
from pathlib import Path

REPO = "osunlp/HippoRAG_v2"
RESULTS_REPO = "goethe0101/neurographdb-results"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
DATASETS = {"musique": "musique", "2wiki": "2wikimultihopqa", "hotpotqa": "hotpotqa"}
MAXW = 6
N_RAND = 200_000
KS = (1, 2, 5, 10)
THRESHOLDS = (0.0, 0.70, 0.80, 0.85)      # 0.0 = 문턱 없음(순수 kNN)
_N = re.compile(r"[^a-z0-9 ]+")


def norm(s):
    return " ".join(_N.sub(" ", s.lower()).split())


def load_official(name):
    from huggingface_hub import hf_hub_download
    key = DATASETS[name]
    corpus = json.load(open(hf_hub_download(REPO, f"{key}_corpus.json", repo_type="dataset")))
    qs = json.load(open(hf_hub_download(REPO, f"{key}.json", repo_type="dataset")))
    titles = [c["title"] for c in corpus]
    texts = [c["text"] for c in corpus]
    out = []
    for r in qs:
        if name == "musique":
            gold = sorted({p["title"] for p in r["paragraphs"] if p.get("is_supporting")})
        else:
            gold = sorted({sf[0] for sf in r["supporting_facts"]})
        out.append({"q": r["question"], "gold": gold})
    return titles, texts, out


def title_edges(titles, texts):
    lookup = {}
    for i, t in enumerate(titles):
        k = norm(t)
        if len(k) >= 5:
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


def main():
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    mdl = AutoModel.from_pretrained(EMBED_MODEL).to(dev).eval()

    def encode(items, batch=128):
        out = []
        for a in range(0, len(items), batch):
            b = tok(items[a:a + batch], padding=True, truncation=True,
                    max_length=512, return_tensors="pt").to(dev)
            with torch.no_grad():
                h = mdl(**b).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(h, dim=-1).cpu())
        return torch.cat(out)

    rng = random.Random(0)
    report = {}
    for name in DATASETS:
        titles, texts, questions = load_official(name)
        tidx = {t: i for i, t in enumerate(titles)}
        n = len(titles)
        te = title_edges(titles, texts)
        P = encode([f"{t}. {x}" for t, x in zip(titles, texts)]).to(dev)
        print(f"\n{'='*78}\n{name} — 문단 {n:,} · 제목엣지 {len(te):,}\n{'='*78}")

        gold_sets = []
        for q in questions:
            ids = [tidx[g] for g in q["gold"] if g in tidx]
            if len(ids) >= 2:
                gold_sets.append(ids)
        rp = [(rng.randrange(n), rng.randrange(n)) for _ in range(N_RAND)]
        rp = [(a, b) for a, b in rp if a != b]

        def score(edges, label):
            hit = sum(1 for g in gold_sets
                      if any((a, b) in edges or (b, a) in edges
                             for a, b in permutations(g, 2)))
            rnd = sum(1 for a, b in rp if (a, b) in edges or (b, a) in edges)
            gr, rr = hit/max(len(gold_sets),1), rnd/max(len(rp),1)
            lift = gr/rr if rr > 0 else float("inf")
            print(f"  {label:<26}{gr:>8.1%}{rr:>10.3%}{lift:>9.0f}{len(edges)/n:>11.2f}")
            return {"linkage": gr, "random": rr, "lift": lift,
                    "edges_per_node": len(edges)/n, "n_edges": len(edges)}

        print(f"  {'엣지 구성':<26}{'정답쌍':>8}{'무작위':>10}{'lift':>9}{'엣지/노드':>11}")
        rows = {"제목만": score(te, "제목만")}

        # kNN 후보를 한 번에 계산 (자기 자신 제외)
        topk = max(KS)
        sims, idxs = [], []
        B = 512
        for a in range(0, n, B):
            s = P[a:a+B] @ P.T
            for r in range(s.shape[0]):
                s[r, a+r] = -1.0
            v, ix = torch.topk(s, topk, dim=1)
            sims.append(v.cpu()); idxs.append(ix.cpu())
        sims = torch.cat(sims).numpy(); idxs = torch.cat(idxs).numpy()

        for k in KS:
            for th in THRESHOLDS:
                knn = set()
                for i in range(n):
                    for c in range(k):
                        if sims[i, c] >= th:
                            knn.add((i, int(idxs[i, c])))
                rows[f"제목+kNN k={k} th={th}"] = score(te | knn, f"제목+kNN k={k} th={th}")

        report[name] = rows

    print(f"\n판정 기준 — 연결률이 올라도 **lift가 한 자릿수 배로 떨어지면 쓸 수 없다.**")
    print(f"(논항 공유 엣지가 952 → 78로 무너져 기각된 전례가 있다)")
    print(f"\n({time.time()-t0:.0f}초)")

    out = Path("/tmp/knn_edges_official.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=str(out), path_in_repo=f"runs/{out.name}",
                        repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"업로드: runs/{out.name}")


if __name__ == "__main__":
    main()
