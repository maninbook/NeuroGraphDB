# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "huggingface_hub>=0.28"]
# ///
"""D-CUE2 — 단서 갱신을 **SOTA 환경에서 다시 잰다.** 만들기 전에 상한부터.

job_cue.py(2026-08-02)는 자체 풀 + bge-base + all@10에서 쟀고 MuSiQue 13.4%p 회수가
나왔다. 그 환경은 이제 안 쓴다. 여기서는 **공식 코퍼스 + NV-Embed-v2 + Recall@5**로
다시 잰다. 임베더가 훨씬 세졌으므로 예전 상한이 그대로 남아 있을 이유가 없다.

── 왜 MuSiQue에 이게 필요한가 ────────────────────────────────────────────────
MuSiQue 근거쌍 연결률 23.4%는 우연이 아니라 **설계 결과**다. 단일홉 질문들을 서로
다른 출처에서 가져와 합성했으므로 근거 문단이 애초에 링크된 적이 없다.
**엣지를 손봐서는 안 뚫린다** — kNN·논항공유가 모두 실패한 이유가 그것이다.
구조가 코퍼스가 아니라 **질문**에 있다.

"인셉션 감독의 배우자는?" — 배우자가 적힌 놀란 문단에는 '인셉션'이 없을 수 있다.
질문과 어휘가 안 겹치니 어떤 임베딩도 못 찾는다. 다리 개체 '놀란'은
**1홉의 답에서만 나온다.** LogicRAG은 그걸 LLM으로 써낸다(질문당 호출 비용).

단서 갱신은 그 **LLM 없는 판본**이다 — 찾은 문단의 임베딩을 질의에 더하면
LLM이 '놀란'이라고 써주지 않아도 그 정보가 질의 벡터로 밀려 들어간다.
SAM(Raaijmakers & Shiffrin 1981): 인출된 내용이 다음 인출의 단서가 된다.

    C0 = q
    C1 = normalize(q + gamma * mean(1라운드 상위 m개 문단 임베딩))
    최종 순위 = sim(모든 문단, C1)

**척도 문제가 없다.** 예전 점수합치기·SAM 시도의 실패는 활성값과 dense 점수를
섞은 데서 왔다. 여기서는 단서를 하나로 만들고 **순위를 한 번만** 매긴다. 합치기가 없다.
gamma=0이면 기준선과 완전히 같다 — 안전장치가 공짜로 딸려온다.

── 이 스크립트는 짓지 않는다. 재기만 한다 ────────────────────────────────────
A. 해당 가능 집합   상위 5에 정답이 **일부만** 든 질문의 비율.
                    단서 갱신이 도울 수 있는 건 이것뿐이다. 작으면 거기서 끝이다.
                    '하나도 못 찾음'은 단서로 쓸 발판 자체가 없다.
B. 오라클 단서      C1을 **상위5에 든 진짜 정답**으로 만들었을 때. = 기제의 상한.
                    정답을 줘도 안 오르면 아이디어가 틀린 것이다.
C. 실제 단서        C1을 **상위 m개**(정답인지 모름)로 만들었을 때. 쓸 수 있는 값.
                    B와의 격차 = 단서 품질이 깎아먹는 양.
D. 부수 피해        gamma 때문에 원래 맞히던 정답을 **잃는** 양.
                    all@10에서 배운 것 — 얻은 만큼 잃으면 0이다. 이번엔 먼저 잰다.

세 데이터셋 다 잰다. MuSiQue만 재면 그게 과적합이다.
채택 여부는 여기서 정하지 않는다 — 상한이 나오면 dev/test로 따로 검증한다.

임베딩은 캐시돼 있다. **GPU 불필요, cpu-upgrade에서 무료.**
"""

import json
from pathlib import Path

RESULTS_REPO = "goethe0101/neurographdb-results"
OFFICIAL = "osunlp/HippoRAG_v2"
EMBED_TAG = "NV-Embed-v2"
DATASETS = {"musique": "musique", "2wiki": "2wikimultihopqa", "hotpotqa": "hotpotqa"}
GAMMAS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8)
MS = (1, 3, 5)
K = 5


def load_official(name):
    """정답은 **문단 인덱스**로 확정한다 (MuSiQue 제목 중복. SOTA.md 정정 참조)."""
    from huggingface_hub import hf_hub_download
    key = DATASETS[name]
    corpus = json.load(open(hf_hub_download(OFFICIAL, f"{key}_corpus.json",
                                            repo_type="dataset")))
    qs = json.load(open(hf_hub_download(OFFICIAL, f"{key}.json", repo_type="dataset")))
    nz = lambda s: " ".join(s.split())
    by_text, by_title = {}, {}
    for i, c in enumerate(corpus):
        by_text.setdefault(nz(c["text"]), i)
        by_title.setdefault(c["title"], i)
    out = []
    for r in qs:
        if name == "musique":
            ids = sorted({by_text[nz(p["paragraph_text"])] for p in r["paragraphs"]
                          if p.get("is_supporting") and nz(p["paragraph_text"]) in by_text})
        else:
            ids = sorted({by_title[sf[0]] for sf in r["supporting_facts"]
                          if sf[0] in by_title})
        out.append(ids)
    return len(corpus), out


def main():
    import numpy as np
    from huggingface_hub import hf_hub_download

    report = {}
    for ds in DATASETS:
        n, golds = load_official(ds)
        P = np.load(hf_hub_download(RESULTS_REPO, f"emb/{ds}_{EMBED_TAG}_P.npy",
                                    repo_type="dataset")).astype(np.float32)
        Q = np.load(hf_hub_download(RESULTS_REPO, f"emb/{ds}_{EMBED_TAG}_Q.npy",
                                    repo_type="dataset")).astype(np.float32)
        # 캐시가 정규화돼 있더라도 가정하지 않는다 — 아니면 gamma 척도가 통째로 어긋난다
        P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
        Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9

        print(f"\n{'='*78}\n{ds}  문단 {n:,}  질문 {len(golds):,}\n{'='*78}")

        S0 = P @ Q.T                                        # (n, nq)
        base = np.argsort(-S0, axis=0)[:K].T                # (nq, K)
        gsets = [set(g) for g in golds]

        def recall(rows):
            tot = m = 0.0
            for i, g in enumerate(golds):
                if not g:
                    continue
                tot += len(gsets[i] & set(rows[i].tolist())) / len(g); m += 1
            return tot / m * 100

        def delta(rows):
            gain = loss = 0
            for i, g in enumerate(golds):
                if not g:
                    continue
                a = gsets[i] & set(base[i].tolist())
                b = gsets[i] & set(rows[i].tolist())
                gain += len(b - a); loss += len(a - b)
            return gain, loss

        def rerank(cues, mask):
            """mask가 False인 질문은 기준 순위를 그대로 쓴다(단서를 못 만든 경우)."""
            rows = base.copy()
            idx = np.flatnonzero(mask)
            if len(idx):
                C = cues[idx]
                C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
                S = P @ C.T                                  # (n, |idx|)
                top = np.argpartition(-S, K, axis=0)[:K].T   # (|idx|, K)
                for r, i in enumerate(idx):
                    rows[i] = top[r]
            return rows

        r0 = recall(base)

        # ── A. 해당 가능 집합 ────────────────────────────────────────────────
        partial = complete = empty = 0
        for i, g in enumerate(golds):
            if not g:
                continue
            got = gsets[i] & set(base[i].tolist())
            if not got:
                empty += 1
            elif len(got) < len(g):
                partial += 1
            else:
                complete += 1
        tot = partial + complete + empty
        print(f"\n[A] 상위 {K}의 상태 — 단서 갱신이 도울 수 있는 곳은 '일부만'뿐이다")
        print(f"    정답 전부 들어옴  {complete:5d} ({complete/tot*100:5.1f}%)  더 얻을 게 없다")
        print(f"    **일부만 들어옴** {partial:5d} ({partial/tot*100:5.1f}%)  ← 해당 가능")
        print(f"    하나도 못 찾음    {empty:5d} ({empty/tot*100:5.1f}%)  발판이 없다")
        print(f"    기준 Recall@{K} {r0:.2f}")

        # ── B. 오라클 단서 ──────────────────────────────────────────────────
        print(f"\n[B] 오라클 단서 (상위{K}에 든 **진짜 정답**으로 C1) — 기제 상한")
        print(f"    {'gamma':>7}{'R@5':>8}{'기준대비':>10}{'회수':>7}{'손실':>7}")
        omask = np.array([bool(gsets[i] & set(base[i].tolist())) for i in range(len(golds))])
        obest = (0.0, r0)
        for gm in GAMMAS:
            if gm == 0.0:
                print(f"    {gm:>7.1f}{r0:>8.2f}{0.0:>+10.2f}{0:>7d}{0:>7d}   (안전장치: 기준과 동일)")
                continue
            cues = np.zeros_like(Q)
            for i in range(len(golds)):
                if omask[i]:
                    hit = [v for v in base[i].tolist() if v in gsets[i]]
                    cues[i] = Q[i] + gm * P[hit].mean(axis=0)
            rows = rerank(cues, omask)
            r = recall(rows); g_, l_ = delta(rows)
            print(f"    {gm:>7.1f}{r:>8.2f}{r-r0:>+10.2f}{g_:>7d}{l_:>7d}")
            if r > obest[1]:
                obest = (gm, r)

        # ── C. 실제 단서 ────────────────────────────────────────────────────
        print(f"\n[C] 실제 단서 (상위 m개로 C1, 정답인지 모름) — 실제로 쓸 수 있는 값")
        print(f"    {'m':>3}{'gamma':>7}{'R@5':>8}{'기준대비':>10}{'회수':>7}{'손실':>7}")
        allm = np.ones(len(golds), dtype=bool)
        rbest = (0, 0.0, r0)
        for m in MS:
            for gm in GAMMAS:
                if gm == 0.0:
                    continue
                cues = Q + gm * P[base[:, :m]].mean(axis=1)
                rows = rerank(cues, allm)
                r = recall(rows); g_, l_ = delta(rows)
                print(f"    {m:>3}{gm:>7.1f}{r:>8.2f}{r-r0:>+10.2f}{g_:>7d}{l_:>7d}")
                if r > rbest[2]:
                    rbest = (m, gm, r)

        og, orv = obest
        rm, rg, rrv = rbest
        print(f"\n  요약 — 기준 {r0:.2f} / 오라클 {orv:.2f}(gamma {og}) / 실제 {rrv:.2f}(m={rm} gamma={rg})")
        if orv - r0 < 0.5:
            print(f"  판정: **기제가 틀렸다.** 정답을 단서로 줘도 안 오른다. 접는다")
        elif rrv - r0 < 0.5:
            print(f"  판정: 기제는 맞으나 **단서 품질이 못 따라간다** "
                  f"(상한 {orv-r0:+.2f} 중 {rrv-r0:+.2f}만 실현)")
        else:
            print(f"  판정: **실현 이득 {rrv-r0:+.2f}%p** — 사전등록하고 dev/test로 검증할 값어치")

        report[ds] = {"base": r0, "addressable_pct": partial / tot * 100,
                      "no_foothold_pct": empty / tot * 100,
                      "oracle": {"gamma": og, "r5": orv, "gain": orv - r0},
                      "real": {"m": rm, "gamma": rg, "r5": rrv, "gain": rrv - r0}}

    print(f"\n{'='*78}\n전체 요약\n{'='*78}")
    print(f"  {'':<10}{'기준':>7}{'해당가능%':>10}{'발판없음%':>10}{'오라클':>9}{'실제':>8}")
    for ds, v in report.items():
        print(f"  {ds:<10}{v['base']:>7.2f}{v['addressable_pct']:>10.1f}"
              f"{v['no_foothold_pct']:>10.1f}{v['oracle']['gain']:>+9.2f}{v['real']['gain']:>+8.2f}")
    print("\n  ※ 전체 1000개에서 잰 **진단**이다. 아직 주장이 아니다.")
    print("     채택하려면 dev/test 분할로 다시 검증한다 — 여기 gamma는 전체에서 골랐으므로 낙관적이다.")

    Path("/tmp/cue2.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj="/tmp/cue2.json",
                        path_in_repo="cue2_official_nvembed.json",
                        repo_id=RESULTS_REPO, repo_type="dataset")


if __name__ == "__main__":
    main()
