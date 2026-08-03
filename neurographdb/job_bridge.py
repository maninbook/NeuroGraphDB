# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "huggingface_hub>=0.28"]
# ///
"""D-BRIDGE — 다리 개체가 **문자열로** 존재하는가. 순수 측정, 아무것도 짓지 않는다.

D-CUE2가 알려준 것 (공식 코퍼스 + NV-Embed-v2 + Recall@5):
    MuSiQue  해당가능 62.2%인데 **오라클조차 +4.43**, 실제는 +0.00 (전 구간 손해)
    진짜 정답 문단을 단서로 줘도 빠진 정답이 안 올라온다.

진단: **문단 임베딩은 문단 전체에 관한 벡터라 다리 개체가 희석된다.**
'인셉션 문단'을 통째로 더하면 '크리스토퍼 놀란'이 개봉연도·장르·줄거리에 묻힌다.
LogicRAG이 LLM을 쓰는 이유가 이것이다 — 다리를 **뽑아내** 새 질의로 써야 한다.

벡터 덧셈이 희석시키는 것을 **문자열 일치는 정확히 분리한다.**
그리고 그건 LLM 없이 된다.

── 무엇을 재는가 ────────────────────────────────────────────────────────────
찾은 정답 문단(발판)에서 **질문에 없던 고유명사 문자열**을 뽑는다.
그것이 다리 후보다. 그리고 두 가지를 잰다:

R. 회수 가능성  빠진 정답 문단이 그 다리 문자열을 **담고 있는가**.
                담고 있지 않으면 문자열로도 못 건넌다 — 아이디어가 죽는다.
                제목 언급 연결률 23.4%와 **직접 비교되는 숫자**다.

P. 변별력      그 다리 문자열이 코퍼스 전체에서 **몇 개 문단에 나오는가**.
                'New York'처럼 500개에 나오면 다리 구실을 못 한다.
                논항 공유가 실패한 이유가 정확히 이것이었다(lift 붕괴).
                lift = P(다리공유 | 정답쌍) / P(다리공유 | 무작위쌍)

**둘 다 봐야 한다.** R만 높고 P가 낮으면 예전 실패의 재판이다.

비교 기준을 같이 낸다 — 제목 언급 엣지의 R과 lift. 새 방식이 그보다 나은가.
세 데이터셋 다 잰다. MuSiQue만 재면 그게 과적합이다.

임베딩은 캐시돼 있다. **GPU 불필요, cpu-upgrade에서 무료.**
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

RESULTS_REPO = "goethe0101/neurographdb-results"
OFFICIAL = "osunlp/HippoRAG_v2"
EMBED_TAG = "NV-Embed-v2"
DATASETS = {"musique": "musique", "2wiki": "2wikimultihopqa", "hotpotqa": "hotpotqa"}
K = 5
N_RANDOM = 20000          # lift의 분모용 무작위 쌍
MAXW = 4                  # 다리 문자열 최대 토큰 수

_TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9.'\-]*")
_CAP = re.compile(r"^[A-Z][A-Za-z0-9.'\-]*$")
_YEAR = re.compile(r"^(1[0-9]{3}|20[0-9]{2})$")
# 문장 첫머리의 관사/대명사가 대문자라서 딸려오는 것들. 다리가 될 수 없다.
_STOP = {"The", "A", "An", "He", "She", "It", "They", "This", "That", "In", "On",
         "At", "For", "As", "But", "And", "Or", "His", "Her", "Its", "Their",
         "After", "Before", "When", "While", "During", "However", "There",
         "These", "Those", "Some", "Both", "Also", "Then", "Since", "By", "Of",
         "From", "With", "To", "Was", "Is", "Are", "Were", "Has", "Have", "Had"}


def spans(text):
    """대문자 연속 구간과 연도를 뽑는다. 파서를 쓰지 않는다 —
    spaCy 의존 파싱은 이 프로젝트에서 두 번 실패했고, 취약함이 원인이었다."""
    toks = _TOK.findall(text)
    out, i, n = set(), 0, len(toks)
    while i < n:
        if _YEAR.match(toks[i]):
            out.add(toks[i]); i += 1; continue
        if _CAP.match(toks[i]) and toks[i] not in _STOP:
            j = i
            while j < n and _CAP.match(toks[j]) and toks[j] not in _STOP and j - i < MAXW:
                j += 1
            run = toks[i:j]
            # 연속 구간 전체와 그 접두어들. 'Christopher Nolan'과 'Nolan' 둘 다 다리가 된다.
            for a in range(len(run)):
                for b in range(a + 1, min(a + MAXW, len(run)) + 1):
                    s = " ".join(run[a:b])
                    if len(s) >= 3:
                        out.add(s)
            i = j
        else:
            i += 1
    return out


def load_official(name):
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
        out.append({"q": r["question"], "gold": ids})
    return corpus, out


def main():
    import numpy as np
    from huggingface_hub import hf_hub_download

    rng = random.Random(0)
    report = {}

    for ds in DATASETS:
        corpus, questions = load_official(ds)
        n = len(corpus)
        titles = [c["title"] for c in corpus]
        texts = [c["text"] for c in corpus]
        print(f"\n{'='*78}\n{ds}  문단 {n:,}  질문 {len(questions):,}\n{'='*78}")

        # 문단별 다리 문자열 집합 + 코퍼스 전체 빈도
        print("  문자열 추출 중...")
        S = [spans(t + " " + b) for t, b in zip(titles, texts)]
        df = Counter()
        for s in S:
            df.update(s)
        print(f"  고유 문자열 {len(df):,}개, 문단당 평균 {sum(len(x) for x in S)/n:.0f}개")

        # 발판 = 상위5에 든 진짜 정답. 임베딩으로 실제 상위5를 구한다.
        P = np.load(hf_hub_download(RESULTS_REPO, f"emb/{ds}_{EMBED_TAG}_P.npy",
                                    repo_type="dataset")).astype(np.float32)
        Q = np.load(hf_hub_download(RESULTS_REPO, f"emb/{ds}_{EMBED_TAG}_Q.npy",
                                    repo_type="dataset")).astype(np.float32)
        P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
        Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
        base = np.argsort(-(P @ Q.T), axis=0)[:K].T

        # ── R. 회수 가능성 ──────────────────────────────────────────────────
        # 발판 정답 → 빠진 정답으로 건널 다리 문자열이 존재하는가
        nb = [0] * 6          # 변별력 등급별 카운트
        addressable = bridged = 0
        title_bridged = 0
        dfs_of_bridges = []
        for qi, rec in enumerate(questions):
            g = rec["gold"]
            if not g:
                continue
            top = set(base[qi].tolist())
            found = [v for v in g if v in top]
            missing = [v for v in g if v not in top]
            if not found or not missing:
                continue
            addressable += 1
            qspans = spans(rec["q"])
            # 발판에서 **질문에 없던** 문자열만 다리 후보다
            cand = set()
            for f in found:
                cand |= S[f]
            cand -= qspans
            ok = False
            for mv in missing:
                shared = cand & S[mv]
                if shared:
                    ok = True
                    dfs_of_bridges.append(min(df[s] for s in shared))
            if ok:
                bridged += 1
            # 비교 기준: 제목 언급으로 건널 수 있었는가
            if any(any(titles[f].lower() in texts[mv].lower() or
                       titles[mv].lower() in texts[f].lower()
                       for f in found) for mv in missing):
                title_bridged += 1

        print(f"\n[R] 회수 가능성 — 발판에서 빠진 정답으로 건널 다리가 있는가")
        print(f"    해당 가능(발판 있고 빠진 정답 있음)  {addressable:5d}")
        print(f"    **고유명사 다리 존재**              {bridged:5d} "
              f"({bridged/max(addressable,1)*100:5.1f}%)")
        print(f"    제목 언급으로 건널 수 있음(비교)     {title_bridged:5d} "
              f"({title_bridged/max(addressable,1)*100:5.1f}%)")

        if dfs_of_bridges:
            a = np.array(dfs_of_bridges)
            print(f"\n    다리 문자열의 코퍼스 빈도 (작을수록 변별력 높음)")
            print(f"      중앙값 {np.median(a):.0f}  25% {np.percentile(a,25):.0f}  "
                  f"75% {np.percentile(a,75):.0f}  최대 {a.max():.0f}")
            for thr in (2, 5, 10, 50):
                print(f"      {thr}개 문단 이하에만 나오는 다리: "
                      f"{(a<=thr).mean()*100:5.1f}%")

        # ── P. 변별력 (lift) ────────────────────────────────────────────────
        # 정답쌍이 다리를 공유할 확률 vs 무작위쌍이 공유할 확률
        gold_pairs = []
        for rec in questions:
            g = rec["gold"]
            for a in range(len(g)):
                for b in range(a + 1, len(g)):
                    gold_pairs.append((g[a], g[b]))
        rng.shuffle(gold_pairs)
        gold_pairs = gold_pairs[:N_RANDOM]

        def share_rate(pairs, maxdf):
            hit = 0
            for a, b in pairs:
                inter = S[a] & S[b]
                if any(df[s] <= maxdf for s in inter):
                    hit += 1
            return hit / max(len(pairs), 1)

        rand_pairs = [(rng.randrange(n), rng.randrange(n)) for _ in range(N_RANDOM)]
        rand_pairs = [(a, b) for a, b in rand_pairs if a != b]

        print(f"\n[P] 변별력 — 정답쌍이 무작위쌍보다 다리를 얼마나 더 공유하는가")
        print(f"    {'빈도상한':>9}{'정답쌍':>9}{'무작위쌍':>10}{'lift':>9}")
        lifts = {}
        for maxdf in (2, 5, 10, 50, 10**9):
            gr = share_rate(gold_pairs, maxdf)
            rr = share_rate(rand_pairs, maxdf)
            lf = gr / rr if rr > 0 else float("inf")
            lifts[maxdf] = lf
            lbl = "제한없음" if maxdf > 10**8 else str(maxdf)
            print(f"    {lbl:>9}{gr*100:>8.1f}%{rr*100:>9.2f}%{lf:>9.1f}")

        report[ds] = {
            "addressable": addressable,
            "bridged_pct": bridged / max(addressable, 1) * 100,
            "title_bridged_pct": title_bridged / max(addressable, 1) * 100,
            "bridge_df_median": float(np.median(dfs_of_bridges)) if dfs_of_bridges else None,
            "lift": {str(k): (None if v == float("inf") else v) for k, v in lifts.items()},
        }

    print(f"\n{'='*78}\n전체 요약\n{'='*78}")
    print(f"  {'':<10}{'해당가능':>9}{'다리있음%':>11}{'제목%':>8}{'다리빈도중앙':>13}{'lift(<=5)':>11}")
    for ds, v in report.items():
        lf = v["lift"].get("5")
        print(f"  {ds:<10}{v['addressable']:>9d}{v['bridged_pct']:>11.1f}"
              f"{v['title_bridged_pct']:>8.1f}"
              f"{(v['bridge_df_median'] or 0):>13.0f}{(lf or 0):>11.1f}")
    print("\n  판정 기준:")
    print("    다리있음%가 제목%보다 크게 높지 않으면 → 새로 얻을 게 없다. 접는다")
    print("    다리있음%는 높은데 lift가 낮으면 → 논항공유 실패의 재판이다. 접는다")
    print("    둘 다 좋으면 → 사전등록하고 dev/test로 검증한다")
    print("  ※ 이것은 **진단**이다. 아직 주장이 아니다.")

    Path("/tmp/bridge.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj="/tmp/bridge.json",
                        path_in_repo="bridge_diagnostic.json",
                        repo_id=RESULTS_REPO, repo_type="dataset")


if __name__ == "__main__":
    main()
