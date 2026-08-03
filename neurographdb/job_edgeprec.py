# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "huggingface_hub>=0.28"]
# ///
"""D-PREC — 엣지 **정밀도**를 정확히 잰다. lift가 왜 잘못된 관문이었는지 확정한다.

E3에서 드러난 것: 2Wiki df<=2 다리 문자열의 **lift가 1035**인데도 엣지를 넣으면
Recall@5가 떨어진다(92.70 → 91.25). lift가 높은데 왜 해로운가.

    **lift는 비율이지 정밀도가 아니다.**
    정답쌍이 무작위쌍보다 1000배 자주 공유해도, 가능한 쌍이 6,119^2이므로
    새로 생긴 엣지의 대부분은 여전히 정답쌍이 아니다.

이 프로젝트에서 **세 번** 같은 방식으로 실패했다. 전부 lift로 관문을 세웠다:
    논항 공유    lift 31~74    기각
    kNN 엣지     (유사도 기반) 기각
    다리 엣지    lift 350~1035 기각

여기서 재는 것은 단 하나 —
    **정밀도 = (엣지 중 정답쌍) / (전체 엣지)**
그리고 비교 기준으로 제목 엣지의 정밀도.

제목 엣지가 왜 되고 나머지는 왜 안 되는지가 이 숫자 하나로 갈리는지 확인한다.
갈리면 앞으로의 관문은 lift가 아니라 정밀도다.

회수도 같이 낸다 — 정밀도만 높고 회수가 0이면 그것도 쓸모없다.
    회수 = (엣지로 이어진 정답쌍) / (전체 정답쌍)

**GPU 불필요, cpu-upgrade에서 무료.**
"""

import json
import re
from collections import defaultdict
from pathlib import Path

RESULTS_REPO = "goethe0101/neurographdb-results"
OFFICIAL = "osunlp/HippoRAG_v2"
DATASETS = {"musique": "musique", "2wiki": "2wikimultihopqa", "hotpotqa": "hotpotqa"}
THRESHOLDS = (2, 3, 5, 10, 20)
MAXW, BSPAN = 6, 4

_N = re.compile(r"[^a-z0-9 ]+")
_TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9.'\-]*")
_CAP = re.compile(r"^[A-Z][A-Za-z0-9.'\-]*$")
_YEAR = re.compile(r"^(1[0-9]{3}|20[0-9]{2})$")
_STOP = {"The", "A", "An", "He", "She", "It", "They", "This", "That", "In", "On",
         "At", "For", "As", "But", "And", "Or", "His", "Her", "Its", "Their",
         "After", "Before", "When", "While", "During", "However", "There",
         "These", "Those", "Some", "Both", "Also", "Then", "Since", "By", "Of",
         "From", "With", "To", "Was", "Is", "Are", "Were", "Has", "Have", "Had"}


def norm(s):
    return " ".join(_N.sub(" ", s.lower()).split())


def spans(text):
    toks = _TOK.findall(text)
    out, i, n = set(), 0, len(toks)
    while i < n:
        if _YEAR.match(toks[i]):
            out.add(toks[i]); i += 1; continue
        if _CAP.match(toks[i]) and toks[i] not in _STOP:
            j = i
            while j < n and _CAP.match(toks[j]) and toks[j] not in _STOP and j - i < BSPAN:
                j += 1
            run = toks[i:j]
            for a in range(len(run)):
                for b in range(a + 1, min(a + BSPAN, len(run)) + 1):
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
    gold_pairs = set()
    for r in qs:
        if name == "musique":
            ids = sorted({by_text[nz(p["paragraph_text"])] for p in r["paragraphs"]
                          if p.get("is_supporting") and nz(p["paragraph_text"]) in by_text})
        else:
            ids = sorted({by_title[sf[0]] for sf in r["supporting_facts"]
                          if sf[0] in by_title})
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                gold_pairs.add((ids[a], ids[b]))
    return [c["title"] for c in corpus], [c["text"] for c in corpus], gold_pairs


def undirected(edges):
    return {(min(a, b), max(a, b)) for a, b in edges if a != b}


def main():
    report = {}
    for ds in DATASETS:
        titles, texts, gold = load_official(ds)
        n = len(titles)
        print(f"\n{'='*78}\n{ds}  문단 {n:,}  정답쌍 {len(gold):,}"
              f"  (가능한 쌍 {n*(n-1)//2:,})\n{'='*78}")

        # 제목 엣지
        lookup = {}
        for i, t in enumerate(titles):
            k = norm(t)
            if len(k) >= 5:
                lookup.setdefault(k, i)
        te = set()
        for i, body in enumerate(texts):
            toks = norm(body).split()
            for w in range(1, MAXW + 1):
                for a in range(len(toks) - w + 1):
                    j = lookup.get(" ".join(toks[a:a + w]))
                    if j is not None and j != i:
                        te.add((i, j))
        te = undirected(te)

        # 다리 문자열 색인
        post = defaultdict(list)
        for i, s in enumerate(spans_all := [spans(t + " " + b)
                                            for t, b in zip(titles, texts)]):
            for x in s:
                post[x].append(i)

        print(f"  {'엣지 집합':<16}{'엣지수':>11}{'정답쌍포함':>11}"
              f"{'정밀도%':>10}{'회수%':>9}{'lift':>10}")

        rows = {}
        base_rate = len(gold) / (n * (n - 1) / 2)

        def show(name, es):
            es = undirected(es)
            hit = len(es & gold)
            prec = hit / len(es) * 100 if es else 0.0
            rec = hit / len(gold) * 100 if gold else 0.0
            lift = (prec / 100) / base_rate if base_rate > 0 else float("inf")
            print(f"  {name:<16}{len(es):>11,}{hit:>11,}{prec:>10.2f}{rec:>9.1f}{lift:>10.0f}")
            rows[name] = {"edges": len(es), "gold_hit": hit, "precision": prec,
                          "recall": rec, "lift": lift}

        show("제목 언급", te)
        for T in THRESHOLDS:
            es = set()
            for x, v in post.items():
                if 2 <= len(v) <= T:
                    for a in range(len(v)):
                        for b in range(a + 1, len(v)):
                            es.add((v[a], v[b]))
            show(f"다리 df<={T}", es)
            if T == THRESHOLDS[0]:
                show(f"제목+다리 df<={T}", te | es)

        report[ds] = rows

    print(f"\n{'='*78}\n결론\n{'='*78}")
    print(f"  {'':<10}{'제목 정밀도':>12}{'다리df2 정밀도':>15}{'배수':>8}")
    for ds, r in report.items():
        p1 = r["제목 언급"]["precision"]
        p2 = r["다리 df<=2"]["precision"]
        print(f"  {ds:<10}{p1:>12.2f}{p2:>15.2f}{p1/max(p2,1e-9):>8.1f}x")
    print("\n  lift는 둘 다 높다. 갈리는 것은 **정밀도**다.")
    print("  → 앞으로 엣지 기제의 관문은 lift가 아니라 정밀도로 세운다.")

    Path("/tmp/edgeprec.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj="/tmp/edgeprec.json",
                        path_in_repo="runs/edge_precision.json",
                        repo_id=RESULTS_REPO, repo_type="dataset")


if __name__ == "__main__":
    main()
