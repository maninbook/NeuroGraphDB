# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow", "huggingface_hub>=0.28"]
# ///
"""FRAMES 코퍼스 구축 — 위키 덤프에서 근거 문서 본문을 뽑아 재사용 가능한 풀로 만든다.

FRAMES(구글, 2024)는 질문과 **위키피디아 링크**만 주고 본문을 안 준다.
질문 824개, 고유 근거 문서 2,506개, 질문당 평균 3.2개(최대 11개).
HotpotQA(2개)보다 홉이 깊고, 위키 제목이 정답 단위라 우리 엣지 정의와 구조가 같다.

여기서는 **한 번만** 덤프를 훑어 그 2,506개 본문을 추출하고 결과 repo에 올린다.
이후 실험은 이 작은 파일만 받아 쓰면 된다.

덤프는 `wikimedia/wikipedia` 20231101.en (41조각). 날짜가 고정돼 재현이 된다.
살아 있는 위키 API를 쓰면 편하지만 시간이 지나면 재현이 깨진다.

**주의 — 제목 매칭이 관문이다.** URL에서 뽑은 제목에 앵커(#절)와 퍼센트 인코딩이 섞여 있다.
첫 조각을 처리한 뒤 매칭률을 찍어서, 기대치(약 1/41 = 2.4%)에 못 미치면 바로 드러난다.
"""

import json
import re
import sys
import time
import unicodedata
import urllib.parse
from pathlib import Path

DUMP = "wikimedia/wikipedia"
DUMP_CFG = "20231101.en"
FRAMES = "google/frames-benchmark"
RESULTS_REPO = "goethe0101/neurographdb-results"
PARQUET_REV = "refs/convert/parquet"
MAX_FILES = int(sys.argv[1]) if len(sys.argv) > 1 else 0     # 0 = 전체 41


def canon(t: str) -> str:
    """비교용 제목 정규화. 앵커 제거 → 유니코드 정규화 → 소문자 → 공백 정리."""
    t = t.split("#")[0]
    t = unicodedata.normalize("NFKC", t).replace("_", " ")
    return " ".join(t.lower().split())


def frames_titles():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    f = hf_hub_download(FRAMES, "default/test/0000.parquet",
                        repo_type="dataset", revision=PARQUET_REV)
    rows = pq.read_table(f).to_pylist()
    want, per_q = {}, []
    for r in rows:
        ts = []
        for k, v in r.items():
            if k.startswith("wikipedia_link") and v and str(v) != "None":
                raw = urllib.parse.unquote(str(v).rsplit("/", 1)[-1])
                c = canon(raw)
                if c:
                    want.setdefault(c, raw.split("#")[0].replace("_", " "))
                    ts.append(c)
        per_q.append({"q": r["Prompt"], "answer": str(r["Answer"]),
                      "gold": sorted(set(ts)),
                      "reasoning": r.get("reasoning_types", "")})
    return want, per_q


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    t0 = time.time()
    want, questions = frames_titles()
    print(f"FRAMES 질문 {len(questions)} | 찾아야 할 고유 제목 {len(want):,}")

    api = HfApi()
    files = sorted(f for f in api.list_repo_files(DUMP, repo_type="dataset")
                   if f.startswith(f"{DUMP_CFG}/") and f.endswith(".parquet"))
    if MAX_FILES:
        files = files[:MAX_FILES]
    print(f"덤프 조각 {len(files)}개 훑는다\n")

    found = {}
    for n, f in enumerate(files, 1):
        p = hf_hub_download(DUMP, f, repo_type="dataset")
        t = pq.read_table(p, columns=["title", "text"])
        d = t.to_pydict()
        for ti, tx in zip(d["title"], d["text"]):
            c = canon(ti)
            if c in want and c not in found:
                found[c] = (ti, tx)
        Path(p).unlink(missing_ok=True)          # 디스크 아끼려고 바로 지운다
        print(f"  [{time.time()-t0:6.1f}s] 조각 {n}/{len(files)} · "
              f"누적 매칭 {len(found):,}/{len(want):,} ({len(found)/len(want):.1%})")
        if n == 1 and len(found) / len(want) < 0.010:
            print("  ! 첫 조각 매칭률이 기대치(약 2.4%)의 절반도 안 된다. "
                  "제목 정규화를 의심할 것")

    miss = sorted(set(want) - set(found))
    print(f"\n매칭 {len(found):,}/{len(want):,} ({len(found)/len(want):.1%}) "
          f"· 못 찾음 {len(miss):,}")
    if miss:
        print("  못 찾은 예:", [want[m] for m in miss[:10]])

    # 근거를 전부 찾은 질문만 평가에 쓸 수 있다
    ok = [q for q in questions if all(g in found for g in q["gold"])]
    print(f"근거를 전부 확보한 질문 {len(ok)}/{len(questions)} "
          f"({len(ok)/len(questions):.1%})")

    titles = [found[c][0] for c in sorted(found)]
    texts = [found[c][1] for c in sorted(found)]
    key2title = {c: found[c][0] for c in found}
    for q in ok:
        q["gold_titles"] = [key2title[g] for g in q["gold"]]

    out_c = Path("/tmp/frames_corpus.parquet")
    pq.write_table(pa.table({"title": titles, "text": texts}), out_c)
    out_q = Path("/tmp/frames_questions.json")
    out_q.write_text(json.dumps(ok, ensure_ascii=False))
    mb = out_c.stat().st_size / 1e6
    print(f"\n코퍼스 {len(titles):,}문서 ({mb:.0f}MB) · 질문 {len(ok)}")

    api.upload_file(path_or_fileobj=str(out_c), path_in_repo="frames/corpus.parquet",
                    repo_id=RESULTS_REPO, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(out_q), path_in_repo="frames/questions.json",
                    repo_id=RESULTS_REPO, repo_type="dataset")
    print(f"업로드 완료 · 총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
