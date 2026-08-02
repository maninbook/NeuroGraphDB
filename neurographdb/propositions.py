"""명제·기호학 표현층 — 언급 엣지에 술어 타입과 극성을 얹는다.

설계 근거와 사전등록은 ../PROPOSITION.md에 있다. 요약하면:

  현재 엣지는 "A의 본문에 B의 제목이 있다"가 전부다. 무타입·무극성이라
  "누가 X를 감독했나"와 "X는 어디서 태어났나"가 **같은 엣지**를 탄다.
  depth≥2에서 도달 범위가 폭발하고 정밀도가 떨어진다(MuSiQue 손실 139건).

**엣지 집합은 통제군과 동일하게 둔다.** 타입과 극성만 추가한다.
엣지가 달라지면 "타입이 효과가 있었다"와 "엣지가 달라졌다"를 구분할 수 없다.
술어를 못 뽑은 엣지는 type=-1로 남고, 그건 확산에서 **모든 술어와 호환**이므로
정확히 통제군처럼 동작한다. 파싱 실패가 정답 경로를 막지 않는다.

**LLM을 쓰지 않는다.** 우리 우위가 "엣지 생성 LLM 비용 0"이라, 문단마다 LLM을 부르면
HippoRAG와 같아져 이 연구를 할 이유가 사라진다. spaCy 의존구문분석만 쓴다.
"""

from __future__ import annotations

import re
from collections import Counter

# 타입 상한. 이보다 드문 술어는 -1(무타입)로 떨어뜨린다.
# 드문 술어는 파싱 신뢰도가 낮고, 무타입은 통제군처럼 동작하므로 안전한 퇴화다.
MAX_TYPES = 64
MAX_TITLE_WORDS = 6
_NORM = re.compile(r"[^a-z0-9 ]+")


def _norm(s: str) -> str:
    return " ".join(_NORM.sub(" ", s.lower()).split())


def _predicate_and_polarity(span):
    """언급 span을 지배하는 술어와 극성을 뽑는다.

    술어 = 가장 가까운 동사의 표제어. **전치사는 타입에 넣지 않는다.**
    질문은 "Who directed X?"(direct)인데 본문은 "X was directed by N"(direct_by)이라
    전치사까지 붙이면 같은 관계인데 타입이 갈린다. 매칭을 살리는 쪽을 택했다.

    극성 = 지배 동사에 `neg` 의존이 달리면 음. "A는 B가 아니다"를 긍정 근거로
    쓰지 않기 위한 것이다.
    """
    node = span.root
    cur, hops = node, 0
    while cur.pos_ not in ("VERB", "AUX") and cur.head is not cur and hops < 5:
        cur = cur.head
        hops += 1
    if cur.pos_ not in ("VERB", "AUX"):
        return None, 1
    pol = -1 if any(c.dep_ == "neg" for c in cur.children) else 1
    return cur.lemma_.lower(), pol


def annotate_edges(titles, texts, edges, nlp, batch_size=64):
    """(src, dst, w) 엣지에 (술어타입, 극성)을 붙여 (src, dst, w, type, pol)로 만든다.

    edges는 build_mention_edges가 만든 것을 **그대로** 받는다. 여기서 추가하거나
    빼지 않는다. 반환 길이는 입력과 항상 같다.

    반환: (typed_edges, vocab, stats)
    """
    lookup = {}
    for i, t in enumerate(titles):
        key = _norm(t)
        if len(key) >= 5:
            lookup.setdefault(key, i)

    # 어떤 (src,dst)에 어떤 술어가 붙는지 모은다. 한 문단이 같은 제목을 여러 번
    # 언급할 수 있으므로 쌍마다 후보가 여럿 나온다.
    found: dict[tuple[int, int], list[tuple[str, int]]] = {}
    need_src = {s for s, _, _ in edges}

    # ner만 끈다. lemmatizer는 술어 표제어에 필요하고, tagger/parser도 마찬가지다.
    docs = nlp.pipe((texts[i] for i in sorted(need_src)),
                    batch_size=batch_size, disable=["ner"])
    for src, doc in zip(sorted(need_src), docs):
        toks = [_NORM.sub(" ", t.text.lower()).strip() for t in doc]
        for n in range(1, MAX_TITLE_WORDS + 1):
            for a in range(len(toks) - n + 1):
                key = " ".join(x for x in toks[a:a + n] if x)
                dst = lookup.get(key)
                if dst is None or dst == src:
                    continue
                pred, pol = _predicate_and_polarity(doc[a:a + n])
                if pred:
                    found.setdefault((src, dst), []).append((pred, pol))

    # 빈도 상위 MAX_TYPES개만 고유 타입을 준다. 나머지는 -1(무타입).
    freq = Counter(p for cands in found.values() for p, _ in cands)
    vocab = {p: i for i, (p, _) in enumerate(freq.most_common(MAX_TYPES))}

    typed, n_typed, n_neg = [], 0, 0
    for s, d, w in edges:
        cands = found.get((s, d))
        if not cands:
            typed.append((s, d, w, -1, 1))       # 파싱 실패 → 통제군과 동일 동작
            continue
        pred, pol = cands[0]                      # 첫 언급을 대표로 쓴다
        tid = vocab.get(pred, -1)
        typed.append((s, d, w, tid, pol))
        n_typed += tid >= 0
        n_neg += pol < 0

    stats = {"n_edges": len(edges), "n_typed": n_typed, "n_negative": n_neg,
             "n_vocab": len(vocab), "coverage": n_typed / max(len(edges), 1),
             "top_predicates": freq.most_common(12)}
    return typed, vocab, stats


def question_types(question: str, vocab: dict, nlp) -> list[int]:
    """질문에서 요구되는 술어 타입을 뽑는다.

    질문의 동사 표제어를 전부 모은다. 어휘에 없는 동사는 버린다.
    **빈 리스트를 반환하면 확산이 제약 없이 돌아간다** — 즉 통제군과 같아진다.
    질문 파싱에 실패해도 성능이 떨어지지 않고 원래대로 돌아갈 뿐이다.
    """
    doc = nlp(question)
    out = []
    for t in doc:
        if t.pos_ in ("VERB", "AUX"):
            tid = vocab.get(t.lemma_.lower())
            if tid is not None and tid not in out:
                out.append(tid)
    return out
