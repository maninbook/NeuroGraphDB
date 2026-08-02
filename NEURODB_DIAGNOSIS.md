# NeuroDB v0.1 실측 진단 (2026-08-01)

`business/BrainDocs/neuro_db_c`. 빌드돼 있고 파이썬 바인딩 동작(python3.13).
API: absorb / connect / think / infer / pulse / contradict / traverse.

**코드는 제대로 연결돼 있다.** `brain.cpp:243`에서 `think()`가 공동활성 상위 8개 쌍에
`hebbian_reinforce(delta=0.03)`를 호출한다. 설계 의도대로다.

**그런데 그 강화가 검색 결과를 바꾸지 못한다.** 이유가 셋이고 서로 맞물려 있다.

## 1. 임베딩에 의미가 없다

n-gram + 랜덤 프로젝션. 완전히 무관한 문장들의 벡터 점수가 좁은 띠에 몰린다.

```
질의: "quantum physics experiment"
  0.7466  Hippocampus indexes episodic memory      ← 1위
  0.7106  The Eiffel Tower is in Paris
  0.6533  REM sleep consolidates declarative memory
  0.6122  Coffee is a popular beverage worldwide
  0.6114  Quantum entanglement violates local realism  ← 정답이 꼴찌
```

정답을 맨 아래에 놓는다. 표면 문자 패턴만 보고 의미를 못 본다.
(ROADMAP Phase 2-1이 이미 skip-gram 교체를 계획하고 있으나, 문제는
"수면 vs sleep" 수준이 아니라 **정답이 역순으로 나오는** 수준이다.)

## 2. LSH가 전부를 seed로 잡아 spreading이 일어나지 않는다

노드 32개 중 **32개 전부가 depth 0**으로 반환된다. 직접 벡터 히트가 전부이므로
엣지를 타고 퍼질 여지가 없다. 그래프가 계산에 관여하지 않는다.

## 3. 그래서 엣지로만 닿는 노드는 절대 올라오지 못한다

seed에 가중치 0.95 엣지로 직결한 target을 101회 반복 조회해도 top-10에 한 번도 못 든다.
depth 1이면 depth_decay 0.65가 곱해지는데, 무관한 filler들의 벡터 점수가 0.89라
0.65×seed로는 이길 수가 없다.

**결론: 벡터 채널이 그래프 채널을 완전히 압도한다. 헤비안은 작동하지만 영향력이 0이다.**

## 이 진단이 NeuroGraphDB의 설계를 정한다

조사에서 찾은 빈틈 (b)와 정확히 이어진다 — **패턴 분리 vs 패턴 완성의 길항**.

- **패턴 분리(pattern separation)**: 비슷한 기억을 구별되게 저장 → 변별력 있는 임베딩,
  그리고 **seed를 적게** 잡기. 지금은 전부 seed라 분리가 0이다
- **패턴 완성(pattern completion)**: 부분 단서로 전체 복원 → 엣지 순회가 실제로 일어나야 한다.
  지금은 depth 0만 나오므로 완성도 0이다

HippoRAG가 잘 되는 이유가 여기 있다. 엔티티 몇 개만 seed로 잡고 **나머지는 PPR이 한다.**
NeuroDB는 반대로 벡터가 다 하고 그래프는 장식이다.

## 그래서 할 일

1. **임베딩 교체** — 의미 변별이 되는 것으로. 이게 없으면 나머지가 다 무의미
2. **seed를 희소하게** — 상위 소수만 seed로. 그래야 spreading이 일할 자리가 생긴다
3. **그 다음에야** 헤비안 강화가 측정 가능한 효과를 낸다

1·2를 고치기 전에 헤비안 효과를 측정하면 반드시 "효과 없음"이 나온다.
순서를 지켜야 한다.
