/* 그래프 검색층 — 진단에서 나온 설계 원리를 그대로 구현한다.
 *
 * NeuroDB v0.1의 문제는 벡터 채널이 그래프 채널을 압도한 것이었다.
 * 32노드 중 32개가 depth 0으로 나와 엣지를 탈 일이 없었고,
 * 그래서 헤비안 강화가 순위에 영향을 주지 못했다.
 *
 * 해마 색인 이론의 두 힘을 명시적으로 분리해 조절 가능하게 둔다:
 *   패턴 분리  seed를 **희소하게** 잡는다. 전부 seed면 분리가 0이다
 *   패턴 완성  그 소수 seed에서 엣지를 타고 퍼진다. 이게 그래프가 일하는 자리다
 *
 * 명제층(PROPOSITION.md)을 위해 엣지에 **술어 타입과 극성**을 실었다.
 * 호환성 규칙 — 타입이 -1이거나 질의 술어 목록이 비면 모든 엣지가 그대로 통과한다.
 * 즉 명제층을 끄면 확산 결과가 이전 코드와 **비트 단위로 같다.** 통제군이 흔들리면
 * 비교가 성립하지 않으므로 이 성질은 반드시 지킨다.
 */
#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>
#include "bm25.h"   /* ScoredDoc */

struct EdgeAttr {
    float   w    = 0.0f;
    int16_t type = -1;   /* 술어 타입 id. -1이면 무타입 = 모든 술어와 호환 */
    int8_t  pol  = 1;    /* 극성. +1 긍정, -1 부정("A는 B가 아니다") */
};

class Graph {
public:
    explicit Graph(int32_t n_nodes) : n_(n_nodes), adj_(n_nodes) {}

    /* 무타입 엣지. 기존 호출부가 그대로 동작한다. */
    void add_edge(int32_t src, int32_t dst, float w);

    /* 명제 엣지 — 술어 타입과 극성을 실어 넣는다. */
    void add_edge_typed(int32_t src, int32_t dst, float w, int16_t type, int8_t pol);

    /* 희소한 seed에서 활성을 퍼뜨린다.
     * depth_decay:  한 홉 건널 때마다 곱해지는 감쇠
     * min_act:      이보다 작아지면 전파를 멈춘다
     * query_types:  질문에서 뽑은 술어 타입. **비어 있으면 타입 제약이 없다**
     * alpha:        질문 술어와 맞지 않는 엣지에 곱하는 감쇠 (하드 차단이 아니다)
     * beta:         극성이 음인 엣지에 곱하는 감쇠
     *
     * 하드 프루닝을 쓰지 않는 이유: 파싱은 취약하다. 한 번 잘못 뽑으면
     * 정답 경로가 통째로 막힌다. 약하게 누르는 편이 실패에 견딘다. */
    std::vector<ScoredDoc> spread(const std::vector<int32_t>&  seeds,
                                  const std::vector<float>&    seed_acts,
                                  int   max_depth,
                                  float depth_decay,
                                  float min_act,
                                  const std::vector<int16_t>& query_types = {},
                                  float alpha = 1.0f,
                                  float beta  = 1.0f) const;

    /* 함께 활성화된 노드 쌍의 엣지를 강화한다. 없으면 만든다.
     * "함께 발화하는 뉴런은 함께 연결된다" — 이게 read가 DB를 바꾸는 지점이다.
     * H1에서 측정 결과 표준 멀티홉 검색에서는 이득 상한이 0이었다(RESULTS.md).
     * 기전 확인용으로 남겨둔다. */
    void reinforce(const std::vector<int32_t>& coactive, float delta, float w_max);

    /* 시간 감쇠. 쓰이지 않는 연결은 약해진다. */
    void decay(float lambda);

    int32_t n_nodes() const { return n_; }
    int64_t n_edges() const;
    float   edge_weight(int32_t src, int32_t dst) const;

private:
    int32_t n_;
    /* 노드별 이웃 목록. 밀도가 낮아 인접 리스트가 맞다. */
    std::vector<std::unordered_map<int32_t, EdgeAttr>> adj_;
};
