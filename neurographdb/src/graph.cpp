#include "graph.h"
#include <algorithm>
#include <queue>

void Graph::add_edge(int32_t src, int32_t dst, float w) {
    add_edge_typed(src, dst, w, -1, 1);
}

void Graph::add_edge_typed(int32_t src, int32_t dst, float w, int16_t type, int8_t pol) {
    if (src < 0 || src >= n_ || dst < 0 || dst >= n_ || src == dst) return;
    auto& e = adj_[src][dst];
    /* 같은 쌍이 여러 번 오면 가장 강한 것을 남긴다.
     * 타입은 그때 함께 갱신한다 — 가장 강한 근거의 술어를 대표로 쓴다. */
    if (w > e.w) { e.w = w; e.type = type; e.pol = pol; }
}

int64_t Graph::n_edges() const {
    int64_t c = 0;
    for (auto& m : adj_) c += static_cast<int64_t>(m.size());
    return c;
}

float Graph::edge_weight(int32_t src, int32_t dst) const {
    if (src < 0 || src >= n_) return 0.0f;
    auto it = adj_[src].find(dst);
    return it == adj_[src].end() ? 0.0f : it->second.w;
}

std::vector<ScoredDoc> Graph::spread(const std::vector<int32_t>& seeds,
                                     const std::vector<float>&   seed_acts,
                                     int max_depth,
                                     float depth_decay,
                                     float min_act,
                                     const std::vector<int16_t>& query_types,
                                     float alpha,
                                     float beta) const {
    std::vector<float> act(n_, 0.0f);
    std::vector<int>   depth(n_, -1);

    const bool typed = !query_types.empty();

    /* 너비 우선. 같은 노드에 여러 경로로 도달하면 가장 높은 활성을 남긴다. */
    std::queue<int32_t> frontier;
    for (size_t i = 0; i < seeds.size(); i++) {
        const int32_t s = seeds[i];
        if (s < 0 || s >= n_) continue;
        const float a = (i < seed_acts.size()) ? seed_acts[i] : 1.0f;
        if (a > act[s]) { act[s] = a; depth[s] = 0; frontier.push(s); }
    }

    while (!frontier.empty()) {
        const int32_t u = frontier.front();
        frontier.pop();
        if (depth[u] >= max_depth) continue;
        for (auto& [v, e] : adj_[u]) {
            /* 술어 호환 — 무타입(-1)은 언제나 통과한다. 질의 술어가 없어도 통과한다.
             * 그래서 명제층을 끄면 아래 곱셈이 전부 1.0이 되어 예전 코드와 같아진다. */
            float f = 1.0f;
            if (typed && e.type >= 0 &&
                std::find(query_types.begin(), query_types.end(), e.type) == query_types.end())
                f = alpha;
            if (e.pol < 0) f *= beta;

            const float cand = act[u] * e.w * f * depth_decay;
            if (cand < min_act || cand <= act[v]) continue;
            act[v] = cand;
            depth[v] = depth[u] + 1;
            frontier.push(v);
        }
    }

    std::vector<ScoredDoc> out;
    out.reserve(64);
    for (int32_t i = 0; i < n_; i++)
        if (act[i] > 0.0f) out.push_back({i, act[i]});
    std::sort(out.begin(), out.end(),
              [](const ScoredDoc& a, const ScoredDoc& b){ return a.score > b.score; });
    return out;
}

void Graph::reinforce(const std::vector<int32_t>& coactive, float delta, float w_max) {
    for (size_t i = 0; i < coactive.size(); i++) {
        for (size_t j = i + 1; j < coactive.size(); j++) {
            const int32_t a = coactive[i], b = coactive[j];
            if (a < 0 || a >= n_ || b < 0 || b >= n_ || a == b) continue;
            /* 양방향으로 강화한다. 연상은 방향이 없다. */
            EdgeAttr& ab = adj_[a][b];
            EdgeAttr& ba = adj_[b][a];
            ab.w = std::min(w_max, ab.w + delta);
            ba.w = std::min(w_max, ba.w + delta);
        }
    }
}

void Graph::decay(float lambda) {
    const float f = 1.0f - lambda;
    for (auto& m : adj_)
        for (auto& [v, e] : m) e.w *= f;
}
