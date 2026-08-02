/* 조밀 벡터 색인 — 사람들이 실제로 쓰는 baseline.
 * 완전탐색 코사인. HotpotQA는 질문당 문단 10개라 근사가 필요 없고,
 * 근사를 넣으면 baseline 성능이 근사 오차만큼 깎여 비교가 불공정해진다. */
#pragma once
#include <vector>
#include <cstdint>
#include "bm25.h"   /* ScoredDoc */

class DenseIndex {
public:
    explicit DenseIndex(int dim) : dim_(dim) {}

    /* 벡터를 넣는다. 내부에서 L2 정규화하므로 코사인 = 내적이 된다. */
    int32_t add(const float* v);

    std::vector<ScoredDoc> search(const float* q, int k) const;

    int32_t size() const { return n_; }
    int dim() const { return dim_; }

private:
    int dim_;
    int32_t n_ = 0;
    std::vector<float> data_;   /* n_ * dim_ 연속 배치 */
};
