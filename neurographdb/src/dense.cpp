#include "dense.h"
#include <algorithm>
#include <cmath>

int32_t DenseIndex::add(const float* v) {
    double norm = 0.0;
    for (int i = 0; i < dim_; i++) norm += static_cast<double>(v[i]) * v[i];
    norm = std::sqrt(norm);
    const float inv = (norm > 1e-12) ? static_cast<float>(1.0 / norm) : 0.0f;
    data_.reserve(data_.size() + dim_);
    for (int i = 0; i < dim_; i++) data_.push_back(v[i] * inv);
    return n_++;
}

std::vector<ScoredDoc> DenseIndex::search(const float* q, int k) const {
    std::vector<ScoredDoc> out;
    if (n_ == 0) return out;

    double qn = 0.0;
    for (int i = 0; i < dim_; i++) qn += static_cast<double>(q[i]) * q[i];
    qn = std::sqrt(qn);
    const float qinv = (qn > 1e-12) ? static_cast<float>(1.0 / qn) : 0.0f;

    out.resize(n_);
    for (int32_t d = 0; d < n_; d++) {
        const float* row = data_.data() + static_cast<size_t>(d) * dim_;
        float dot = 0.0f;
        for (int i = 0; i < dim_; i++) dot += row[i] * q[i];
        out[d] = {d, dot * qinv};
    }
    const int kk = std::min<int>(k, n_);
    std::partial_sort(out.begin(), out.begin() + kk, out.end(),
                      [](const ScoredDoc& a, const ScoredDoc& b){ return a.score > b.score; });
    out.resize(kk);
    return out;
}
