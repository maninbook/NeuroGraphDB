#include "bm25.h"
#include "tokenize.h"
#include <algorithm>
#include <cmath>

int32_t BM25::add(const std::string& text) {
    const int32_t id = n_docs_++;
    auto toks = tokenize(text);
    doc_len_.push_back(static_cast<int32_t>(toks.size()));
    total_len_ += toks.size();

    /* 문서 내 빈도를 먼저 모아서 포스팅에 한 번만 넣는다 */
    std::unordered_map<std::string,int32_t> tf;
    tf.reserve(toks.size());
    for (auto& t : toks) tf[t]++;
    for (auto& [term, f] : tf) postings_[term].emplace_back(id, f);
    finalized_ = false;
    return id;
}

void BM25::finalize() {
    avgdl_ = n_docs_ ? static_cast<float>(total_len_ / n_docs_) : 0.0f;
    idf_.clear();
    idf_.reserve(postings_.size());
    for (auto& [term, plist] : postings_) {
        const double nq = static_cast<double>(plist.size());
        /* Robertson-Sparck Jones IDF, 음수 방지를 위해 +1 */
        idf_[term] = static_cast<float>(
            std::log((n_docs_ - nq + 0.5) / (nq + 0.5) + 1.0));
    }
    finalized_ = true;
}

std::vector<ScoredDoc> BM25::search(const std::string& query, int k) const {
    std::vector<ScoredDoc> out;
    if (!finalized_ || n_docs_ == 0) return out;

    std::vector<float> acc(n_docs_, 0.0f);
    for (auto& term : tokenize(query)) {
        auto it = postings_.find(term);
        if (it == postings_.end()) continue;
        const float idf = idf_.at(term);
        for (auto& [doc, f] : it->second) {
            const float denom = f + k1_ * (1.0f - b_ + b_ * doc_len_[doc] / avgdl_);
            acc[doc] += idf * (f * (k1_ + 1.0f)) / denom;
        }
    }
    out.reserve(n_docs_);
    for (int32_t i = 0; i < n_docs_; i++)
        if (acc[i] > 0.0f) out.push_back({i, acc[i]});

    const int kk = std::min<int>(k, static_cast<int>(out.size()));
    std::partial_sort(out.begin(), out.begin() + kk, out.end(),
                      [](const ScoredDoc& a, const ScoredDoc& b){ return a.score > b.score; });
    out.resize(kk);
    return out;
}
