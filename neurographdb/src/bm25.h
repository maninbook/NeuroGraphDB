/* BM25 어휘 검색 — baseline 하한.
 * 이걸 못 이기면 그래프도 임베딩도 얘기할 게 없다. */
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

struct ScoredDoc { int32_t doc; float score; };

class BM25 {
public:
    explicit BM25(float k1 = 1.2f, float b = 0.75f) : k1_(k1), b_(b) {}

    /* 문서를 색인한다. 반환값은 내부 문서 번호. */
    int32_t add(const std::string& text);

    /* 색인을 확정한다. IDF와 평균 문서길이를 여기서 계산한다. */
    void finalize();

    /* 상위 k개. finalize() 후에만 유효. */
    std::vector<ScoredDoc> search(const std::string& query, int k) const;

    int32_t size() const { return n_docs_; }

private:
    float k1_, b_;
    int32_t n_docs_ = 0;
    double  total_len_ = 0.0;
    float   avgdl_ = 0.0f;
    std::vector<int32_t> doc_len_;
    /* term → [(doc, tf), ...] */
    std::unordered_map<std::string, std::vector<std::pair<int32_t,int32_t>>> postings_;
    std::unordered_map<std::string, float> idf_;
    bool finalized_ = false;
};
