/* 간단한 영어 토크나이저 — BM25용.
 * 소문자화 + 비영숫자 분리. HotpotQA/MuSiQue 같은 영어 위키 텍스트가 대상이라
 * 이 정도로 충분하다. 형태소 분석은 넣지 않는다 — 넣으면 baseline이 아니라
 * 이미 튜닝된 것이 되어 비교가 불공정해진다. */
#pragma once
#include <string>
#include <vector>
#include <cctype>

inline std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    cur.reserve(24);
    for (unsigned char c : s) {
        if (std::isalnum(c)) {
            cur.push_back(static_cast<char>(std::tolower(c)));
        } else if (!cur.empty()) {
            out.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}
