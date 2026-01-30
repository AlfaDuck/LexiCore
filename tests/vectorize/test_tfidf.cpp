#include <cassert>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "vectorize/TF-IDF.hpp"

static std::unordered_map<std::string, float>
to_map(const std::vector<std::pair<std::string, float>>& v) {
    std::unordered_map<std::string, float> m;
    m.reserve(v.size());
    for (const auto& kv : v) m[kv.first] = kv.second;
    return m;
}

static float l2_norm(const std::vector<std::pair<std::string, float>>& v) {
    float s = 0.0f;
    for (const auto& kv : v) s += kv.second * kv.second;
    return std::sqrt(s);
}

static float dot(const std::vector<std::pair<std::string, float>>& a,
                 const std::vector<std::pair<std::string, float>>& b) {
    auto am = to_map(a);
    float s = 0.0f;
    for (const auto& kv : b) {
        auto it = am.find(kv.first);
        if (it != am.end()) s += it->second * kv.second;
    }
    return s;
}

static void test_tfidf_multi_docs_normalized_and_similarity() {
    std::vector<std::vector<std::string>> docs = {
        {"common", "common", "apple", "banana", "unique1"},
        {"common", "apple", "banana", "banana", "unique2"},
        {"common", "car", "bus", "train", "unique3"}
    };

    // Act
    auto tfidf = LexiCore::vectorize::tf_idf(docs);

    assert(tfidf.size() == docs.size());

    for (const auto& doc_vec : tfidf) {
        float n = l2_norm(doc_vec);
        assert(std::fabs(n - 1.0f) < 1e-4f);
    }

    auto m0 = to_map(tfidf[0]);
    assert(m0.count("unique1") == 1);
    assert(m0.count("common") == 1);
    assert(std::fabs(m0["unique1"]) > std::fabs(m0["common"]));

    float sim01 = dot(tfidf[0], tfidf[1]);
    float sim02 = dot(tfidf[0], tfidf[2]);

    assert(sim01 > sim02);
}

static void test_tfidf_single_doc_normalized() {
    // Arrange
    std::vector<std::string> doc = {"apple", "apple", "banana", "common"};

    // Act
    auto vec = LexiCore::vectorize::tf_idf(doc);

    // Assert: norm ~1
    float n = l2_norm(vec);
    assert(std::fabs(n - 1.0f) < 1e-4f);

    auto m = to_map(vec);
    assert(m.count("apple") == 1);
    assert(m.count("banana") == 1);
    assert(m.count("common") == 1);
}

int main() {
    test_tfidf_multi_docs_normalized_and_similarity();
    test_tfidf_single_doc_normalized();
    return 0;
}
