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

static bool contains_term(const std::vector<std::pair<std::string, float>>& v,
                          const std::string& term) {
    for (const auto& kv : v) {
        if (kv.first == term) return true;
    }
    return false;
}

static void assert_normalized_if_nonempty(const std::vector<std::pair<std::string, float>>& v) {
    if (v.empty()) return; // empty vec is fine (e.g. OOV-only or before fit)
    float n = l2_norm(v);
    assert(std::fabs(n - 1.0f) < 1e-4f);
}

static void test_transform_before_fit_returns_empty() {
    LexiCore::vectorize::TF_IDF model;

    std::vector<std::string> doc = {"a", "b", "c"};
    auto v = model.transform(doc);
    assert(v.empty());

    std::vector<std::vector<std::string>> docs = {{"a"}, {"b"}};
    auto vv = model.transform(docs);
    assert(vv.empty());
}

static void test_tfidf_multi_docs_normalized_and_similarity() {
    std::vector<std::vector<std::string>> docs = {
        {"common", "common", "apple", "banana", "unique1"},
        {"common", "apple", "banana", "banana", "unique2"},
        {"common", "car", "bus", "train", "unique3"}
    };

    LexiCore::vectorize::TF_IDF model;

    // fit on corpus
    model.fit(docs);

    // transform corpus in the same space
    auto tfidf = model.transform(docs);

    assert(tfidf.size() == docs.size());
    for (const auto& doc_vec : tfidf) {
        assert_normalized_if_nonempty(doc_vec);
    }

    // sanity: important terms exist
    auto m0 = to_map(tfidf[0]);
    assert(m0.count("unique1") == 1);
    assert(m0.count("common") == 1);

    // common appears in all docs -> should be downweighted vs unique term (not guaranteed in every corpus,
    // but with this setup it should hold)
    assert(std::fabs(m0["unique1"]) > std::fabs(m0["common"]));

    // similarity sanity
    float sim01 = dot(tfidf[0], tfidf[1]); // share apple/banana/common
    float sim02 = dot(tfidf[0], tfidf[2]); // mostly only common
    assert(sim01 > sim02);
}

static void test_oov_terms_are_ignored_in_transform() {
    std::vector<std::vector<std::string>> docs = {
        {"apple", "banana"},
        {"banana", "car"}
    };

    LexiCore::vectorize::TF_IDF model;
    model.fit(docs);

    // New doc contains OOV term "zzz"
    std::vector<std::string> new_doc = {"apple", "zzz", "zzz"};
    auto v = model.transform(new_doc);

    // "zzz" should not appear
    assert(!contains_term(v, "zzz"));
    // "apple" should appear
    assert(contains_term(v, "apple"));

    assert_normalized_if_nonempty(v);
}

static void test_fit_transform_is_equivalent_to_fit_plus_transform() {
    std::vector<std::vector<std::string>> docs = {
        {"a", "a", "b"},
        {"a", "c"}
    };

    LexiCore::vectorize::TF_IDF m1;
    auto ft = m1.fit_transform(docs);

    LexiCore::vectorize::TF_IDF m2;
    m2.fit(docs);
    auto tt = m2.transform(docs);

    assert(ft.size() == tt.size());

    // compare as maps (order-independent)
    for (size_t i = 0; i < ft.size(); ++i) {
        auto mft = to_map(ft[i]);
        auto mtt = to_map(tt[i]);
        assert(mft.size() == mtt.size());
        for (const auto& kv : mft) {
            auto it = mtt.find(kv.first);
            assert(it != mtt.end());
            assert(std::fabs(it->second - kv.second) < 1e-6f);
        }
    }
}

static void test_reset_clears_model_state() {
    std::vector<std::vector<std::string>> docs = {
        {"apple", "banana"},
        {"banana", "car"}
    };

    LexiCore::vectorize::TF_IDF model;
    model.fit(docs);

    // after fit, transform should be non-empty for in-vocab doc
    auto v1 = model.transform(std::vector<std::string>{"apple"});
    assert(!v1.empty());

    // reset and then transform should return empty (unfitted)
    model.reset();
    auto v2 = model.transform(std::vector<std::string>{"apple"});
    assert(v2.empty());

    // re-fit should work again
    model.fit(docs);
    auto v3 = model.transform(std::vector<std::string>{"apple"});
    assert(!v3.empty());
    assert_normalized_if_nonempty(v3);
}

static void test_single_doc_api_normalized() {
    // this uses fit(single)+transform(single) path indirectly via fit_transform(single)
    std::vector<std::string> doc = {"apple", "apple", "banana", "common"};

    LexiCore::vectorize::TF_IDF model;
    auto vec = model.fit_transform(doc);

    assert_normalized_if_nonempty(vec);

    auto m = to_map(vec);
    assert(m.count("apple") == 1);
    assert(m.count("banana") == 1);
    assert(m.count("common") == 1);
}

int main() {
    test_transform_before_fit_returns_empty();
    test_tfidf_multi_docs_normalized_and_similarity();
    test_oov_terms_are_ignored_in_transform();
    test_fit_transform_is_equivalent_to_fit_plus_transform();
    test_reset_clears_model_state();
    test_single_doc_api_normalized();
    return 0;
}