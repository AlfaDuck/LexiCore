#include <cassert>
#include <string>
#include <unordered_map>
#include <vector>

#include "vectorize/bow.hpp"

static std::unordered_map<std::string, int>
to_map(const std::vector<std::pair<std::string, int>>& v) {
    std::unordered_map<std::string, int> m;
    for (const auto& kv : v) m[kv.first] = kv.second;
    return m;
}

static void test_bow_single_document_counts() {
    std::vector<std::string> words = {
        "apple", "banana", "apple", "orange", "banana", "apple"
    };

    auto bow = LexiCore::vectorize::BagOfWords::fit_transform(words);
    auto m = to_map(bow);

    assert(m["apple"] == 3);
    assert(m["banana"] == 2);
    assert(m["orange"] == 1);
}

static void test_bow_multiple_documents() {
    std::vector<std::vector<std::string>> docs = {
        {"apple", "apple", "banana"},
        {"car", "bus", "car", "train"}
    };

    auto bows = LexiCore::vectorize::BagOfWords::fit_transform(docs);

    assert(bows.size() == 2);

    auto m0 = to_map(bows[0]);
    auto m1 = to_map(bows[1]);

    assert(m0["apple"] == 2);
    assert(m0["banana"] == 1);

    assert(m1["car"] == 2);
    assert(m1["bus"] == 1);
    assert(m1["train"] == 1);
}

static void test_bow_top_n() {
    std::vector<std::vector<std::string>> docs = {
        {"a", "b", "a", "c", "a", "b"}
    };

    auto bows = LexiCore::vectorize::BagOfWords::fit_transform(docs, 1);

    assert(bows.size() == 1);
    assert(bows[0].size() == 1);
    assert(bows[0][0].first == "a");
    assert(bows[0][0].second == 3);
}

int main() {
    test_bow_single_document_counts();
    test_bow_multiple_documents();
    test_bow_top_n();
    return 0;
}
