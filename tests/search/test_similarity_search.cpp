#include <cassert>
#include <string>
#include <vector>

#include "search/similarity_search.hpp"

static void test_similarity_search_bow_picks_best_doc() {
    std::vector<std::vector<std::string>> files = {
        {"apple", "apple", "banana", "orange", "fruit", "common"},
        {"car", "bus", "train", "vehicle", "common"},
        {"apple", "car", "mixed", "common"}
    };

    auto r = LexiCore::search::similarity_search_bow(files, "apple banana fruit");
    assert(r.first == 0);
    assert(r.second > 0.0f);
}

static void test_similarity_search_bow_handles_no_match() {
    std::vector<std::vector<std::string>> files = {
        {"apple", "banana", "fruit", "common"},
        {"car", "bus", "train", "vehicle", "common"}
    };

    auto r = LexiCore::search::similarity_search_bow(files, "zzzz qqqq");
    assert(r.first == -1);
    assert(r.second >= 0.0f && r.second < 1e-6f);
}

static void test_similarity_search_tfidf_picks_best_doc() {
    std::vector<std::vector<std::string>> files = {
        {"apple", "apple", "banana", "orange", "fruit", "common"},
        {"car", "bus", "train", "vehicle", "common"},
        {"apple", "car", "mixed", "common"}
    };

    auto r = LexiCore::search::similarity_search_tfidf(files, "apple banana fruit");
    assert(r.first == 0);
    assert(r.second > 0.0f);
}

static void test_similarity_search_tfidf_handles_no_match() {
    std::vector<std::vector<std::string>> files = {
        {"apple", "banana", "fruit", "common"},
        {"car", "bus", "train", "vehicle", "common"}
    };

    auto r = LexiCore::search::similarity_search_tfidf(files, "zzzz qqqq");
    assert(r.first == -1);
    assert(r.second >= 0.0f && r.second < 1e-6f);
}

int main() {
    test_similarity_search_bow_picks_best_doc();
    test_similarity_search_bow_handles_no_match();
    test_similarity_search_tfidf_picks_best_doc();
    test_similarity_search_tfidf_handles_no_match();
    return 0;
}
