#include <cassert>
#include <string>
#include <vector>

#include "preprocess/preprocessing.hpp"
#include "vectorize/bow.hpp"
#include "vectorize/TF-IDF.hpp"
#include "search/similarity_search.hpp"


static void test_similarity_search_bow_vec_picks_best_doc() {
    std::vector<std::vector<std::string>> tokens = {
        {"apple", "apple", "banana", "orange", "fruit", "common"},
        {"car", "bus", "train", "vehicle", "common"},
        {"apple", "car", "mixed", "common"}
    };

    auto bows = LexiCore::vectorize::bag_of_word(tokens);

    std::vector<std::string> q_tokens = {"apple", "banana", "fruit"};
    auto q_bow = LexiCore::vectorize::bag_of_word(q_tokens);

    auto r = LexiCore::search::similarity_search_bow_vec(bows, q_bow);
    assert(r.first == 0);
    assert(r.second > 0.0f);
}

static void test_similarity_search_bow_vec_handles_no_match() {
    std::vector<std::vector<std::string>> tokens = {
        {"apple", "banana", "fruit", "common"},
        {"car", "bus", "train", "vehicle", "common"}
    };

    auto bows = LexiCore::vectorize::bag_of_word(tokens);

    std::vector<std::string> q_tokens = {"zzzz", "qqqq"};
    auto q_bow = LexiCore::vectorize::bag_of_word(q_tokens);

    auto r = LexiCore::search::similarity_search_bow_vec(bows, q_bow);
    assert(r.first == -1);
    assert(r.second >= 0.0f && r.second < 1e-6f);
}


static void test_similarity_search_tfidf_vec_picks_best_doc() {
    std::vector<std::vector<std::string>> tokens = {
        {"apple", "apple", "banana", "orange", "fruit", "common"},
        {"car", "bus", "train", "vehicle", "common"},
        {"apple", "car", "mixed", "common"}
    };

    auto tfidfs = LexiCore::vectorize::tf_idf(tokens);

    std::vector<std::string> q_tokens = {"apple", "banana", "fruit"};
    auto q_tfidf = LexiCore::vectorize::tf_idf(q_tokens);

    auto r = LexiCore::search::similarity_search_tfidf_vec(tfidfs, q_tfidf);
    assert(r.first == 0);
    assert(r.second > 0.0f);
}

static void test_similarity_search_tfidf_vec_handles_no_match() {
    std::vector<std::vector<std::string>> tokens = {
        {"apple", "banana", "fruit", "common"},
        {"car", "bus", "train", "vehicle", "common"}
    };

    auto tfidfs = LexiCore::vectorize::tf_idf(tokens);

    std::vector<std::string> q_tokens = {"zzzz", "qqqq"};
    auto q_tfidf = LexiCore::vectorize::tf_idf(q_tokens);

    auto r = LexiCore::search::similarity_search_tfidf_vec(tfidfs, q_tfidf);
    assert(r.first == -1);
    assert(r.second >= 0.0f && r.second < 1e-6f);
}

int main() {
    test_similarity_search_bow_vec_picks_best_doc();
    test_similarity_search_bow_vec_handles_no_match();
    test_similarity_search_tfidf_vec_picks_best_doc();
    test_similarity_search_tfidf_vec_handles_no_match();
    return 0;
}
