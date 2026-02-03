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

    // Build TF-IDF vectors using ONE fitted model
    LexiCore::vectorize::TF_IDF model;
    model.fit(tokens);
    auto tfidfs = model.transform(tokens);

    std::vector<std::string> q_tokens = {"apple", "banana", "fruit"};
    auto q_tfidf = model.transform(q_tokens);

    auto r = LexiCore::search::similarity_search_tfidf_vec(tfidfs, q_tfidf);
    assert(r.first == 0);
    assert(r.second > 0.0f);
}

static void test_similarity_search_tfidf_vec_handles_no_match() {
    std::vector<std::vector<std::string>> tokens = {
        {"apple", "banana", "fruit", "common"},
        {"car", "bus", "train", "vehicle", "common"}
    };

    LexiCore::vectorize::TF_IDF model;
    model.fit(tokens);
    auto tfidfs = model.transform(tokens);

    // Query is only OOV terms -> after transform it becomes empty vector
    std::vector<std::string> q_tokens = {"zzzz", "qqqq"};
    auto q_tfidf = model.transform(q_tokens);

    auto r = LexiCore::search::similarity_search_tfidf_vec(tfidfs, q_tfidf);

    // With an empty query vector, cosine similarity should be 0 for all docs
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