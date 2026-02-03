#include "search/similarity_search.hpp"

#include "preprocess/preprocessing.hpp"
#include "similarity/cosine.hpp"
#include "vectorize/bow.hpp"
#include "vectorize/TF-IDF.hpp"

namespace LexiCore::search {

std::pair<int, float> similarity_search_bow(
    const std::vector<std::vector<std::string>>& tokenized_files,
    const std::string& input) {

    auto input_tokens = LexiCore::preprocess::preprocess(input);
    auto input_vec    = LexiCore::vectorize::bag_of_word(input_tokens);
    auto files_vecs   = LexiCore::vectorize::bag_of_word(tokenized_files);

    float best_score = 0.0f;
    int best_index   = -1;

    for (int i = 0; i < static_cast<int>(files_vecs.size()); ++i) {
        float s = LexiCore::similarity::cosine_similarity(files_vecs[i], input_vec);
        if (s > best_score) {
            best_score = s;
            best_index = i;
        }
    }

    return {best_index, best_score};
}

std::pair<int, float> similarity_search_tfidf(
    const std::vector<std::vector<std::string>>& tokenized_files,
    const std::string& input) {

    auto input_tokens = LexiCore::preprocess::preprocess(input);

    LexiCore::vectorize::TF_IDF model;
    model.fit(tokenized_files);

    auto files_vecs = model.transform(tokenized_files);
    auto input_vec  = model.transform(input_tokens);

    float best_score = 0.0f;
    int best_index   = -1;

    for (int i = 0; i < static_cast<int>(files_vecs.size()); ++i) {
        float s = LexiCore::similarity::cosine_similarity(files_vecs[i], input_vec);
        if (s > best_score) {
            best_score = s;
            best_index = i;
        }
    }

    return {best_index, best_score};
}

std::pair<int, float> similarity_search_bow_vec(
    const std::vector<std::vector<std::pair<std::string, int>>>& bow_files,
    const std::vector<std::pair<std::string, int>>& bow_query) {

    float best_score = 0.0f;
    int best_index   = -1;

    for (int i = 0; i < static_cast<int>(bow_files.size()); ++i) {
        float s = LexiCore::similarity::cosine_similarity(bow_files[i], bow_query);
        if (s > best_score) {
            best_score = s;
            best_index = i;
        }
    }

    return {best_index, best_score};
}

std::pair<int, float> similarity_search_tfidf_vec(
    const std::vector<std::vector<std::pair<std::string, float>>>& tfidf_files,
    const std::vector<std::pair<std::string, float>>& tfidf_query) {

    float best_score = 0.0f;
    int best_index   = -1;

    for (int i = 0; i < static_cast<int>(tfidf_files.size()); ++i) {
        float s = LexiCore::similarity::cosine_similarity(tfidf_files[i], tfidf_query);
        if (s > best_score) {
            best_score = s;
            best_index = i;
        }
    }

    return {best_index, best_score};
}

}