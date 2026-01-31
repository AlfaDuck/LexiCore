#include <iostream>
#include <string>
#include <vector>

#include "io/file_reader.hpp"
#include "preprocess/preprocessing.hpp"
#include "vectorize/bow.hpp"
#include "vectorize/TF-IDF.hpp"
#include "similarity/cosine.hpp"
#include "search/similarity_search.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage:\n";
        std::cout << "  lexicore_app <file1> <file2> ...\n";
        return 1;
    }

    std::vector<std::string> paths;
    paths.reserve(argc - 1);
    for (int i = 1; i < argc; ++i) {
        paths.emplace_back(argv[i]);
    }

    auto files = LexiCore::io::read_files(paths);
    if (files.empty()) {
        std::cerr << "No valid files could be read.\n";
        return 1;
    }

    // -------- preprocess once --------
    auto tokens = LexiCore::preprocess::preprocess(files);

    // -------- vectorize once --------
    auto bows   = LexiCore::vectorize::bag_of_word(tokens);
    auto tfidfs = LexiCore::vectorize::tf_idf(tokens);

    // -------- pairwise similarity (BoW) --------
    std::cout << "Pairwise cosine similarity (BoW):\n";
    for (size_t i = 0; i < bows.size(); ++i) {
        for (size_t j = i + 1; j < bows.size(); ++j) {
            float s = LexiCore::similarity::cosine_similarity(bows[i], bows[j]);
            std::cout << "  [" << i << "] vs [" << j << "] = " << s << "\n";
        }
    }

    // -------- pairwise similarity (TF-IDF) --------
    std::cout << "\nPairwise cosine similarity (TF-IDF):\n";
    for (size_t i = 0; i < tfidfs.size(); ++i) {
        for (size_t j = i + 1; j < tfidfs.size(); ++j) {
            float s = LexiCore::similarity::cosine_similarity(tfidfs[i], tfidfs[j]);
            std::cout << "  [" << i << "] vs [" << j << "] = " << s << "\n";
        }
    }

    // -------- interactive query --------
    std::cout << "\nEnter a query sentence (empty to exit):\n> ";
    std::string query;
    std::getline(std::cin, query);

    if (!query.empty()) {
        // preprocess + vectorize query once
        auto q_tokens = LexiCore::preprocess::preprocess(query);
        auto q_bow    = LexiCore::vectorize::bag_of_word(q_tokens);
        auto q_tfidf  = LexiCore::vectorize::tf_idf(q_tokens);

        auto [idx_bow, score_bow] =
            LexiCore::search::similarity_search_bow_vec(bows, q_bow);

        auto [idx_tfidf, score_tfidf] =
            LexiCore::search::similarity_search_tfidf_vec(tfidfs, q_tfidf);

        std::cout << "\nResults:\n";

        std::cout << "BoW   -> ";
        if (idx_bow >= 0) {
            std::cout << "best file[" << idx_bow << "] = " << paths[idx_bow]
                      << " | score = " << score_bow << "\n";
        } else {
            std::cout << "no match | score = " << score_bow << "\n";
        }

        std::cout << "TF-IDF -> ";
        if (idx_tfidf >= 0) {
            std::cout << "best file[" << idx_tfidf << "] = " << paths[idx_tfidf]
                      << " | score = " << score_tfidf << "\n";
        } else {
            std::cout << "no match | score = " << score_tfidf << "\n";
        }
    }

    return 0;
}
