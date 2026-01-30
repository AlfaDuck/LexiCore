#include "preprocess/preprocessing.hpp"
#include "similarity/cosine.hpp"
#include "vectorize/bow.hpp"
#include "vectorize/TF-IDF.hpp"

namespace LexiCore::search
{
    std::pair<int, float> similarity_search_bow(const std::vector<std::vector<std::string>>& files,
        const std::string& input) {

        auto input_words = LexiCore::preprocess::preprocess(input);
        auto input_words_count = LexiCore::vectorize::bag_of_word(input_words);
        auto files_words = LexiCore::vectorize::bag_of_word(files);

        float best_similarity = 0.0f;
        int best_index = -1;

        for (int i = 0; i < static_cast<int>(files_words.size()); i++) {
            float s = LexiCore::similarity::cosine_similarity(files_words[i], input_words_count);
            if (s > best_similarity) {
                best_similarity = s;
                best_index = i;
            }
        }

        return {best_index, best_similarity};
    }

    std::pair<int, float> similarity_search_tfidf(const std::vector<std::vector<std::string>>& files,
        const std::string& input) {

        auto input_words = LexiCore::preprocess::preprocess(input);
        auto input_words_count = LexiCore::vectorize::tf_idf(input_words);
        auto files_words = LexiCore::vectorize::tf_idf(files);

        float best_similarity = 0.0f;
        int best_index = -1;

        for (int i = 0; i < static_cast<int>(files_words.size()); i++) {
            float s = LexiCore::similarity::cosine_similarity(files_words[i], input_words_count);
            if (s > best_similarity) {
                best_similarity = s;
                best_index = i;
            }
        }

        return {best_index, best_similarity};
    }

    std::pair<int, float> similarity_search_bow_raw(const std::vector<std::vector<std::string>>& raw_files_lines,
        const std::string& input)
    {
        auto tokenized = LexiCore::preprocess::preprocess(raw_files_lines);
        return similarity_search_bow(tokenized, input);
    }

    std::pair<int, float> similarity_search_tfidf_raw(const std::vector<std::vector<std::string>>& raw_files_lines,
        const std::string& input)
    {
        auto tokenized = LexiCore::preprocess::preprocess(raw_files_lines);
        return similarity_search_tfidf(tokenized, input);
    }
}
