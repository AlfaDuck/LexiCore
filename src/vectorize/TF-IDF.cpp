#include "vectorize/TF-IDF.hpp"

#include <unordered_map>
#include <cmath>

#include "vectorize/bow.hpp"

namespace LexiCore::vectorize {

    std::vector<std::vector<std::pair<std::string, float>>> tf_idf(
        const std::vector<std::vector<std::string>>& files_words)
    {
        std::vector<std::vector<std::pair<std::string, int>>> bow = LexiCore::vectorize::bag_of_word(files_words);
        std::vector<std::vector<std::pair<std::string, float>>> tf;

        for (const auto& pairs : bow) {
            std::vector<std::pair<std::string, float>> words;
            for (const auto& pair : pairs) {
                if (pair.second == 0) continue;

                std::pair<std::string, float> word;

                word.first = pair.first;
                word.second = std::log(pair.second + 1);

                words.push_back(word);
            }
            tf.push_back(words);
        }

        std::unordered_map<std::string, float> idf;
        for (const auto& pairs : bow)
            for (const auto& pair : pairs) idf[pair.first] = idf[pair.first] + 1.0f;

        for (const auto& pairs : idf)
            idf[pairs.first] = std::log((files_words.size() + 1.0f) / (idf[pairs.first] + 1.0f)) + 1.0f;

        std::vector<std::vector<std::pair<std::string, float>>> tfidf;
        for (const auto& pairs : tf)
        {
            std::vector<std::pair<std::string, float>> words;
            for (const auto& pair : pairs)
            {
                std::pair<std::string, float> word;
                word.first = pair.first;
                word.second = pair.second * idf[word.first];
                words.push_back(word);
            }
            tfidf.push_back(words);
        }

        for (auto& doc : tfidf) {
            float norm2 = 0.0f;
            for (const auto& kv : doc) norm2 += kv.second * kv.second;

            float norm = std::sqrt(norm2);
            if (norm > 0.0f) {
                for (auto& kv : doc) kv.second /= norm;
            }
        }

        return tfidf;
    }

    std::vector<std::pair<std::string, float>> tf_idf(
        const std::vector<std::string>& files_words)
    {
        std::vector<std::pair<std::string, int>> bow = LexiCore::vectorize::bag_of_word(files_words);
        std::vector<std::pair<std::string, float>> tf;

        for (const auto& pair : bow) {
            if (pair.second == 0) continue;

            std::pair<std::string, float> word;

            word.first = pair.first;
            word.second = std::log(pair.second + 1);

            tf.push_back(word);
        }

        std::unordered_map<std::string, float> idf;
        for (const auto& pair : bow) idf[pair.first] = idf[pair.first] + 1.0f;

        for (const auto& pairs : idf)
            idf[pairs.first] = std::log((files_words.size() + 1.0f) / (idf[pairs.first] + 1.0f)) + 1.0f;

        std::vector<std::pair<std::string, float>> tfidf;
        for (const auto& pair : tf) {
            std::pair<std::string, float> word;
            word.first = pair.first;
            word.second = pair.second * idf[word.first];
            tfidf.push_back(word);
        }

        float norm2 = 0.0f;
        for (auto& kv : tfidf) norm2 += kv.second * kv.second;

        float norm = std::sqrt(norm2);
        if (norm > 0.0f) {
            for (auto& kv: tfidf) kv.second /= norm;
        }

        return tfidf;
    }

}
