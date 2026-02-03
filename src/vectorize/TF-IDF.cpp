#include "vectorize/TF-IDF.hpp"
#include "vectorize/bow.hpp"

#include <cmath>

namespace LexiCore::vectorize {

    void TF_IDF::fit(const std::vector<std::vector<std::string>> &files_words) {
        if (N != -1) return;

        N = files_words.size();
        auto bow_files = bag_of_word(files_words);

        for (const auto& pairs : bow_files)
            for (const auto& pair : pairs) df[pair.first] = df[pair.first] + 1;

        for (const auto& pairs : df)
            idf[pairs.first] = std::log((N + 1.0f) / ((float)df[pairs.first] + 1.0f)) + 1.0f;
    }

    void TF_IDF::fit(const std::vector<std::string> &files_word) {
        if (N != -1) return;

        N = 1;
        auto bow_file = bag_of_word(files_word);

        for (const auto& pair : bow_file)
            df[pair.first] = df[pair.first] + 1;

        for (const auto& pairs : df)
            idf[pairs.first] = std::log((N + 1.0f) / ((float)df[pairs.first] + 1.0f)) + 1.0f;
    }

    std::vector<std::vector<std::pair<std::string, float>>> TF_IDF::transform(
        const std::vector<std::vector<std::string> > &files_words) const {

        std::vector<std::vector<std::pair<std::string, float>>> tfidf;
        if (N == -1) return tfidf;

        auto bow = bag_of_word(files_words);

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

        for (const auto& pairs : tf) {
            std::vector<std::pair<std::string, float>> words;

            for (const auto& pair : pairs) {
                auto it = idf.find(pair.first);
                if (it == idf.end()) {
                    continue;
                }

                std::pair<std::string, float> word;
                word.first  = pair.first;
                word.second = pair.second * it->second;

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

    std::vector<std::pair<std::string, float>> TF_IDF::transform(
        const std::vector<std::string> &files_words) const {

        std::vector<std::pair<std::string, float>> tfidf;
        if (N == -1) return tfidf;

        auto bow = bag_of_word(files_words);

        std::vector<std::pair<std::string, float>> tf;
        tf.reserve(bow.size());

        for (const auto& pair : bow) {
            if (pair.second == 0) continue;
            tf.push_back({pair.first, std::log(pair.second + 1.0f)});
        }

        tfidf.reserve(tf.size());
        for (const auto& pair : tf) {
            auto it = idf.find(pair.first);
            if (it == idf.end()) continue;
            tfidf.push_back({pair.first, pair.second * it->second});
        }

        float norm2 = 0.0f;
        for (const auto& kv : tfidf) norm2 += kv.second * kv.second;

        float norm = std::sqrt(norm2);
        if (norm > 0.0f) {
            for (auto& kv : tfidf) kv.second /= norm;
        }
        return tfidf;
    }

    std::vector<std::vector<std::pair<std::string, float>>> TF_IDF::fit_transform(
        const std::vector<std::vector<std::string> > &files_words) {

        fit(files_words);
        return transform(files_words);
    }

    std::vector<std::pair<std::string, float>> TF_IDF::fit_transform(
        const std::vector<std::string> &files_word) {

        fit(files_word);
        return transform(files_word);
    }

    void TF_IDF::reset() {
        N = -1;
        df.clear();
        idf.clear();
    }

}
