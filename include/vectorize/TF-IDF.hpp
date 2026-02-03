#pragma once

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

namespace LexiCore::vectorize {

    class TF_IDF {
        std::unordered_map<std::string, float> idf;
        std::unordered_map<std::string, int> df;
        int N = -1;

    public:
        void fit(const std::vector<std::vector<std::string>>& files_words);
        void fit(const std::vector<std::string>& files_words);

        std::vector<std::vector<std::pair<std::string, float>>> transform(
            const std::vector<std::vector<std::string>>& files_words) const;
        std::vector<std::pair<std::string, float>> transform(
            const std::vector<std::string>& files_words) const;

        std::vector<std::vector<std::pair<std::string, float>>> fit_transform(
            const std::vector<std::vector<std::string>>& files_words);
        std::vector<std::pair<std::string, float>> fit_transform(
            const std::vector<std::string>& files_words);

        void reset();
    };

}
