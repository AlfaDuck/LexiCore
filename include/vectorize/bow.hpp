#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace LexiCore::vectorize {

    class BagOfWords {
    public:
        static std::vector<std::vector<std::pair<std::string, int>>> fit_transform(
            const std::vector<std::vector<std::string>>& files_words,
            std::optional<int> top_n = std::nullopt);
        static std::vector<std::pair<std::string, int>> fit_transform(
            const std::vector<std::string>& words,
            std::optional<int> top_n = std::nullopt);
    };

}
