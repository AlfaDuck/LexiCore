#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace LexiCore::vectorize {

    // Bag-of-Words counts for multiple documents.
    // If n is set, keeps only top-n tokens per document by frequency (descending).
    std::vector<std::vector<std::pair<std::string, int>>> bag_of_word(
        const std::vector<std::vector<std::string>>& files_words,
        std::optional<int> n = std::nullopt
    );

    // Bag-of-Words counts for a single document.
    std::vector<std::pair<std::string, int>> bag_of_word(
        const std::vector<std::string>& words
    );

}
