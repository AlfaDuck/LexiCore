#include "vectorize/bow.hpp"

#include <algorithm>
#include <unordered_map>

namespace LexiCore::vectorize {

    std::vector<std::vector<std::pair<std::string, int>>> BagOfWords::fit_transform(
        const std::vector<std::vector<std::string>>& files_words,
        std::optional<int> top_n) {

        std::vector<std::vector<std::pair<std::string, int>>> file_items;
        file_items.reserve(files_words.size());

        for (const auto& words : files_words) {
            std::unordered_map<std::string, int> counts;
            for (const auto& w : words) counts[w]++;

            std::vector<std::pair<std::string, int>> items;
            items.reserve(counts.size());
            for (const auto& kv : counts) items.push_back(kv);

            std::sort(items.begin(), items.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

            if (top_n.has_value()) {
                int limit = *top_n;
                if (limit > 0 && limit < static_cast<int>(items.size())) items.resize(limit);
            }

            file_items.push_back(std::move(items));
        }

        return file_items;
    }

    std::vector<std::pair<std::string, int>> BagOfWords::fit_transform(
        const std::vector<std::string>& words,
        std::optional<int> top_n) {

        std::unordered_map<std::string, int> counts;
        for (const auto& w : words) counts[w]++;

        std::vector<std::pair<std::string, int>> items;
        items.reserve(counts.size());
        for (const auto& kv : counts) items.push_back(kv);

        std::sort(items.begin(), items.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        if (top_n.has_value()) {
            int limit = *top_n;
            if (limit > 0 && limit < static_cast<int>(items.size())) items.resize(limit);
        }

        return items;
    }

}