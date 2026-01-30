#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace LexiCore::vectorize {

    std::vector<std::vector<std::pair<std::string, float>>> tf_idf(
        const std::vector<std::vector<std::string>>& files_words);

    std::vector<std::pair<std::string, float>> tf_idf(
        const std::vector<std::string>& files_words);

}
