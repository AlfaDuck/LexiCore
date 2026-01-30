#pragma once

#include <string>
#include <utility>
#include <vector>

namespace LexiCore::similarity {

    float cosine_similarity(
        const std::vector<std::pair<std::string, int>>& a,
        const std::vector<std::pair<std::string, int>>& b
    );

    float cosine_similarity(
        const std::vector<std::pair<std::string, float>>& a,
        const std::vector<std::pair<std::string, float>>& b
    );

}
