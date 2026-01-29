#pragma once

#include <string>
#include <utility>
#include <vector>

namespace nlptk::similarity {

    float cosine_similarity(
        const std::vector<std::pair<std::string, int>>& a,
        const std::vector<std::pair<std::string, int>>& b
    );

}
