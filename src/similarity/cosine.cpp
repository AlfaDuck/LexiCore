#include "similarity/cosine.hpp"

#include <cmath>
#include <unordered_map>

namespace LexiCore::similarity {

    float cosine_similarity(const std::vector<std::pair<std::string, int>>& a,
                            const std::vector<std::pair<std::string, int>>& b) {
        std::unordered_map<std::string, int> am, bm;
        am.reserve(a.size());
        bm.reserve(b.size());

        for (const auto& item : a) am[item.first] = item.second;
        for (const auto& item : b) bm[item.first] = item.second;

        float dot_product = 0.0f;
        for (const auto& kv : am) {
            auto it = bm.find(kv.first);
            if (it != bm.end()) dot_product += static_cast<float>(kv.second * it->second);
        }

        float norm_a = 0.0f, norm_b = 0.0f;
        for (const auto& kv : am) norm_a += static_cast<float>(kv.second * kv.second);
        for (const auto& kv : bm) norm_b += static_cast<float>(kv.second * kv.second);

        norm_a = std::sqrt(norm_a);
        norm_b = std::sqrt(norm_b);

        if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
        return dot_product / (norm_a * norm_b);
    }

}
