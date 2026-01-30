#include <string>
#include <utility>
#include <vector>

namespace LexiCore::search
{
    std::pair<int, float> similarity_search_bow(const std::vector<std::vector<std::string>>& files,
        const std::string& input);

    std::pair<int, float> similarity_search_tfidf(const std::vector<std::vector<std::string>>& files,
        const std::string& input);
}