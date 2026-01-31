#include <string>
#include <utility>
#include <vector>

namespace LexiCore::search
{
    std::pair<int, float> similarity_search_bow(
        const std::vector<std::vector<std::string>>& tokenized_files,
        const std::string& input);

    std::pair<int, float> similarity_search_tfidf(
        const std::vector<std::vector<std::string>>& tokenized_files,
        const std::string& input);

    std::pair<int, float> similarity_search_bow_vec(
        const std::vector<std::vector<std::pair<std::string, int>>>& bow_files,
        const std::vector<std::pair<std::string, int>>& bow_query);

    std::pair<int, float> similarity_search_tfidf_vec(
        const std::vector<std::vector<std::pair<std::string, float>>>& tfidf_files,
        const std::vector<std::pair<std::string, float>>& tfidf_query);
}