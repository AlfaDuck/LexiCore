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

    std::pair<int, float> similarity_search_bow_raw(
        const std::vector<std::vector<std::string>>& raw_files_lines,
        const std::string& input);

    std::pair<int, float> similarity_search_tfidf_raw(
        const std::vector<std::vector<std::string>>& raw_files_lines,
        const std::string& input);

}