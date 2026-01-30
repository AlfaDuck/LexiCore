#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "include/io/file_reader.hpp"
#include "include/preprocess/preprocessing.hpp"
#include "include/vectorize/bow.hpp"
#include "include/similarity/cosine.hpp"

namespace {

    std::pair<int, float>
    search_similarity(const std::vector<std::vector<std::pair<std::string, int>>>& files_words) {
        std::string input;
        std::cout << "Enter a sentence to find similarity of it to other files: ";
        std::getline(std::cin, input);

        auto input_words = LexiCore::preprocess::preprocess(input);
        auto input_words_count = LexiCore::vectorize::bag_of_word(input_words);

        float best_similarity = 0.0f;
        int best_index = -1;

        for (int i = 0; i < static_cast<int>(files_words.size()); i++) {
            float s = LexiCore::similarity::cosine_similarity(files_words[i], input_words_count);
            if (s > best_similarity) {
                best_similarity = s;
                best_index = i;
            }
        }

        std::cout << "Best file index: " << best_index
                  << " | Similarity: " << best_similarity << "\n";

        return {best_index, best_similarity};
    }

}

int main() {
    std::string base = "D:/Code/C++/NLPTK/";
    std::vector<std::string> paths = {
            base + "data/1.txt",
            base + "data/2.txt",
            base + "data/3.txt"
    };

    auto results = LexiCore::io::read_files(paths);
    auto words = LexiCore::preprocess::preprocess(results);
    auto files_words = LexiCore::vectorize::bag_of_word(words);

    for (int i = 0; i < static_cast<int>(files_words.size()); i++) {
        for (int j = i + 1; j < static_cast<int>(files_words.size()); j++) {
            float s = LexiCore::similarity::cosine_similarity(files_words[i], files_words[j]);
            std::cout << "similarity between " << paths[i] << " and " << paths[j]
                      << " = " << s << "\n";
        }
    }

    (void)search_similarity(files_words);
    return 0;
}
