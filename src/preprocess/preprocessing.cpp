#include "preprocess/preprocessing.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_set>
#include <vector>

namespace nlptk::preprocess {

namespace {

const std::unordered_set<std::string> STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "at",
    "for", "with", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as", "by", "from"
};

inline bool is_token_char(unsigned char c) {
    return std::isalnum(c) != 0;
}

std::vector<std::vector<std::string>>
split_to_word(const std::vector<std::vector<std::string>>& input_data) {
    std::vector<std::vector<std::string>> words;
    words.reserve(input_data.size());

    for (const auto& file : input_data) {
        std::vector<std::string> file_words;

        for (const auto& sentence : file) {
            std::string word;
            word.reserve(32);

            for (unsigned char ch : sentence) {
                if (!is_token_char(ch)) {
                    if (!word.empty()) {
                        file_words.push_back(std::move(word));
                        word.clear();
                        word.reserve(32);
                    }
                } else {
                    word.push_back(static_cast<char>(ch));
                }
            }

            if (!word.empty()) {
                file_words.push_back(std::move(word));
            }
        }

        words.push_back(std::move(file_words));
    }

    return words;
}

std::vector<std::string>
split_to_word(const std::string& input_data) {
    std::vector<std::string> words;
    std::string word;
    word.reserve(32);

    for (unsigned char ch : input_data) {
        if (!is_token_char(ch)) {
            if (!word.empty()) {
                words.push_back(std::move(word));
                word.clear();
                word.reserve(32);
            }
        } else {
            word.push_back(static_cast<char>(ch));
        }
    }

    if (!word.empty()) {
        words.push_back(std::move(word));
    }

    return words;
}

std::vector<std::vector<std::string>>
cleaning_word(const std::vector<std::vector<std::string>>& file_words) {
    std::vector<std::vector<std::string>> file_clean_words;
    file_clean_words.reserve(file_words.size());

    for (const auto& words : file_words) {
        std::vector<std::string> clean_words;
        clean_words.reserve(words.size());

        for (const auto& w : words) {
            std::string lower_word = w;
            for (char& ch : lower_word) {
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            }

            if (lower_word.size() <= 2 || lower_word.size() > 30) continue;
            if (STOPWORDS.contains(lower_word)) continue;

            clean_words.push_back(std::move(lower_word));
        }

        file_clean_words.push_back(std::move(clean_words));
    }

    return file_clean_words;
}

std::vector<std::string>
cleaning_word(const std::vector<std::string>& words) {
    std::vector<std::string> clean_words;
    clean_words.reserve(words.size());

    for (const auto& w : words) {
        std::string lower_word = w;
        for (char& ch : lower_word) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }

        if (lower_word.size() <= 2 || lower_word.size() > 30) continue;
        if (STOPWORDS.contains(lower_word)) continue;

        clean_words.push_back(std::move(lower_word));
    }

    return clean_words;
}

}

std::vector<std::vector<std::string>>
preprocess(const std::vector<std::vector<std::string>>& input_data) {
    auto file_words = split_to_word(input_data);
    return cleaning_word(file_words);
}

std::vector<std::string>
preprocess(const std::string& input_data) {
    auto file_words = split_to_word(input_data);
    return cleaning_word(file_words);
}

}
