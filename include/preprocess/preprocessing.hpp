#pragma once

#include <string>
#include <vector>

namespace nlptk::preprocess {

    // Preprocess multiple documents (each document is a list of lines).
    std::vector<std::vector<std::string>>
    preprocess(const std::vector<std::vector<std::string>>& input_data);

    // Preprocess a single text string.
    std::vector<std::string>
    preprocess(const std::string& input_text);

}
