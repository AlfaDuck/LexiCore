#pragma once

#include <string>
#include <vector>

namespace LexiCore::io {

    // Reads multiple text files and returns lines for each file.
    std::vector<std::vector<std::string>>
    read_files(const std::vector<std::string>& paths);

}
