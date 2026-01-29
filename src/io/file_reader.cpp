#include "io/file_reader.hpp"

#include <fstream>
#include <iostream>

namespace nlptk::io {

    std::vector<std::vector<std::string>>
    read_files(const std::vector<std::string>& paths) {
        std::vector<std::vector<std::string>> all_lines;
        all_lines.reserve(paths.size());

        for (const auto& path : paths) {
            std::ifstream file(path);

            if (!file.is_open()) {
                std::cout << "Could not open file " << path << std::endl;
                continue;
            }

            std::string line;
            std::vector<std::string> lines;

            while (std::getline(file, line)) {
                lines.push_back(line);
            }

            all_lines.push_back(std::move(lines));
        }

        return all_lines;
    }

} // namespace nlptk::io
