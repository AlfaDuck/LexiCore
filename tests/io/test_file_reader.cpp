#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "io/file_reader.hpp"

namespace fs = std::filesystem;

static void write_file(const fs::path& p, const std::vector<std::string>& lines) {
    std::ofstream out(p, std::ios::binary);
    assert(out.is_open());
    for (const auto& line : lines) out << line << "\n";
}

static void test_read_files_reads_multiple_files() {
    // Arrange
    fs::path dir = fs::temp_directory_path() / "lexicore_tests_io";
    fs::create_directories(dir);

    fs::path f1 = dir / "a.txt";
    fs::path f2 = dir / "b.txt";

    write_file(f1, {"line1", "line2"});
    write_file(f2, {"x", "y", "z"});

    std::vector<std::string> paths = { f1.string(), f2.string() };

    // Act
    auto docs = LexiCore::io::read_files(paths);

    // Assert
    assert(docs.size() == 2);

    assert(docs[0].size() == 2);
    assert(docs[0][0] == "line1");
    assert(docs[0][1] == "line2");

    assert(docs[1].size() == 3);
    assert(docs[1][0] == "x");
    assert(docs[1][1] == "y");
    assert(docs[1][2] == "z");

    // Cleanup
    fs::remove_all(dir);
}

static void test_read_files_skips_missing_file() {
    // Arrange
    fs::path dir = fs::temp_directory_path() / "lexicore_tests_io_missing";
    fs::create_directories(dir);

    fs::path existing = dir / "ok.txt";
    fs::path missing  = dir / "missing.txt";

    write_file(existing, {"only"});

    std::vector<std::string> paths = { existing.string(), missing.string() };

    // Act
    auto docs = LexiCore::io::read_files(paths);

    // Assert
    // Your implementation: on missing file, it 'continue's and does NOT push an empty vector.
    assert(docs.size() == 1);
    assert(docs[0].size() == 1);
    assert(docs[0][0] == "only");

    // Cleanup
    fs::remove_all(dir);
}

int main() {
    test_read_files_reads_multiple_files();
    test_read_files_skips_missing_file();
    return 0;
}
