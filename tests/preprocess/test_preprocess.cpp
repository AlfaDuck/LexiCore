#include <cassert>
#include <set>
#include <string>
#include <vector>

#include "preprocess/preprocessing.hpp"

static std::set<std::string> to_set(const std::vector<std::string>& v) {
    return std::set<std::string>(v.begin(), v.end());
}

static void test_preprocess_string_basic() {
    const std::string text = "This is a TEST, and it is from the System in 2025!";

    auto tokens = LexiCore::preprocess::preprocess(text);
    auto s = to_set(tokens);

    assert(s.count("test") == 1);
    assert(s.count("system") == 1);
    assert(s.count("2025") == 1);

    assert(s.count("this") == 0);
    assert(s.count("is") == 0);
    assert(s.count("a") == 0);
    assert(s.count("and") == 0);
    assert(s.count("from") == 0);
    assert(s.count("the") == 0);
    assert(s.count("it") == 0);

    assert(s.count("in") == 0);
}

static void test_preprocess_length_filter() {
    const std::string text =
        "ok good excellent supercalifragilisticexpialidocious_is_very_long_word";

    auto tokens = LexiCore::preprocess::preprocess(text);
    auto s = to_set(tokens);

    assert(s.count("ok") == 0);
    assert(s.count("good") == 1);
    assert(s.count("excellent") == 1);

    for (const auto& tok : tokens) {
        assert(tok.size() <= 30);
    }
}

static void test_preprocess_multi_document() {
    std::vector<std::vector<std::string>> docs = {
        { "Apple apple, banana!", "This is a test." },   // doc0
        { "Car bus train.", "The vehicle is fast." }     // doc1
    };

    auto out = LexiCore::preprocess::preprocess(docs);

    assert(out.size() == 2);

    auto s0 = to_set(out[0]);
    assert(s0.count("apple") == 1);
    assert(s0.count("banana") == 1);
    assert(s0.count("test") == 1);

    assert(s0.count("this") == 0);
    assert(s0.count("is") == 0);

    auto s1 = to_set(out[1]);
    assert(s1.count("car") == 1);
    assert(s1.count("bus") == 1);
    assert(s1.count("train") == 1);
    assert(s1.count("vehicle") == 1);
    assert(s1.count("fast") == 1);

    assert(s1.count("the") == 0);
    assert(s1.count("is") == 0);
}

int main() {
    test_preprocess_string_basic();
    test_preprocess_length_filter();
    test_preprocess_multi_document();
    return 0;
}
