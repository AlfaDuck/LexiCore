#include <cassert>
#include <cmath>
#include <string>
#include <vector>

#include "similarity/cosine.hpp"

static bool almost_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

static void test_cosine_identical_vectors() {
    std::vector<std::pair<std::string, int>> v = {
        {"apple", 2},
        {"banana", 1},
        {"orange", 3}
    };

    float s = LexiCore::similarity::cosine_similarity(v, v);
    assert(almost_equal(s, 1.0f));
}

static void test_cosine_orthogonal_vectors() {
    std::vector<std::pair<std::string, int>> a = {
        {"apple", 1},
        {"banana", 2}
    };

    std::vector<std::pair<std::string, int>> b = {
        {"car", 3},
        {"bus", 4}
    };

    float s = LexiCore::similarity::cosine_similarity(a, b);
    assert(almost_equal(s, 0.0f));
}

static void test_cosine_partial_overlap() {
    std::vector<std::pair<std::string, int>> a = {
        {"apple", 2},
        {"banana", 1}
    };

    std::vector<std::pair<std::string, int>> b = {
        {"apple", 1},
        {"car", 3}
    };

    float s = LexiCore::similarity::cosine_similarity(a, b);
    assert(s > 0.0f);
    assert(s < 1.0f);
}

static void test_cosine_empty_vector() {
    std::vector<std::pair<std::string, int>> a;
    std::vector<std::pair<std::string, int>> b = {
        {"apple", 1}
    };

    float s1 = LexiCore::similarity::cosine_similarity(a, b);
    float s2 = LexiCore::similarity::cosine_similarity(b, a);

    assert(almost_equal(s1, 0.0f));
    assert(almost_equal(s2, 0.0f));
}

int main() {
    test_cosine_identical_vectors();
    test_cosine_orthogonal_vectors();
    test_cosine_partial_overlap();
    test_cosine_empty_vector();
    return 0;
}
