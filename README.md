# LexiCore

**LexiCore** is a modern, lightweight C++ library for classical Natural Language Processing (NLP).
It focuses on building a clean, testable, and extensible text-processing pipeline using modern C++.

The project is designed with:

* Clean architecture
* Strong separation of concerns
* Full unit-test coverage
* Future readiness for GPU / CUDA acceleration

in mind.

---

## ✨ Features

### Text Preprocessing

* Tokenization
* Lowercasing
* Stopwords removal
* Token length filtering

### Vectorization

* Bag-of-Words (BoW)
* TF-IDF

    * Log-scaled Term Frequency
    * Smoothed Inverse Document Frequency
    * L2 normalization

### Similarity

* Cosine similarity for sparse vectors

### Search

* Similarity search using Bag-of-Words
* Similarity search using TF-IDF vectors

### Engineering

* Modern C++ (C++20)
* Namespace-based modular design
* Clean and scalable CMake setup
* Unit tests for all core modules

---

## 🗂 Project Structure

```text
LexiCore/
├── include/
│   └── LexiCore/
│       ├── io/
│       ├── preprocess/
│       ├── vectorize/
│       ├── similarity/
│       └── search/
├── src/
│   ├── io/
│   ├── preprocess/
│   ├── vectorize/
│   ├── similarity/
│   └── search/
├── tests/
│   ├── io/
│   ├── preprocess/
│   ├── vectorize/
│   ├── similarity/
│   └── search/
├── main.cpp
├── CMakeLists.txt
└── README.md
```

---

## 🚀 Build & Run

### Requirements

* CMake ≥ 3.20
* C++20 compatible compiler (GCC / Clang / MSVC)
* Ninja (optional, recommended)

---

### Build

```bash
git clone https://github.com/your-username/LexiCore.git
cd LexiCore

cmake -S . -B build
cmake --build build
```

---

### Run main application

```bash
./build/lexicore_app
```

---

### Run tests

Run all tests at once:

```bash
ctest --test-dir build
```

Run individual test executables:

```bash
./build/test_file_reader
./build/test_preprocess
./build/test_bow
./build/test_tfidf
./build/test_cosine
./build/test_search
```

---

## 🧪 Testing Philosophy

LexiCore follows a strict testing-first mindset.

* Each core module is tested independently
* Tests are deterministic and isolated
* No hidden global state
* Designed to support both unit and end-to-end tests

This ensures correctness, maintainability, and confidence during refactoring.

---

## 🧠 Example Usage

### Preprocessing

```cpp
auto tokens = LexiCore::preprocess::preprocess(
    "This is a simple test sentence"
);
```

---

### Bag-of-Words

```cpp
auto bow = LexiCore::vectorize::bag_of_word(tokens);
```

---

### TF-IDF

```cpp
auto tfidf = LexiCore::vectorize::tf_idf(tokens);
```

---

### Similarity Search (BoW)

```cpp
auto [index, score] =
    LexiCore::search::similarity_search_bow(
        documents,
        "apple banana fruit"
    );
```

---

### Similarity Search (TF-IDF)

```cpp
auto [index, score] =
    LexiCore::search::similarity_search_tfidf(
        documents,
        "apple banana fruit"
    );
```

---

## 🛣 Roadmap

* CUDA-accelerated vectorization
* GPU-based similarity search
* Vocabulary persistence
* Incremental indexing
* Python bindings

---

## 📄 License

This project is licensed under the MIT License.