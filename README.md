# LexiCore

**LexiCore** is a modern, lightweight C++ library for classical Natural Language Processing (NLP).
It focuses on building a clean, testable, and extensible text-processing pipeline using modern C++.

The project is designed with:

- Clean architecture
- Strong separation of concerns
- Explicit state management
- Full unit-test coverage
- Future readiness for GPU / CUDA acceleration
- Planned Python bindings

in mind.

---

## ✨ Features

### Text Preprocessing
- Tokenization
- Lowercasing
- Stopword removal
- Token length filtering

### Vectorization

#### Bag-of-Words (BoW)
- Stateless utility API
- Optional top-N term selection
- Deterministic output ordering

#### TF-IDF
- Stateful, class-based design
- Log-scaled Term Frequency
- Smoothed Inverse Document Frequency
- L2 normalization
- Explicit `fit / transform` pipeline
- Proper handling of out-of-vocabulary (OOV) terms

### Similarity
- Cosine similarity for sparse vectors

### Search
- Similarity search using Bag-of-Words vectors
- Similarity search using TF-IDF vectors
- Clear separation between vectorization and search logic

### Engineering
- Modern C++ (C++20)
- Namespace-based modular design
- Header / implementation separation
- Clean and scalable CMake setup
- Full unit test coverage for all core modules

---

## 🗂 Project Structure

```text
LexiCore/
├── include/
│   ├── io/
│   ├── preprocess/
│   ├── vectorize/
│   │   ├── bow.hpp
│   │   └── TF-IDF.hpp
│   ├── similarity/
│   └── search/
├── src/
│   ├── io/
│   ├── preprocess/
│   ├── vectorize/
│   │   ├── bow.cpp
│   │   └── TF-IDF.cpp
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
````

---

## 🚀 Build & Run

### Requirements

* CMake ≥ 3.20
* C++20 compatible compiler (GCC / Clang / MSVC)
* Ninja (optional, recommended)

---

### Build

```bash
git clone https://github.com/AlfaDuck/LexiCore.git
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

```bash
ctest --test-dir build --output-on-failure
```

---

## 🧪 Testing Philosophy

LexiCore follows a strict testing-first mindset.

* Each core module is tested independently
* Tests are deterministic and isolated
* No hidden global state
* Explicit validation of edge cases (OOV, empty inputs, normalization)

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
using LexiCore::vectorize::BagOfWords;

auto bows = BagOfWords::fit_transform(tokens);
auto top_bows = BagOfWords::fit_transform(tokens, 10);
```

---

### TF-IDF (Class-based API)

```cpp
using LexiCore::vectorize::TF_IDF;

TF_IDF model;

// Fit on corpus
model.fit(documents_tokens);

// Transform corpus
auto tfidf_vectors = model.transform(documents_tokens);

// Transform query using the same model
auto query_vec = model.transform(query_tokens);
```

---

### Similarity Search (BoW vectors)

```cpp
auto [index, score] =
    LexiCore::search::similarity_search_bow_vec(
        bow_vectors,
        query_bow
    );
```

---

### Similarity Search (TF-IDF vectors)

```cpp
auto [index, score] =
    LexiCore::search::similarity_search_tfidf_vec(
        tfidf_vectors,
        query_tfidf
    );
```

---

## 🛣 Roadmap

* CUDA-accelerated vectorization
* GPU-based similarity search
* Incremental indexing
* Vocabulary persistence
* Python bindings (via pybind11)
* Prebuilt Python wheels

---

## 📄 License

This project is licensed under the MIT License.