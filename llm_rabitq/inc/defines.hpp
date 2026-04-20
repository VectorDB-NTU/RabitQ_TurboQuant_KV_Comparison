#pragma once

#include <stdint.h>
#include <random>

// Eigen is optional — only needed for Matrix rotator mode.
// Install: apt install libeigen3-dev (provides /usr/include/eigen3/Eigen/Dense)
// Or download header-only Eigen to inc/third/Eigen/
#if __has_include("third/Eigen/Dense")
#include "third/Eigen/Dense"
#define HAS_EIGEN 1
#elif __has_include(<Eigen/Dense>)
#include <Eigen/Dense>
#define HAS_EIGEN 1
#else
#define HAS_EIGEN 0
#endif

#define FORCE_INLINE inline __attribute__((always_inline))

using PID = uint32_t;

#if HAS_EIGEN
using FloatRowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using RowMajorMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
RowMajorMatrix<T> random_gaussian_matrix(size_t rows, size_t cols) {
    RowMajorMatrix<T> rand(rows, cols);
#ifdef DEBUG_BATCH_CONSTRUCT
    static std::mt19937 gen(42);
#else
    static std::random_device rd;
    static std::mt19937 gen(rd());
#endif
    std::normal_distribution<T> dist(0, 1);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            rand(i, j) = dist(gen);
    return rand;
}
#endif
