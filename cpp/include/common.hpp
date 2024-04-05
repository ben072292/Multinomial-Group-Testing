#ifndef MGT_CORE_H_
#define MGT_CORE_H_

#define MGT_API(ret, func, args...)            \
    extern "C"                                 \
        __attribute__((visibility("default"))) \
        ret                                    \
        func(args)

#if defined(_WIN32) || defined(__WIN32__)
    #ifdef MYLIBRARY_EXPORTS
        #define EXPORT __declspec(dllexport)
    #else
        #define EXPORT __declspec(dllimport)
    #endif
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

#include "bmgt.h"
#include "log.h"
#include <chrono>
#include <cmath> // std:abs(double) support for older gcc
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>
#ifdef ENABLE_OMP
#include <omp.h>
#endif
#include <sstream>
#include <string>
#include <vector>
#ifdef ENABLE_SIMD
#if defined(__SSE2__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
#endif

#ifdef ENABLE_SIMD
#if defined(__AVX512F__)
#define SIMD_WIDTH 64 / sizeof(bin_enc)
typedef int VecIntType __attribute__((vector_size(16 * sizeof(int))));
typedef double VecDoubleType __attribute__((vector_size(8 * sizeof(double))));
#define SIMD_SET1(var)                                                                 \
    {                                                                                  \
        var, var, var, var, var, var, var, var, var, var, var, var, var, var, var, var \
    }
#define SIMD_SET_INC                                         \
    {                                                        \
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 \
    }

inline void *vector_alloc(std::size_t n)
{
    void *tmp = 0;
    if (posix_memalign(&tmp, sizeof(VecIntType), n))
    {
        throw std::bad_alloc();
    }
    return tmp;
}

#elif defined(__AVX2__)
#define SIMD_WIDTH 32 / sizeof(bin_enc)
typedef int VecIntType __attribute__((vector_size(8 * sizeof(int))));
typedef double VecDoubleType __attribute__((vector_size(4 * sizeof(double))));
#define SIMD_SET1(var)                         \
    {                                          \
        var, var, var, var, var, var, var, var \
    }
#define SIMD_SET_INC           \
    {                          \
        0, 1, 2, 3, 4, 5, 6, 7 \
    }
inline void *vector_alloc(std::size_t n)
{
    void *tmp = 0;
    if (posix_memalign(&tmp, sizeof(VecIntType), n))
    {
        throw std::bad_alloc();
    }
    return tmp;
}
#elif defined(__SSE2__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#define SIMD_WIDTH 16 / sizeof(bin_enc)
typedef int VecIntType __attribute__((vector_size(4 * sizeof(int))));
typedef double VecDoubleType __attribute__((vector_size(2 * sizeof(double))));
#define SIMD_SET1(var)     \
    {                      \
        var, var, var, var \
    }
#define SIMD_SET_INC \
    {                \
        0, 1, 2, 3   \
    }
inline void *vector_alloc(std::size_t n)
{
    void *tmp = 0;
    if (posix_memalign(&tmp, sizeof(VecIntType), n))
    {
        throw std::bad_alloc();
    }
    return tmp;
}
#else
#error "Unsupported SIMD instruction set."
#endif
#endif

inline std::string get_curr_time()
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
    return oss.str();
}

inline std::string to_binary(bin_enc n, int len)
{
    std::string r;
    while (n != 0)
    {
        r = (n % 2 == 0 ? "0" : "1") + r;
        n /= 2;
    }
    int curr_len = r.length();
    for (int i = 0; i < len - curr_len; i++)
    {
        r = "0" + r;
    }
    return r;
}

inline std::string hardware_config_summary()
{
    std::ostringstream oss;
    oss << "OpenMP is,";
#ifdef ENABLE_OMP
    oss << "Enabled,omp_num_threads is set to," << omp_get_num_threads() << std::endl;
#else
    oss << "Disabled" << std::endl;
#endif

    oss << "SIMD is,";
#ifdef ENABLE_SIMD
#if defined(__AVX512F__)
    oss << "Enabled, Architecture support is AVX-512";
#elif defined(__AVX2__)
    oss << "Enabled, Architecture support is AVX-2";
#elif defined(__SSE2__)
    oss << "Enabled, Architecture support is SSE-2";
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    oss << "Enabled, Architecture support is ARM Neon";
#else
#error "Unsupported SIMD instruction set."
#endif
#else
    oss << "Disabled";
#endif
    return oss.str();
}

#endif