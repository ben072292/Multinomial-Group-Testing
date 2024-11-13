#ifndef BMGT_CORE_H_
#define BMGT_CORE_H_

#define BMGT_API(ret, func, args...)            \
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
#include <cstring>
#include <vector>
#ifdef ENABLE_SIMD
#if defined USE_INTEL_INTRINSICS && (defined(__SSE2__) || defined(__AVX2__) || defined(__AVX512F__))
#include <immintrin.h>
#endif
#endif

#ifdef ENABLE_SIMD
#if defined(__AVX512F__)
#define SIMD_WIDTH 64 / sizeof(bin_enc)
inline void *vector_alloc(std::size_t n)
{
    void *tmp = 0;
    if (posix_memalign(&tmp, 64, n))
    {
        throw std::bad_alloc();
    }
    return tmp;
}
#ifndef USE_INTEL_INTRINSICS
typedef int VecIntType __attribute__((vector_size(16 * sizeof(int))));
typedef double VecDoubleType __attribute__((vector_size(8 * sizeof(double))));
#define SIMD_SET0()                                    \
    {                                                  \
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 \
    }
#define SIMD_SET1(var)                                                                 \
    {                                                                                  \
        var, var, var, var, var, var, var, var, var, var, var, var, var, var, var, var \
    }
#define SIMD_SET_INC()                                       \
    {                                                        \
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 \
    }
#else
typedef __m512i VecIntType;
typedef __m512d VecDoubleType;
#define SIMD_SET0() _mm512_setzero_epi32()
#define SIMD_SET1(var) _mm512_set1_epi32(var)
#define SIMD_SET1_DOUBLE(var) _mm512_set1_pd(var)
#define SIMD_SET_INC() _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define SIMD_AND(a, b) _mm512_and_si512(a, b)
#define SIMD_OR(a, b) _mm512_or_si512(a, b)
#define SIMD_ADD(a, b) _mm512_add_epi32(a, b)
#define SIMD_ADD_DOUBLE(a, b) _mm512_add_pd(a, b)
#define SIMD_SUB(a, b) _mm512_sub_epi32(a, b)
#define SIMD_MUL(a, b) _mm512_mullo_epi32(a, b) // Low-part multiplication
#define SIMD_SLLI(a, imm) _mm512_slli_epi32(a, imm)
#define SIMD_SRLI(a, imm) _mm512_srai_epi32(a, imm) // srli gives wrong results, investigate
#define SIMD_LOWER_HALF(vec) _mm512_castsi512_si256(vec)
#define SIMD_UPPER_HALF(vec) _mm512_extracti64x4_epi64(vec, 1)
#endif

#elif defined(__AVX2__)
#define SIMD_WIDTH 32 / sizeof(bin_enc)
inline void *vector_alloc(std::size_t n)
{
    void *tmp = 0;
    if (posix_memalign(&tmp, 32, n))
    {
        throw std::bad_alloc();
    }
    return tmp;
}
#ifndef USE_INTEL_INTRINSICS
typedef int VecIntType __attribute__((vector_size(8 * sizeof(int))));
typedef double VecDoubleType __attribute__((vector_size(4 * sizeof(double))));
#define SIMD_SET0()            \
    {                          \
        0, 0, 0, 0, 0, 0, 0, 0 \
    }
#define SIMD_SET1(var)                         \
    {                                          \
        var, var, var, var, var, var, var, var \
    }
#define SIMD_SET_INC()         \
    {                          \
        0, 1, 2, 3, 4, 5, 6, 7 \
    }
#else
typedef __m256i VecIntType;
typedef __m256d VecDoubleType;
#define SIMD_SET0() _mm256_setzero_si256()
#define SIMD_SET1(var) _mm256_set1_epi32(var)
#define SIMD_SET1_DOUBLE(var) _mm256_set1_pd(var)
#define SIMD_SET_INC() _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0)
#define SIMD_AND(a, b) _mm256_and_si256(a, b)
#define SIMD_OR(a, b) _mm256_or_si256(a, b)
#define SIMD_ADD(a, b) _mm256_add_epi32(a, b)
#define SIMD_ADD_DOUBLE(a, b) _mm256_add_pd(a, b)
#define SIMD_SUB(a, b) _mm256_sub_epi32(a, b)
#define SIMD_MUL(a, b) _mm256_mullo_epi32(a, b) // Low-part multiplication
#define SIMD_SLLI(a, imm) _mm256_slli_epi32(a, imm)
#define SIMD_SRLI(a, imm) _mm256_srli_epi32(a, imm)
#define SIMD_LOWER_HALF(vec) _mm256_castsi256_si128(vec)      // Directly use the lower 128 bits
#define SIMD_UPPER_HALF(vec) _mm256_extractf128_si256(vec, 1) // Extract the upper 128 bits
#endif
#elif defined(__SSE2__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#undef USE_INTEL_INTRINSICS
#define SIMD_WIDTH 16 / sizeof(bin_enc)
typedef int VecIntType __attribute__((vector_size(4 * sizeof(int))));
typedef double VecDoubleType __attribute__((vector_size(2 * sizeof(double))));
#define SIMD_SET0() \
    {               \
        0, 0, 0, 0  \
    }
#define SIMD_SET1(var)     \
    {                      \
        var, var, var, var \
    }
#define SIMD_SET_INC() \
    {                  \
        0, 1, 2, 3     \
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