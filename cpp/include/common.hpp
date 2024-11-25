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
#include "intrinsic_wrapper.hpp"
#include <chrono>
#include <cmath> // std:abs(double) support for older gcc
#include <cstdlib>
#include <ctime>
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
    char* omp_threads = getenv("OMP_NUM_THREADS");
    oss << "Enabled,omp_num_threads is set to," << std::stoi(omp_threads) << std::endl;
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