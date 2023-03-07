#ifndef MGT_CORE_H_
#define MGT_CORE_H_

#ifndef PROFAPI
#define MGT_API(ret, func, args...)            \
    extern "C"                                 \
        __attribute__((visibility("default"))) \
        ret                                    \
        func(args)
#endif // end PROFAPI

#include <chrono>
#include <cmath> // std:abs(double) support for older gcc
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <omp.h>
#include <sstream>
#include <string>
#include <vector>

// Differentiate binary encoded states from regular index
typedef int bin_enc;

typedef int int8_v __attribute__ ((vector_size (8 * sizeof(int))));
typedef int int16_v __attribute__ ((vector_size (16 * sizeof(int))));
typedef double double4_v __attribute__((vector_size (4 * sizeof(double))));
typedef double double8_v __attribute__((vector_size (8 * sizeof(double))));

// Product_lattice type
#define MP_NON_DILUTION 1
#define MP_DILUTION 2
#define DP_NON_DILUTION 3
#define DP_DILUTION 4

// Product_lattice copy operation
#define NO_COPY_PROB_DIST 0
#define SHALLOW_COPY_PROB_DIST 1
#define DEEP_COPY_PROB_DIST 2

// Product_lattice parallelism
#define DATA_PARALLELISM 0
#define MODEL_PARALLELISM 1

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

inline std::string get_curr_time()
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
    return oss.str();
}

static int8_v* int8_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(int8_v), sizeof(int8_v) * n)) {
        throw std::bad_alloc();
    }
    return (int8_v*)tmp;
}

static int16_v* int16_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(int16_v), sizeof(int16_v) * n)) {
        throw std::bad_alloc();
    }
    return (int16_v*)tmp;
}

static double4_v* double4_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_v), sizeof(double4_v) * n)) {
        throw std::bad_alloc();
    }
    return (double4_v*)tmp;
}

static double8_v* double8_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double8_v), sizeof(double8_v) * n)) {
        throw std::bad_alloc();
    }
    return (double8_v*)tmp;
}

#endif