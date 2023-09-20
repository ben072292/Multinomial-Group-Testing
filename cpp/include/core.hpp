#ifndef MGT_CORE_H_
#define MGT_CORE_H_

#define MGT_API(ret, func, args...)            \
    extern "C"                                 \
        __attribute__((visibility("default"))) \
        ret func(args)                         

#include "log.h"
#include "dilution.hpp"
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

// Vector types
typedef int _mm512_si __attribute__((vector_size(16 * sizeof(int))));
typedef double _mm512_d __attribute__((vector_size(8 * sizeof(double))));

inline _mm512_si *__mm512_si_alloc(std::size_t n)
{
    void *tmp = 0;
    if (posix_memalign(&tmp, sizeof(_mm512_si), sizeof(_mm512_si) * n))
    {
        throw std::bad_alloc();
    }
    return (_mm512_si *)tmp;
}

inline _mm512_d *__mm512_d_alloc(std::size_t n)
{
    void *tmp = 0;
    if (posix_memalign(&tmp, sizeof(_mm512_d), sizeof(_mm512_d) * n))
    {
        throw std::bad_alloc();
    }
    return (_mm512_d *)tmp;
}

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

#endif