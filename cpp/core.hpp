#ifndef MGT_CORE_H_
#define MGT_CORE_H_

#ifndef PROFAPI
#define MGT_API(ret, func, args...)        \
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
    ret func(args)
#endif // end PROFAPI

#include <iostream>
#include <cstdlib>
#include <limits>
#include <cmath> // std:abs(double) support for older gcc
#include <vector>
#include <omp.h>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <mpi.h>
#include <omp.h>

// Differentiate binary encoded states from regular index
typedef int bin_enc;

// Product_lattice copy operation
#define NO_COPY_PROB_DIST 0
#define SHALLOW_COPY_PROB_DIST 1
#define DEEP_COPY_PROB_DIST 2

// Product_lattice parallelism
#define DATA_PARALLELISM 0
#define MODEL_PARALLELISM 1

inline std::string to_binary(bin_enc n, int len){
    std::string r;
    while(n!=0) {r=(n%2==0 ?"0":"1")+r; n/=2;}
    int curr_len = r.length();
    for(int i = 0; i < len-curr_len; i++){
        r = "0" + r;
    }
    return r;
}

inline std::string get_curr_time(){
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
    return oss.str();
}

#endif