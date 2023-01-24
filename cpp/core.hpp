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
#include "/opt/homebrew/include/mpi.h"

inline std::string to_binary(int n, int len)
{
    std::string r;
    while(n!=0) {r=(n%2==0 ?"0":"1")+r; n/=2;}
    int curr_len = r.length();
    for(int i = 0; i < len-curr_len; i++){
        r = "0" + r;
    }
    return r;
}

#endif