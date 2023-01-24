#pragma once

typedef struct Halving_res{
    double min;
    int candidate;

    Halving_res(double val1 = 2.0, int val2 = -1){min = val1; candidate = val2;}
    inline void reset(){min = 2.0, candidate = -1;}
} Halving_res;