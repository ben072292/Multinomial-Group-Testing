#pragma once
#include <algorithm>

// trim true state by the given percent
int *trim_true_states(double *values, int n, double percent, int &resultSize);

// use symmetric property to trim true states
int *symmetric_true_state(int n);

// used to compute symmetric coefficient
double n_choose_k(int n, int k);