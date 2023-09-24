#pragma once
#include "core.hpp"
#include <algorithm>

// trim true state by the given percent
int *trim_true_states(double *values, int n, double percent, int &resultSize);