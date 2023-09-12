#include "dilution.hpp"

double **generate_dilution(int n, double alpha, double h)
{
    double **ret = new double *[n];
    int k;
    for (int rk = 1; rk <= n; rk++)
    {
        ret[rk - 1] = new double[rk + 1];
        ret[rk - 1][0] = alpha;
        for (int r = 1; r <= rk; r++)
        {
            k = rk - r;
            ret[rk - 1][r] = 1 - alpha * r / (k * h + r);
        }
    }
    return ret;
}