#include "tree_trim.hpp"

static bool compare_indices(const std::pair<double, int> &a, const std::pair<double, int> &b)
{
    return a.first < b.first;
}

int *trim_true_states(double *values, int n, double percent, int &resultSize)
{
    // Create an array of pairs (value, index) to store original indices
    std::pair<double, int> *indexed_values = new std::pair<double, int>[n];
    for (int i = 0; i < n; ++i)
    {
        indexed_values[i] = std::make_pair(values[i], i);
    }
    // Sort the indexed values based on the actual values in ascending order
    std::sort(indexed_values, indexed_values + n, compare_indices);

    double threshold = percent / 100.0;
    int *remainingIndices = new int[n];
    resultSize = 0;
    double currentSum = 0.0;

    for (int i = 0; i < n; ++i)
    {
        if (currentSum + indexed_values[i].first > threshold)
        {
            remainingIndices[resultSize++] = indexed_values[i].second;
        }
        else
        {
            currentSum += indexed_values[i].first;
        }
    }

    delete[] indexed_values;
    return remainingIndices;
}
