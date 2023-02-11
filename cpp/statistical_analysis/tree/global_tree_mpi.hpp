#pragma once
#include "global_tree.hpp"
// #include "mpi.h"

class Global_tree_mpi : public Global_tree{
    public:
    Global_tree_mpi(Product_lattice* lattice, bin_enc ex, bin_enc res, int curr_stage) : Global_tree(lattice, ex, res, curr_stage){}
    Global_tree_mpi(Product_lattice* lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double** dilution);
    // Recoding total mpi time
    Global_tree_mpi(Product_lattice* lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double** dilution, std::chrono::nanoseconds mpi_times[]);
    Global_tree_mpi(const Global_tree_mpi &other, bool deep) : Global_tree(other, deep){}

};