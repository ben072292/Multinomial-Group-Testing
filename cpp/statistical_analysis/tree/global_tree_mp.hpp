#pragma once
#include "global_tree.hpp"
// #include "mpi.h"

class Global_tree_mp : public Global_tree{
    public:
    Global_tree_mp(Product_lattice* lattice, bin_enc ex, bin_enc res, int curr_stage) : Global_tree(lattice, ex, res, curr_stage){}
    Global_tree_mp(Product_lattice* lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double** dilution);
    // Recoding total mpi time
    Global_tree_mp(Product_lattice* lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double** dilution, std::chrono::nanoseconds mpi_times[]);
    Global_tree_mp(const Global_tree_mp &other, bool deep) : Global_tree(other, deep){}

};