#pragma once
#include "global_tree.hpp"
// #include "mpi.h"

class Global_tree_mpi : public Global_tree{
    public:
    Global_tree_mpi(Product_lattice* lattice, bin_enc ex, bin_enc res, int curr_stage) : Global_tree(lattice, ex, res, curr_stage){}
    Global_tree_mpi(Product_lattice* lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double** dilution, int rank, int world_size, MPI_Op* halving_op, MPI_Datatype* halving_res_type, Halving_res& halving_res);
    // recoding total mpi time
    Global_tree_mpi(Product_lattice* lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double** dilution, int rank, int world_size, MPI_Op* halving_op, MPI_Datatype* halving_res_type, Halving_res& halving_res, std::chrono::microseconds* global_time);
    Global_tree_mpi(const Global_tree_mpi &other, bool deep) : Global_tree(other, deep){}

};