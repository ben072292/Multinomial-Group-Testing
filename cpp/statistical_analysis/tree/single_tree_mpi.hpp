#pragma once
#include "single_tree.hpp"
// #include "mpi.h"

class Single_tree_mpi : public Single_tree{
    public:
    Single_tree_mpi(Product_lattice* lattice, int ex, int res, int curr_stage) : Single_tree(lattice, ex, res, curr_stage){}
    Single_tree_mpi(Product_lattice* lattice, int ex, int res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double** dilution, int rank, int world_size, MPI_Op* halving_op, MPI_Datatype* halving_res_type, Halving_res& halving_res);
    Single_tree_mpi(const Single_tree_mpi &other, bool deep) : Single_tree(other, deep){}

};