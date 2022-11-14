#pragma once
#include "single_tree.hpp"
#include "/opt/homebrew/Cellar/open-mpi/4.1.4_2/include/mpi.h"

class Single_tree_mpi : public Single_tree{
    public:
    Single_tree_mpi(Product_lattice* lattice, int ex, int res, int cur_stage) : Single_tree(lattice, ex, res, cur_stage){}
    Single_tree_mpi(Product_lattice* lattice, int ex, int res, int k, int cur_stage, double thres_up, double thres_lo, int stage, double** dilution, int rank, int world_size, MPI_Op* halving_op);
    Single_tree_mpi(const Single_tree_mpi &other, bool deep) : Single_tree(other, deep){}

};