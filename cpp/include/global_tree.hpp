#pragma once
#include "global_tree_intra.hpp"
// #include "mpi.h"

class Global_tree : public Global_tree_intra
{
protected:
    static int rank, world_size;

public:
    Global_tree() : Global_tree_intra() {}
    Global_tree(Product_lattice *lattice, bin_enc ex, bin_enc res, int curr_stage) : Global_tree_intra(lattice, ex, res, curr_stage) {}
    Global_tree(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage);
    // Recoding total mpi time
    Global_tree(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, bool perf);
    Global_tree(const Tree &other, bool deep);
    virtual std::string type() override { return "Global Tree"; }

    static Tree_perf *tree_perf;
    static MPI_Datatype tree_stat_type;
    static MPI_Op tree_stat_op;
    static void MPI_Global_tree_Initialize(int subjs, int k);
    static void MPI_Global_tree_Free();
};