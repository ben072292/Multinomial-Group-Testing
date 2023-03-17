#pragma once
#include "global_tree.hpp"
// #include "mpi.h"

class Global_tree_mpi : public Global_tree
{
protected:
    static int rank, world_size;
    static Global_tree_mpi** sequence_tracer;
public:
    Global_tree_mpi(Product_lattice *lattice, bin_enc ex, bin_enc res, int curr_stage) : Global_tree(lattice, ex, res, curr_stage) {}
    Global_tree_mpi(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double **dilution);
    // Recoding total mpi time
    Global_tree_mpi(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double **dilution, std::chrono::nanoseconds halving_times[], std::chrono::nanoseconds mp_update_times[], std::chrono::nanoseconds dp_update_times[], std::chrono::nanoseconds mp_dp_update_times[]);
    // Fusion Tree
    Global_tree_mpi(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double* pi0, double **dilution, std::chrono::nanoseconds halving_times[], std::chrono::nanoseconds mp_update_times[], std::chrono::nanoseconds dp_update_times[], std::chrono::nanoseconds mp_dp_update_times[]);
    Global_tree_mpi(const Global_tree_mpi &other, bool deep) : Global_tree(other, deep) {}

    double fusion_branch_prob(int ex, int res, double* pi0, double** dilution);

    static MPI_Datatype tree_stat_type;
    static MPI_Op tree_stat_op;
    static void MPI_Global_tree_Initialize(int depth, int k);
    static void MPI_Global_tree_Free();
};