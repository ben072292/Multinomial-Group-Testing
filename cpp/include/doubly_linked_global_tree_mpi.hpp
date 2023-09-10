#pragma once
#include "global_tree_mpi.hpp"
// #include "mpi.h"

class Doubly_linked_global_tree_mpi : public Global_tree_mpi
{
protected:
    Global_tree *_parent;

public:
    Global_tree *parent() const { return _parent; }
    Doubly_linked_global_tree_mpi(Product_lattice *lattice, bin_enc ex, bin_enc res, int curr_stage, Global_tree *parent) : Global_tree_mpi(lattice, ex, res, curr_stage) { _parent = parent; }
    Doubly_linked_global_tree_mpi(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double **dilution, Global_tree *parent);
    // Recoding total mpi time
    Doubly_linked_global_tree_mpi(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double **dilution, bool perf, Global_tree *parent);
    // Fusion Tree
    Doubly_linked_global_tree_mpi(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double *pi0, double **__restrict__ dilution, double prun_thres_sum, double curr_prun_thres_sum, double prun_thres, Global_tree *parent);
    Doubly_linked_global_tree_mpi(const Doubly_linked_global_tree_mpi &other, bool deep) : Global_tree_mpi(other, deep) { _parent = other.parent(); }
};