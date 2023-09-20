#pragma once
#include "tree.hpp"

class Distributed_tree : public Tree
{
protected:
    static int rank, world_size;
    static Distributed_tree **backtrack;
    bin_enc _halving;

public:
    Distributed_tree(Product_lattice *lattice, bin_enc halving, bin_enc ex, bin_enc res, int curr_stage, double prob);
    Distributed_tree(Product_lattice *lattice, bin_enc halving, bin_enc ex, bin_enc res, int k, int curr_stage, int expansion_depth);
    Distributed_tree(const Tree &other, bool deep);
    inline bin_enc halving() const { return _halving; }
    inline void halving(bin_enc halving) { _halving = halving; }
    /* deprecated, use lazy_eval */
    static void eval(Tree *node, Product_lattice *orig_lattice, bin_enc true_state);
    static void lazy_eval(Tree *node, Product_lattice *orig_lattice, bin_enc true_state);
    static MPI_Datatype tree_stat_type;
    static MPI_Op tree_stat_op;
    static void MPI_Distributed_tree_Initialize(int subjs, int k, int search_depth);
    static void MPI_Distributed_tree_Free();
    virtual std::string type() override { return "Distributed Tree"; }
};