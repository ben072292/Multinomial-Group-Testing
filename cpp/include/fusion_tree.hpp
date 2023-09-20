#pragma once
#include "global_tree.hpp"

class Fusion_tree : public Global_tree
{
protected:
    static Fusion_tree **sequence_tracer;

public:
    Fusion_tree(Product_lattice *lattice, bin_enc ex, bin_enc res, int curr_stage) : Global_tree(lattice, ex, res, curr_stage) {}
    Fusion_tree(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double prun_thres_sum, double curr_prun_thres_sum, double prun_thres);
    Fusion_tree(const Tree &other, bool deep);
    double fusion_branch_prob(int ex, int res);
    virtual std::string type() override { return "Fusion Tree"; }
    static void MPI_Fusion_tree_Initialize(int subjs, int k);
    static void MPI_Fusion_tree_Free();
};