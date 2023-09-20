#pragma once
#include "tree.hpp"

class Global_tree_intra : public Tree
{
public:
    Global_tree_intra() : Tree() {}
    Global_tree_intra(Product_lattice *lattice, bin_enc ex, bin_enc res, int curr_stage) : Tree(lattice, ex, res, curr_stage, 0.0) {} // must be initialized to 0.0 so that stat tree can generate correct info
    Global_tree_intra(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage);
    Global_tree_intra(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, std::chrono::nanoseconds halving_times[]);
    Global_tree_intra(const Tree &other, bool deep);
    virtual std::string type() override { return "Global Tree Serial"; }
};