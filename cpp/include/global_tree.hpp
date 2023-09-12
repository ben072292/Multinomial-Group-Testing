#pragma once
#include "tree.hpp"

class Global_tree : public Tree
{
public:
    Global_tree() : Tree() {}
    Global_tree(Product_lattice *lattice, bin_enc ex, bin_enc res, int curr_stage) : Tree(lattice, ex, res, curr_stage, 0.0) {} // must be initialized to 0.0 so that stat tree can generate correct info
    Global_tree(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage);
    Global_tree(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, std::chrono::nanoseconds halving_times[]);
    Global_tree(const Tree &other, bool deep);
    virtual std::string type() override { return "Global Tree Serial"; }
};