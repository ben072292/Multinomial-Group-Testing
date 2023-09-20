#pragma once
#include "distributed_tree.hpp"

class Global_partial_tree : public Distributed_tree
{
public:
    Global_partial_tree(Product_lattice *lattice, bin_enc halving, bin_enc ex, bin_enc res, int curr_stage, double prob) : Distributed_tree(lattice, halving, ex, res, curr_stage, prob){}
    Global_partial_tree(const Tree &other, bool deep);
    /* Deprecated, much slower than lazy eval */
    static void eval(Tree *node, Product_lattice *orig_lattice, bin_enc true_state);
    static void lazy_eval(Tree *node, Product_lattice *orig_lattice, bin_enc true_state);
    virtual std::string type() override { return "Global Partial Tree"; }
};