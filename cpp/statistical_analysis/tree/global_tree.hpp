#pragma once
#include "../../core.hpp"
#include "../../product_lattice_model/product_lattice.hpp"
#include "../../product_lattice_model/product_lattice_dilution.hpp"
#include "../../product_lattice_model/product_lattice_non_dilution.hpp"
#include "../util/tree_stat.hpp"
#include "../util/tree_perf.hpp"

class Global_tree
{
protected:
    Product_lattice *_lattice;
    bin_enc _ex, _res;
    int _curr_stage;
    double _branch_prob;
    Global_tree **_children;
    
public:
    Global_tree() {}
    Global_tree(Product_lattice *lattice, bin_enc ex, bin_enc res, int curr_stage);
    Global_tree(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double **dilution);
    Global_tree(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double **dilution, std::chrono::nanoseconds halving_times[]);
    Global_tree(const Global_tree &other, bool deep);
    virtual ~Global_tree();
    void increase_stage(int k, double thres_up, double thres_lo, int stage);
    Product_lattice *lattice() const { return _lattice; }
    inline void lattice(Product_lattice *lattice) { _lattice = lattice; }
    inline bin_enc ex() const { return _ex; }
    bin_enc true_ex(bin_enc halving);
    inline bin_enc ex_res() const { return _res; }
    inline int ex_count() const { return _lattice->test_count(); }
    inline int curr_stage() const { return _curr_stage; }
    inline int curr_subjs() const { return _lattice->curr_subjs(); }
    inline int variants() const { return _lattice->variants(); }
    inline double branch_prob() const { return _branch_prob; }
    inline Global_tree **children() const { return _children; }
    inline bool is_classified() const { return _lattice->is_classified();}
    void parse(bin_enc true_state, const Product_lattice *org_lattice, double *pi0, double thres_branch, double sym_coef, Tree_stat *ret) const;
    inline double total_positive() const { return __builtin_popcount(_lattice->pos_clas_atoms()); }
    inline double total_negative() const { return __builtin_popcount(_lattice->neg_clas_atoms()); }
    bool is_correct_clas(bin_enc true_state) const;
    double fp(bin_enc true_state) const;
    double fn(bin_enc true_state) const;
    static void find_all_leaves(const Global_tree *node, std::vector<const Global_tree *> *leaves);
    static void find_all_stat(const Global_tree *node, std::vector<const Global_tree *> *leaves, double thres_branch);
    static void find_clas_stat(const Global_tree *node, std::vector<const Global_tree *> *leaves, double thres_branch);
    static void find_unclas_stat(const Global_tree *node, std::vector<const Global_tree *> *leaves, double thres_branch);
    void apply_true_state(const Product_lattice *org_lattice, bin_enc true_state, double thres_branch, double **dilution);
    static void apply_true_state_helper(const Product_lattice *org_lattice, Global_tree *node, bin_enc true_state, double prob, double thres_branch, double **dilution);
    std::string shrinking_stat() const;
    
};