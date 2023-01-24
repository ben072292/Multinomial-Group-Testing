#pragma once
#include "../../core.hpp"
#include "../../product_lattice_model/product_lattice.hpp"
#include "../../product_lattice_model/product_lattice_dilution.hpp"
#include "../../product_lattice_model/product_lattice_non_dilution.hpp"
#include "../util/tree_stat.hpp"

class Single_tree {
    protected: 
    Product_lattice* lattice_;
    int ex_, res_, curr_stage_;
    double branch_prob_;
    Single_tree** children_;
    bool is_clas_;

    public:
    Single_tree(Product_lattice* lattice, int ex, int res, int curr_stage);
    Single_tree(Product_lattice* lattice, int ex, int res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double** dilution);
    Single_tree(const Single_tree &other, bool deep);
    virtual ~Single_tree();
    void increase_stage(int k, double thres_up, double thres_lo, int stage);
    Product_lattice* lattice() const {return lattice_;}
    inline void lattice(Product_lattice* lattice){lattice_ = lattice;}
    inline int ex() const {return ex_;}
    int true_ex(int halving);
    inline int ex_res() const {return res_;}
    inline int ex_count() const {return lattice_->test_count();}
    inline int curr_stage() const {return curr_stage_;}
    inline int curr_subjs() const {return lattice_->curr_subjs();}
    inline int orig_subjs() const {return lattice_->orig_subjs();}
    inline int variants() const {return lattice_->variants();}
    inline double branch_prob() const {return branch_prob_;}
    inline Single_tree **children() const {return children_;}
    inline bool is_classified() const {return is_clas_;}
    void parse(int true_state, const Product_lattice* org_lattice, double* pi0, double thres_branch, double sym_coef, Tree_stat* ret) const;
    inline double total_positive() const {return __builtin_popcount(lattice_->pos_clas());}
    inline double total_negative() const {return __builtin_popcount(lattice_->neg_clas());}
    bool is_correct_clas(int true_state) const;
    double fp(int true_state) const;
    double fn(int true_state) const;
    static void find_all_leaves(const Single_tree* node, std::vector<const Single_tree*> *leaves);
    static void find_all_stat(const Single_tree* node, std::vector<const Single_tree*> *leaves, double thres_branch);
    static void find_clas_stat(const Single_tree* node, std::vector<const Single_tree*> *leaves, double thres_branch);
    static void find_unclas_stat(const Single_tree* node, std::vector<const Single_tree*> *leaves, double thres_branch);
    void apply_true_state(const Product_lattice* org_lattice, int true_state, double thres_branch, double** dilution);
    static void apply_true_state_helper(const Product_lattice* org_lattice, Single_tree* node, int true_state, double prob, double thres_branch, double** dilution);
    void sequence_analyzer();

};