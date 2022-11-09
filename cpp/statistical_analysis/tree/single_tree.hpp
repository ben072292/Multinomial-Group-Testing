#pragma once
#include "../../product_lattice_model/product_lattice.hpp"
#include "../../product_lattice_model/product_lattice_dilution.hpp"
#include "../../product_lattice_model/product_lattice_non_dilution.hpp"
#include "../util/tree_stat.hpp"
#include <vector>

class Single_tree {
    protected: 
    Product_lattice* lattice_;
    int ex_, res_, ex_count_, cur_stage_, atom_, variant_;
    double branch_prob_;
    Single_tree** children_;
    bool is_clas_;
    int* clas_stat_;

    public:
    Single_tree(Product_lattice* lattice, int ex, int res, int cur_stage, bool prev_lattice);
    Single_tree(Product_lattice* lattice, int ex, int res, int k, int cur_stage, double thres_up, double thres_lo, int stage);
    Single_tree(const Single_tree &other, bool deep);
    ~Single_tree();
    void increase_stage(int k, double thres_up, double thres_lo, int stage);

    Product_lattice* lattice() const {return lattice_;}
    void lattice(Product_lattice* lattice){lattice_ = lattice;}
    int ex() const {return ex_;}
    void ex(int ex){ex_ = ex;}
    int ex_res() const {return res_;}
    void ex_res(int res){res_ = res;}
    int ex_count() const {return ex_count_;}
    void ex_count(int ex_count){ex_count_ = ex_count;}
    int cur_stage() const {return cur_stage_;}
    void cur_stage(int cur_stage){cur_stage_ = cur_stage;}
    int atom() const {return atom_;}
    void atom(int lattice_sz){atom_ = lattice_sz;}
    int variant() const {return variant_;}
    void variant(int variant) {variant_ = variant;}
    double branch_prob() const {return branch_prob_;}
    void branch_prob(double branch_prob){branch_prob_ = branch_prob;}
    Single_tree **children() const {return children_;}
    bool is_classified() const {return is_clas_;}
    void is_classified(bool is_clas) {is_clas_ = is_clas;}
    int* classified_stat() const {return clas_stat_;}
    void parse(int true_state, Product_lattice* org_lattice, double* pi0, double sym_coef, Tree_stat* ret) const;
    int actual_true_state() const;
    double total_positive() const;
    double total_negative() const;
    bool is_correct_clas(int true_state) const;
    double fp(int true_state) const;
    double fn(int true_state) const;
    static void find_all(const Single_tree* node, std::vector<const Single_tree*> *leaves);
    static void find_clas(const Single_tree* node, std::vector<const Single_tree*> *leaves);
    static void find_unclas(const Single_tree* node, std::vector<const Single_tree*> *leaves);
    Single_tree *apply_true_state(const Product_lattice* org_lattice, int true_state, double thres_branch) const;
    static void apply_true_state_helper(const Product_lattice* org_lattice, Single_tree* ret, const Single_tree* node, int true_state, double prob, double thres_branch, double** dilution);

};