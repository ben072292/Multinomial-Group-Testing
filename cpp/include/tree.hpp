#pragma once
#include "core.hpp"
#include "product_lattice.hpp"
#include "tree_perf.hpp"
#include "tree_stat.hpp"

class Tree
{
protected:
    bin_enc _ex, _res;
    int _curr_stage;
    double _branch_prob;
    Product_lattice *_lattice;
    Tree **_children;
    static int _search_depth;
    static double _thres_up, _thres_lo, _thres_branch;
    static double **_dilution;

public:
    Tree(){};
    Tree(Product_lattice *lattice, bin_enc ex, bin_enc res, int curr_stage, double prob);
    Tree(const Tree &other, bool deep);
    virtual ~Tree();
    Product_lattice *lattice() const { return _lattice; }
    inline void lattice(Product_lattice *lattice) { _lattice = lattice; }
    inline bin_enc ex() const { return _ex; }
    bin_enc true_ex(bin_enc halving);
    inline bin_enc ex_res() const { return _res; }
    inline int curr_stage() const { return _curr_stage; }
    inline int curr_subjs() const { return _lattice->curr_subjs(); }
    inline static int variants() { return Product_lattice::variants(); }
    inline double branch_prob() const { return _branch_prob; }
    inline void branch_prob(double prob) { _branch_prob = prob; }
    inline static double thres_up() { return _thres_up; }
    inline static void thres_up(double thres_up) { _thres_up = thres_up; }
    inline static double thres_lo() { return _thres_lo; }
    inline static void thres_lo(double thres_lo) { _thres_lo = thres_lo; }
    inline static double thres_branch() { return _thres_branch; }
    inline static void thres_branch(double thres_branch) { _thres_branch = thres_branch; }
    inline static int search_depth() { return _search_depth; }
    inline static void search_depth(int search_depth) { _search_depth = search_depth; }
    inline static double **dilution() { return _dilution; }
    inline static void dilution(double **dilution) { _dilution = dilution; }
    inline Tree **children() const { return _children; }
    inline void children(int num)
    {
        _children = new Tree *[num]
        { nullptr };
    }
    inline bool is_classified() const { return _lattice->is_classified(); }
    void parse(bin_enc true_state, const Product_lattice *org_lattice, double sym_coef, Tree_stat *ret) const;
    inline double total_positive() const { return __builtin_popcount(_lattice->pos_clas_atoms()); }
    inline double total_negative() const { return __builtin_popcount(_lattice->neg_clas_atoms()); }
    bool is_correct_clas(bin_enc true_state) const;
    double fp(bin_enc true_state) const;
    double fn(bin_enc true_state) const;
    static void find_all_leaves(const Tree *node, std::vector<const Tree *> *leaves);
    static void find_all_stat(const Tree *node, std::vector<const Tree *> *leaves);
    static void find_clas_stat(const Tree *node, std::vector<const Tree *> *leaves);
    static void find_unclas_stat(const Tree *node, std::vector<const Tree *> *leaves);
    void apply_true_state(const Product_lattice *org_lattice, bin_enc true_state);
    std::string shrinking_stat() const;
    virtual std::string type() { return "Base Tree"; }
    unsigned long size_estimator();
    inline void destroy_posterior_probs() { delete[] _lattice->posterior_probs(); _lattice->posterior_probs(nullptr);}
};