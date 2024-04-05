#pragma once
#include "common.hpp"

struct Tree_stat
{
private:
    int _depth, _k, _total_leaves;
    double _unclas, _exp_stage, _exp_test, _stage_sd, _test_sd;
    double _correct[30];
    double _incorrect[30];
    double _fp[30];
    double _fn[30];

public:
    Tree_stat(int depth, int k)
    {
        _depth = depth;
        _k = k;
        clear();
    }

    ~Tree_stat(){}

    void clear();
    int depth() const { return _depth; }
    int k() const { return _k; }
    double *correct() { return _correct; }
    double *incorrect() { return _incorrect; }
    double *fp() { return _fp; }
    double *fn() { return _fn; }
    double unclassified() const { return _unclas; }
    void unclassified(double val) { _unclas = val; }
    double expected_stage() const { return _exp_stage; }
    void expected_stage(double val) { _exp_stage = val; }
    double expected_test() const { return _exp_test; }
    void expected_test(double val) { _exp_test = val; }
    double stage_sd() const { return _stage_sd; }
    void stage_sd(double val) { _stage_sd = val; }
    double test_sd() const { return _test_sd; }
    void test_sd(double val) { _test_sd = val; }
    int total_leaves() const { return _total_leaves; }
    void total_leaves(int val) { _total_leaves = val; }
    void merge(Tree_stat *other);
    void output_detail() const;

    static void tree_stat_reduce(Tree_stat* in, Tree_stat* inout, int* len, MPI_Datatype *dptr);
    static void create_tree_stat_type(MPI_Datatype *tree_stat_type, int stages, int k);

};