#pragma once

class Tree_stat
{
private:
    int depth_, k_, total_leaves_;
    double *correct_, *incorrect_, *fp_, *fn_,
        unclas_, exp_stage_, exp_test_, stage_sd_, test_sd_;

public:
    Tree_stat(int depth, int k)
    {
        depth_ = depth;
        k_ = k;
        correct_ = new double[depth * k + 1];
        incorrect_ = new double[depth * k + 1];
        fp_ = new double[depth * k + 1];
        fn_ = new double[depth * k + 1];
        clear();
    }

    ~Tree_stat()
    {
        delete[] correct_;
        delete[] incorrect_;
        delete[] fp_;
        delete[] fn_;
    }

    void clear();
    int depth() const { return depth_; }
    int k() const { return k_; }
    double *correct() const { return correct_; }
    double *incorrect() const { return incorrect_; }
    double *fp() const { return fp_; }
    double *fn() const { return fn_; }
    double unclassified() const { return unclas_; }
    void unclassified(double val) { unclas_ = val; }
    double expected_stage() const { return exp_stage_; }
    void expected_stage(double val) { exp_stage_ = val; }
    double expected_test() const { return exp_test_; }
    void expected_test(double val) { exp_test_ = val; }
    double stage_sd() const { return stage_sd_; }
    void stage_sd(double val) { stage_sd_ = val; }
    double test_sd() const { return test_sd_; }
    void test_sd(double val) { test_sd_ = val; }
    int total_leaves() const { return total_leaves_; }
    void total_leaves(int val) { total_leaves_ = val; }
    void merge(Tree_stat *other);
    void output_detail() const;
};