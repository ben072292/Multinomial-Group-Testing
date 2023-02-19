#include "tree_stat.hpp"
#include <iostream>

void Tree_stat::clear()
{
    for (int i = 0; i < depth() * k() + 1; i++)
    {
        correct_[i] = 0.0;
        incorrect_[i] = 0.0;
        fp_[i] = 0.0;
        fn_[i] = 0.0;
    }
    unclas_ = 0.0;
    exp_stage_ = 0.0;
    exp_test_ = 0.0;
    stage_sd_ = 0.0;
    test_sd_ = 0.0;
    total_leaves_ = 0;
}
void Tree_stat::merge(Tree_stat *other)
{
    for (int i = 0; i <= depth_ * k_; i++)
    {
        correct_[i] += other->correct_[i];
        incorrect_[i] += other->incorrect_[i];
        fp_[i] += other->fp_[i];
        fn_[i] += other->fn_[i];
    }
    unclas_ += other->unclas_;
    exp_stage_ += other->exp_stage_;
    exp_test_ += other->exp_test_;
    stage_sd_ += other->stage_sd_;
    test_sd_ += other->test_sd_;
}

void Tree_stat::output_detail() const
{
    std::cout << "\n\nClassification Statistics: \n\n"
              << "Stagewise Statistics\n"
              << "Stage,Classification,FP,FN\n";

    double correctSum = 0, wrongSum = 0, fpTotal = 0, fnTotal = 0;
    for (int i = 0; i < depth_; i++)
    {
        double tempCorrectProbTotal = 0;
        double tempWrongProbTotal = 0;
        double tempFPTotal = 0;
        double tempFNTotal = 0;
        for (int j = 1; j <= k_; j++)
        {
            tempCorrectProbTotal += correct_[i * k_ + j];
            tempWrongProbTotal += incorrect_[i * k_ + j];
            tempFPTotal += fp_[i * k_ + j];
            tempFNTotal += fn_[i * k_ + j];
        }
        correctSum += tempCorrectProbTotal;
        wrongSum += tempWrongProbTotal;
        fpTotal += tempFPTotal;
        fnTotal += tempFNTotal;

        std::cout << (i + 1)
                  << ","
                  << (tempCorrectProbTotal + tempWrongProbTotal) * 100
                  << " %,"
                  << tempFPTotal * 100
                  << " %,"
                  << tempFNTotal * 100
                  << " %"
                  << std::endl;
    }
    std::cout << "Total,"
              << (1 - unclas_) * 100
              << " %,"
              << fpTotal * 100
              << " %," << fnTotal * 100 << " %\n\n"
              << std::endl;

    std::cout << "Potentially Can Be Classified in Future Stages:,"
              << unclas_ * 100
              << " %"
              << std::endl;
    std::cout << "Total Probability Of Classified Sequences But With < .1 % Branch Probability:,"
              << (1 - unclas_ - (correctSum + wrongSum)) * 100
              << " %"
              << std::endl;
    std::cout << "Expected Average Number Of Classification Stages:," << exp_stage_ << std::endl;
    std::cout << "Expected Average Number Of Classification Tests:," << exp_test_ << std::endl;
    std::cout << "Standard Deviation For Number of Stages:," << stage_sd_ << std::endl;
    std::cout << "Standard Deviation For Number of Tests:," << test_sd_ << std::endl;
}
