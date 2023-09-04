#include "tree_stat.hpp"

void Tree_stat::clear()
{
    for (int i = 0; i < 20; i++)
    {
        _correct[i] = 0.0;
        _incorrect[i] = 0.0;
        _fp[i] = 0.0;
        _fn[i] = 0.0;
    }
    _unclas = 0.0;
    _exp_stage = 0.0;
    _exp_test = 0.0;
    _stage_sd = 0.0;
    _test_sd = 0.0;
    _total_leaves = 0;
}
void Tree_stat::merge(Tree_stat *other)
{
    for (int i = 0; i <= _depth * _k; i++)
    {
        _correct[i] += other->_correct[i];
        _incorrect[i] += other->_incorrect[i];
        _fp[i] += other->_fp[i];
        _fn[i] += other->_fn[i];
    }
    _unclas += other->_unclas;
    _exp_stage += other->_exp_stage;
    _exp_test += other->_exp_test;
    _stage_sd += other->_stage_sd;
    _test_sd += other->_test_sd;
}

void Tree_stat::output_detail() const
{
    std::cout << "\n\nClassification Statistics: \n\n"
              << "Stagewise Statistics\n"
              << "Stage,Classification,FP,FN\n";

    double correctSum = 0, wrongSum = 0, fpTotal = 0, fnTotal = 0;
    for (int i = 0; i < _depth; i++)
    {
        double tempCorrectProbTotal = 0;
        double tempWrongProbTotal = 0;
        double tempFPTotal = 0;
        double tempFNTotal = 0;
        for (int j = 1; j <= _k; j++)
        {
            tempCorrectProbTotal += _correct[i * _k + j];
            tempWrongProbTotal += _incorrect[i * _k + j];
            tempFPTotal += _fp[i * _k + j];
            tempFNTotal += _fn[i * _k + j];
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
              << (1 - _unclas) * 100
              << " %,"
              << fpTotal * 100
              << " %," << fnTotal * 100 << " %\n\n"
              << std::endl;

    std::cout << "Potentially Can Be Classified in Future Stages:,"
              << _unclas * 100
              << " %"
              << std::endl;
    std::cout << "Total Probability Of Classified Sequences But Below The Branch Probability Threshold:,"
              << (1 - _unclas - (correctSum + wrongSum)) * 100
              << " %"
              << std::endl;
    std::cout << "Expected Average Number Of Classification Stages:," << _exp_stage << std::endl;
    std::cout << "Expected Average Number Of Classification Tests:," << _exp_test << std::endl;
    std::cout << "Standard Deviation For Number of Stages:," << _stage_sd << std::endl;
    std::cout << "Standard Deviation For Number of Tests:," << _test_sd << std::endl;
}

void Tree_stat::tree_stat_reduce(Tree_stat *in, Tree_stat *inout, int *len, MPI_Datatype *dptr)
{
    inout->merge(in);
}

void Tree_stat::create_tree_stat_type(MPI_Datatype *tree_stat_type, int depth, int k)
{
    int lengths[12] = {1, 1, 1, 1, 1, 1, 1, 1, 20, 20, 20, 20};

    // Calculate displacements
    // In C, by default padding can be inserted between fields. MPI_Get_address will allow
    // to get the address of each struct field and calculate the corresponding displacement
    // relative to that struct base address. The displacements thus calculated will therefore
    // include padding if any.
    MPI_Aint displacements[12];
    struct Tree_stat dummy_tree_stat(0, 0);
    MPI_Aint base_address;
    MPI_Get_address(&dummy_tree_stat, &base_address);
    MPI_Get_address(&dummy_tree_stat._depth, &displacements[0]);
    MPI_Get_address(&dummy_tree_stat._k, &displacements[1]);
    MPI_Get_address(&dummy_tree_stat._total_leaves, &displacements[2]);
    MPI_Get_address(&dummy_tree_stat._unclas, &displacements[3]);
    MPI_Get_address(&dummy_tree_stat._exp_stage, &displacements[4]);
    MPI_Get_address(&dummy_tree_stat._exp_test, &displacements[5]);
    MPI_Get_address(&dummy_tree_stat._stage_sd, &displacements[6]);
    MPI_Get_address(&dummy_tree_stat._test_sd, &displacements[7]);
    MPI_Get_address(&dummy_tree_stat._correct, &displacements[8]);
    MPI_Get_address(&dummy_tree_stat._incorrect, &displacements[9]);
    MPI_Get_address(&dummy_tree_stat._fp, &displacements[10]);
    MPI_Get_address(&dummy_tree_stat._fn, &displacements[11]);

    for (int i = 0; i < 12; i++)
    {
        displacements[i] = MPI_Aint_diff(displacements[i], base_address);
    }

    MPI_Datatype types[12] = {MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Type_create_struct(12, lengths, displacements, types, tree_stat_type);
}
