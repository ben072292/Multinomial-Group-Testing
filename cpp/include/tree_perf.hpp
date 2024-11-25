#pragma once
#include "common.hpp"

class Tree_perf
{
private:
    int _subjs, _variants;
    int **_count;
    std::chrono::nanoseconds *_BBPA_time;
    int * _BBPA_count;
    std::chrono::nanoseconds **_update_time;
    std::chrono::nanoseconds **_classification_identification_time;
    std::chrono::nanoseconds **_parallel_shrinking_time;
    std::chrono::nanoseconds **_serial_shrinking_time;
    std::chrono::nanoseconds **_parallel_to_serial_transition_time;

public:
    Tree_perf(int subjs) : _subjs(subjs)
    {
        _BBPA_time = new std::chrono::nanoseconds[subjs + 1]{std::chrono::nanoseconds::zero()};
        _update_time = new std::chrono::nanoseconds *[subjs + 1];
        _classification_identification_time = new std::chrono::nanoseconds *[subjs + 1];
        _count = new int *[subjs + 1];
        _BBPA_count = new int[subjs+1]{0};
        _parallel_shrinking_time = new std::chrono::nanoseconds *[subjs + 1];
        _serial_shrinking_time = new std::chrono::nanoseconds *[subjs + 1];
        _parallel_to_serial_transition_time = new std::chrono::nanoseconds *[subjs + 1];
        for (int i = 0; i <= subjs; i++)
        {
            _count[i] = new int[subjs + 1]{0};
            _update_time[i] = new std::chrono::nanoseconds[subjs + 1]{std::chrono::nanoseconds::zero()};
            _classification_identification_time[i] = new std::chrono::nanoseconds[subjs + 1]{std::chrono::nanoseconds::zero()};
            _parallel_shrinking_time[i] = new std::chrono::nanoseconds[subjs + 1]{std::chrono::nanoseconds::zero()};
            _serial_shrinking_time[i] = new std::chrono::nanoseconds[subjs + 1]{std::chrono::nanoseconds::zero()};
            _parallel_to_serial_transition_time[i] = new std::chrono::nanoseconds[subjs + 1]{std::chrono::nanoseconds::zero()};
        }
    }

    ~Tree_perf()
    {
        for (int i = 0; i <= _subjs; i++)
        {
            delete[] _update_time[i];
            delete[] _classification_identification_time[i];
            delete[] _count[i];
            delete[] _parallel_shrinking_time[i];
            delete[] _serial_shrinking_time[i];
            delete[] _parallel_to_serial_transition_time[i];
        }
        delete[] _BBPA_time;
        delete[] _BBPA_count;
        delete[] _update_time;
        delete[] _classification_identification_time;
        delete[] _count;
        delete[] _parallel_shrinking_time;
        delete[] _serial_shrinking_time;
        delete[] _parallel_to_serial_transition_time;
    }

    inline int count(int prev, int curr) { return _count[prev][curr]; }
    inline int per_subj_counts(int subj)
    {
        int ret = 0;
        for (int i = 0; i <= _subjs; i++)
        {
            ret += _count[subj][i];
        }
        return ret;
    }
    inline int total_counts()
    {
        int ret = 0;
        for (int i = 0; i <= _subjs; i++)
        {
            ret += per_subj_counts(i);
        }
        return ret;
    }
    inline std::chrono::nanoseconds per_subj_BBPA_time(int subj)
    {
        return _BBPA_time[subj];
    }
    inline std::chrono::nanoseconds total_BBPA_time()
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _subjs; i++)
        {
            ret += per_subj_BBPA_time(i);
        }
        return ret;
    }
    inline std::chrono::nanoseconds update_time(int prev, int curr) { return _update_time[prev][curr]; }
    inline std::chrono::nanoseconds per_subj_update_time(int subj)
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _subjs; i++)
        {
            ret += _update_time[subj][i];
        }
        return ret;
    }
    inline std::chrono::nanoseconds total_update_time()
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _subjs; i++)
        {
            ret += per_subj_update_time(i);
        }
        return ret;
    }
    inline std::chrono::nanoseconds classification_identification_time(int prev, int curr) { return _classification_identification_time[prev][curr]; }
    inline std::chrono::nanoseconds per_subj_classification_identification_time(int subj)
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _subjs; i++)
        {
            ret += _classification_identification_time[subj][i];
        }
        return ret;
    }
    inline std::chrono::nanoseconds total_classification_identification_time()
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _subjs; i++)
        {
            ret += per_subj_classification_identification_time(i);
        }
        return ret;
    }
    inline std::chrono::nanoseconds parallel_shrinking_time(int prev, int curr) { return _parallel_shrinking_time[prev][curr]; }
    inline std::chrono::nanoseconds per_subj_parallel_shrinking_time(int subj)
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _subjs; i++)
        {
            ret += _parallel_shrinking_time[subj][i];
        }
        return ret;
    }

    inline std::chrono::nanoseconds total_parallel_shrinking_time()
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _subjs; i++)
        {
            ret += per_subj_parallel_shrinking_time(i);
        }
        return ret;
    }
    inline std::chrono::nanoseconds serial_shrinking_time(int prev, int curr) { return _serial_shrinking_time[prev][curr]; }
    inline std::chrono::nanoseconds per_subj_serial_shrinking_time(int subj)
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _subjs; i++)
        {
            ret += _serial_shrinking_time[subj][i];
        }
        return ret;
    }
    inline std::chrono::nanoseconds total_serial_shrinking_time()
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _subjs; i++)
        {
            ret += per_subj_serial_shrinking_time(i);
        }
        return ret;
    }
    inline std::chrono::nanoseconds parallel_to_serial_transition_time(int prev, int curr) { return _parallel_to_serial_transition_time[prev][curr]; }
    inline std::chrono::nanoseconds per_subj_parallel_to_serial_transition_time(int subj)
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _subjs; i++)
        {
            ret += _parallel_to_serial_transition_time[subj][i];
        }
        return ret;
    }
    inline std::chrono::nanoseconds total_parallel_to_serial_transition_time()
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _subjs; i++)
        {
            ret += per_subj_parallel_to_serial_transition_time(i);
        }
        return ret;
    }

    inline void accumulate_count(int prev, int curr) { _count[prev][curr]++; }
    inline void accumulate_update_time(int prev, int curr, std::chrono::nanoseconds val) { _update_time[prev][curr] += val; }
    inline void accumulate_classification_identification_time(int prev, int curr, std::chrono::nanoseconds val) { _classification_identification_time[prev][curr] += val; }
    inline void accumulate_BBPA_time(int subj, std::chrono::nanoseconds val) { _BBPA_time[subj] += val; _BBPA_count[subj]++;}
    inline void accumulate_parallel_shrinking_time(int prev, int curr, std::chrono::nanoseconds val) { _parallel_shrinking_time[prev][curr] += val; }
    inline void accumulate_serial_shrinking_time(int prev, int curr, std::chrono::nanoseconds val) { _serial_shrinking_time[prev][curr] += val; }
    inline void accumulate_parallel_to_serial_transition_time(int prev, int curr, std::chrono::nanoseconds val) { _parallel_to_serial_transition_time[prev][curr] += val; }
    void output()
    {
        std::cout << "Previous Subjects,Current Subjects,Count,BBPA Time,Update Time,Classification Identification Time,Parallel Shrinking Time,Serial Shrinking Time,Parallel to Serial Transition Time,Count Percentage,BBPA Percentage,Update Percentage,Classification Identification Percentage,Parallel Shrinking Percentage,Serial Shrinking Percentage,Parallel to Serial Transition Percentage" << std::endl;
        for (int i = 1; i <= _subjs; i++)
        {
            std::cout << i << ",,"
                      << per_subj_counts(i) << ","
                      << per_subj_BBPA_time(i).count()/1e9 << ","
                      << per_subj_update_time(i).count()/1e9 << ","
                      << per_subj_classification_identification_time(i).count()/1e9 << ","
                      << per_subj_parallel_shrinking_time(i).count()/1e9 << ","
                      << per_subj_serial_shrinking_time(i).count()/1e9 << ","
                      << per_subj_parallel_to_serial_transition_time(i).count()/1e9 << ","
                      << ((double)per_subj_counts(i) / (double)total_counts() * 100) << "%,"
                      << ((double)per_subj_BBPA_time(i).count() / total_BBPA_time().count() * 100) << "%,"
                      << ((double)per_subj_update_time(i).count() / total_update_time().count() * 100) << "%,"
                      << ((double)per_subj_classification_identification_time(i).count() / total_classification_identification_time().count() * 100) << "%,"
                      << ((double)per_subj_parallel_shrinking_time(i).count() / total_parallel_shrinking_time().count() * 100) << "%,"
                      << ((double)per_subj_serial_shrinking_time(i).count() / total_serial_shrinking_time().count() * 100) << "%,"
                      << ((double)per_subj_parallel_to_serial_transition_time(i).count() / total_parallel_to_serial_transition_time().count() * 100) << "%\n";
        }
        std::cout << "Total,,"
                  << total_counts() << ","
                  << total_BBPA_time().count()/1e9 << ","
                  << total_update_time().count()/1e9 << ","
                  << total_classification_identification_time().count()/1e9 << ","
                  << total_parallel_shrinking_time().count()/1e9 << ","
                  << total_serial_shrinking_time().count()/1e9 << ","
                  << total_parallel_to_serial_transition_time().count()/1e9 << ",100%,100%,100%,100%,100%,100%\n\n";
    }

    void output_perf_stat()
    {
        std::cout << "Previous Subjects,Current Subjects,Count,BBPA Time,Update Time,Classification Identification Time,Parallel Shrinking Time,Serial Shrinking Time,Parallel To Serial Transition Time,Count Percentage,BBPA Percentage,Update Percentage,Classification Identification Percentage,Parallel Shrinking Percentage,Serial Shrinking Percentage,Parallel To Serial Transition Percentage" << std::endl;
        for (int i = _subjs; i >= 0; i--)
        {
            for (int j = i; j >= 0; j--)
            {
                std::cout << i << "," << j << ","
                          << _count[i][j] << ",,"
                          << _update_time[i][j].count()/1e9 << ","
                          << _classification_identification_time[i][j].count()/1e9 << ","
                          << _parallel_shrinking_time[i][j].count()/1e9 << ","
                          << _serial_shrinking_time[i][j].count()/1e9 << ","
                          << _parallel_to_serial_transition_time[i][j].count()/1e9 << ","
                          << ((double)_count[i][j] / total_counts() * 100) << "%,,"
                          << ((double)_update_time[i][j].count() / total_update_time().count() * 100) << "%,"
                          << ((double)_classification_identification_time[i][j].count() / total_update_time().count() * 100) << "%,"
                          << ((double)_parallel_shrinking_time[i][j].count() / total_parallel_shrinking_time().count() * 100) << "%,"
                          << ((double)_serial_shrinking_time[i][j].count() / total_serial_shrinking_time().count() * 100) << "%,"
                          << ((double)_parallel_to_serial_transition_time[i][j].count() / total_parallel_to_serial_transition_time().count() * 100) << "%\n";
            }
            std::cout << "Subtotal,,"
                      << per_subj_counts(i) << " (BBPA: " << _BBPA_count[i] << "),"
                      << per_subj_BBPA_time(i).count()/1e9 << ","
                      << per_subj_update_time(i).count()/1e9 << ","
                      << per_subj_classification_identification_time(i).count()/1e9 << ","
                      << per_subj_parallel_shrinking_time(i).count()/1e9 << ","
                      << per_subj_serial_shrinking_time(i).count()/1e9 << ","
                      << per_subj_parallel_to_serial_transition_time(i) .count()/1e9<< ","
                      << ((double)per_subj_counts(i) / total_counts() * 100) << "%,"
                      << ((double)per_subj_BBPA_time(i).count() / total_BBPA_time().count() * 100) << "%,"
                      << ((double)per_subj_update_time(i).count() / total_update_time().count() * 100) << "%,"
                      << ((double)per_subj_classification_identification_time(i).count() / total_classification_identification_time().count() * 100) << "%,"
                      << ((double)per_subj_parallel_shrinking_time(i).count() / total_parallel_shrinking_time().count() * 100) << "%,"
                      << ((double)per_subj_serial_shrinking_time(i).count() / total_serial_shrinking_time().count() * 100) << "%,"
                      << ((double)per_subj_parallel_to_serial_transition_time(i).count() / total_parallel_to_serial_transition_time().count() * 100) << "%\n\n";
        }
        std::cout << "Total,,"
                  << total_counts() << ","
                  << total_BBPA_time().count()/1e9 << ","
                  << total_update_time().count()/1e9 << ","
                  << total_classification_identification_time().count()/1e9 << ","
                  << total_parallel_shrinking_time().count()/1e9 << ","
                  << total_serial_shrinking_time().count()/1e9 << ","
                  << total_parallel_to_serial_transition_time().count()/1e9 << ",100%,100%,100%,100%,100%,100%\n\n";
        
        std::cout << "Total BBPA Time," << total_BBPA_time().count()/1e9 << "s\n"
                  << "Total Update Time," << total_update_time().count()/1e9 << "s\n"
                  << "Total Classification Identification Time," << total_classification_identification_time().count()/1e9 << "s\n"
                  << "Total Parallel Shrinking Time," << total_parallel_shrinking_time().count()/1e9 << "s\n" 
                  << "Total Serial Shrinking Time," << total_serial_shrinking_time().count()/1e9 << "s\n"
                  << "Total Parallel To Serial Transition Time," << total_parallel_to_serial_transition_time().count()/1e9 << "s\n"
                  << "Total Shrinking Time," << (total_parallel_shrinking_time() + total_parallel_to_serial_transition_time() + total_serial_shrinking_time()).count()/1e9 << "s\n"
                  << "Total Time," << (total_BBPA_time() + total_update_time() + total_parallel_shrinking_time() + total_parallel_to_serial_transition_time() + total_serial_shrinking_time()).count()/1e9 << "s\n";
    }
};