#pragma once

#include "../../core.hpp"
class Tree_perf
{
private:
    int _atoms, _variants;
    int **_count;
    std::chrono::nanoseconds *_halving_time;
    std::chrono::nanoseconds **_update_time;
    std::chrono::nanoseconds **_mp_time;
    std::chrono::nanoseconds **_dp_time;
    std::chrono::nanoseconds **_mp_dp_time;

public:
    Tree_perf(int atoms) : _atoms(atoms)
    {
        _halving_time = new std::chrono::nanoseconds[atoms + 1]{std::chrono::nanoseconds::zero()};
        _update_time = new std::chrono::nanoseconds *[atoms + 1];
        _count = new int *[atoms + 1];
        _mp_time = new std::chrono::nanoseconds *[atoms + 1];
        _dp_time = new std::chrono::nanoseconds *[atoms + 1];
        _mp_dp_time = new std::chrono::nanoseconds *[atoms + 1];
        for (int i = 0; i <= atoms; i++)
        {
            _count[i] = new int[atoms + 1]{0};
            _update_time[i] = new std::chrono::nanoseconds[atoms + 1]{std::chrono::nanoseconds::zero()};
            _mp_time[i] = new std::chrono::nanoseconds[atoms + 1]{std::chrono::nanoseconds::zero()};
            _dp_time[i] = new std::chrono::nanoseconds[atoms + 1]{std::chrono::nanoseconds::zero()};
            _mp_dp_time[i] = new std::chrono::nanoseconds[atoms + 1]{std::chrono::nanoseconds::zero()};
        }
    }

    ~Tree_perf()
    {
        for (int i = 0; i < _atoms; i++)
        {
            delete[] _update_time[i];
            delete[] _count[i];
            delete[] _mp_time[i];
            delete[] _dp_time[i];
            delete[] _mp_dp_time[i];
        }
        delete[] _halving_time;
        delete[] _update_time;
        delete[] _count;
        delete[] _mp_time;
        delete[] _dp_time;
        delete[] _mp_dp_time;
    }

    inline int count(int prev, int curr) { return _count[prev][curr]; }
    inline int per_atom_counts(int atom)
    {
        int ret = 0;
        for (int i = 0; i <= _atoms; i++)
        {
            ret += _count[atom][i];
        }
        return ret;
    }
    inline int total_counts()
    {
        int ret = 0;
        for (int i = 0; i <= _atoms; i++)
        {
            ret += per_atom_counts(i);
        }
        return ret;
    }
    inline std::chrono::nanoseconds per_atom_halving_time(int atom)
    {
        return _halving_time[atom];
    }
    inline std::chrono::nanoseconds total_halving_time()
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _atoms; i++)
        {
            ret += per_atom_halving_time(i);
        }
        return ret;
    }
    inline std::chrono::nanoseconds update_time(int prev, int curr) { return _update_time[prev][curr]; }
    inline std::chrono::nanoseconds per_atom_update_time(int atom)
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _atoms; i++)
        {
            ret += _update_time[atom][i];
        }
        return ret;
    }
    inline std::chrono::nanoseconds total_update_time()
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _atoms; i++)
        {
            ret += per_atom_update_time(i);
        }
        return ret;
    }
    inline std::chrono::nanoseconds mp_time(int prev, int curr) { return _mp_time[prev][curr]; }
    inline std::chrono::nanoseconds per_atom_mp_time(int atom)
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _atoms; i++)
        {
            ret += _mp_time[atom][i];
        }
        return ret;
    }

    inline std::chrono::nanoseconds total_mp_time()
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _atoms; i++)
        {
            ret += per_atom_mp_time(i);
        }
        return ret;
    }
    inline std::chrono::nanoseconds dp_time(int prev, int curr) { return _dp_time[prev][curr]; }
    inline std::chrono::nanoseconds per_atom_dp_time(int atom)
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _atoms; i++)
        {
            ret += _dp_time[atom][i];
        }
        return ret;
    }
    inline std::chrono::nanoseconds total_dp_time()
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _atoms; i++)
        {
            ret += per_atom_dp_time(i);
        }
        return ret;
    }
    inline std::chrono::nanoseconds mp_dp_time(int prev, int curr) { return _mp_dp_time[prev][curr]; }
    inline std::chrono::nanoseconds per_atom_mp_dp_time(int atom)
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _atoms; i++)
        {
            ret += _mp_dp_time[atom][i];
        }
        return ret;
    }
    inline std::chrono::nanoseconds total_mp_dp_time()
    {
        std::chrono::nanoseconds ret = std::chrono::nanoseconds::zero();
        for (int i = 0; i <= _atoms; i++)
        {
            ret += per_atom_mp_dp_time(i);
        }
        return ret;
    }

    inline void accumulate_count(int prev, int curr) { _count[prev][curr]++; }
    inline void accumulate_update_time(int prev, int curr, std::chrono::nanoseconds val) { _update_time[prev][curr] += val; }
    inline void accumulate_halving_time(int atom, std::chrono::nanoseconds val) { _halving_time[atom] += val; }
    inline void accumulate_mp_time(int prev, int curr, std::chrono::nanoseconds val) { _mp_time[prev][curr] += val; }
    inline void accumulate_dp_time(int prev, int curr, std::chrono::nanoseconds val) { _dp_time[prev][curr] += val; }
    inline void accumulate_mp_dp_time(int prev, int curr, std::chrono::nanoseconds val) { _mp_dp_time[prev][curr] += val; }
    void output()
    {
        std::cout << "Previous Subjects,Current Subjects,Count,Halving Time,Update Time,MP Time,DP Time,MP-DP Time,Count Percentage,Halving Percentage,Update Percentage,MP Percentage,DP Percentage,MP-DP Percentage" << std::endl;
        for (int i = 1; i <= _atoms; i++)
        {
            std::cout << i << ",,"
                      << per_atom_counts(i) << ","
                      << per_atom_halving_time(i).count()/1e9 << ","
                      << per_atom_update_time(i).count()/1e9 << ","
                      << per_atom_mp_time(i).count()/1e9 << ","
                      << per_atom_dp_time(i).count()/1e9 << ","
                      << per_atom_mp_dp_time(i).count()/1e9 << ","
                      << ((double)per_atom_counts(i) / (double)total_counts() * 100) << "%,"
                      << ((double)per_atom_halving_time(i).count() / total_halving_time().count() * 100) << "%,"
                      << ((double)per_atom_update_time(i).count() / total_update_time().count() * 100) << "%,"
                      << ((double)per_atom_mp_time(i).count() / total_mp_time().count() * 100) << "%,"
                      << ((double)per_atom_dp_time(i).count() / total_dp_time().count() * 100) << "%,"
                      << ((double)per_atom_mp_dp_time(i).count() / total_mp_dp_time().count() * 100) << "%\n";
        }
        std::cout << "Total,,"
                  << total_counts() << ","
                  << total_halving_time().count()/1e9 << ","
                  << total_update_time().count()/1e9 << ","
                  << total_mp_time().count()/1e9 << ","
                  << total_dp_time().count()/1e9 << ","
                  << total_mp_dp_time().count()/1e9 << ",100%,100%,100%,100%,100%,100%\n\n";
    }

    void output_verbose()
    {
        std::cout << "Previous Subjects,Current Subjects,Count,Halving Time,Update Time,MP Time,DP Time,MP-DP Transition Time,Count Percentage,Halving Percentage,Update Percentage,MP Percentage,DP Percentage,MP-DP Percentage" << std::endl;
        for (int i = _atoms; i >= 0; i--)
        {
            for (int j = i; j >= 0; j--)
            {
                std::cout << i << "," << j << ","
                          << _count[i][j] << ",,"
                          << _update_time[i][j].count()/1e9 << ","
                          << _mp_time[i][j].count()/1e9 << ","
                          << _dp_time[i][j].count()/1e9 << ","
                          << _mp_dp_time[i][j].count()/1e9 << ","
                          << ((double)_count[i][j] / total_counts() * 100) << "%,,"
                          << ((double)_update_time[i][j].count() / total_update_time().count() * 100) << "%,"
                          << ((double)_mp_time[i][j].count() / total_mp_time().count() * 100) << "%,"
                          << ((double)_dp_time[i][j].count() / total_dp_time().count() * 100) << "%,"
                          << ((double)_mp_dp_time[i][j].count() / total_mp_dp_time().count() * 100) << "%\n";
            }
            std::cout << "Subtotal,,"
                      << per_atom_counts(i) << ","
                      << per_atom_halving_time(i).count()/1e9 << ","
                      << per_atom_update_time(i).count()/1e9 << ","
                      << per_atom_mp_time(i).count()/1e9 << ","
                      << per_atom_dp_time(i).count()/1e9 << ","
                      << per_atom_mp_dp_time(i) .count()/1e9<< ","
                      << ((double)per_atom_counts(i) / total_counts() * 100) << "%,"
                      << ((double)per_atom_halving_time(i).count() / total_halving_time().count() * 100) << "%,"
                      << ((double)per_atom_update_time(i).count() / total_update_time().count() * 100) << "%,"
                      << ((double)per_atom_mp_time(i).count() / total_mp_time().count() * 100) << "%,"
                      << ((double)per_atom_dp_time(i).count() / total_dp_time().count() * 100) << "%,"
                      << ((double)per_atom_mp_dp_time(i).count() / total_mp_dp_time().count() * 100) << "%\n\n";
        }
        std::cout << "Total,,"
                  << total_counts() << ","
                  << total_halving_time().count()/1e9 << ","
                  << total_update_time().count()/1e9 << ","
                  << total_mp_time().count()/1e9 << ","
                  << total_dp_time().count()/1e9 << ","
                  << total_mp_dp_time().count()/1e9 << ",100%,100%,100%,100%,100%,100%\n\n";
        
        std::cout << "Total Halving Time," << total_halving_time().count()/1e9 << "s\n"
                  << "Total Update Time," << total_update_time().count()/1e9 << "s\n"
                  << "Total MP Time," << total_mp_time().count()/1e9 << "s\n" 
                  << "Total MP-DP Time," << total_mp_dp_time().count()/1e9 << "s\n"
                  << "Total DP Time," << total_dp_time().count()/1e9 << "s\n"
                  << "Total Shrinking Time," << (total_mp_time() + total_mp_dp_time() + total_dp_time()).count()/1e9 << "s\n"
                  << "Total Time," << (total_halving_time() + total_update_time() + total_mp_time() + total_mp_dp_time() + total_dp_time()).count()/1e9 << "s\n";
    }
};