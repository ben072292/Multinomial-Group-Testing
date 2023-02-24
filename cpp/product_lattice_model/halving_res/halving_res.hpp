#pragma once
#include "../../core.hpp"

typedef struct Halving_res
{
    double min;
    bin_enc candidate;

    Halving_res(double val1 = 2.0, bin_enc val2 = -1)
    {
        min = val1;
        candidate = val2;
    }

    inline void reset() { min = 2.0, candidate = -1; }

    inline static void create_halving_res_type(MPI_Datatype *halving_res_type)
    {
        int lengths[2] = {1, 1};

        // Calculate displacements
        // In C, by default padding can be inserted between fields. MPI_Get_address will allow
        // to get the address of each struct field and calculate the corresponding displacement
        // relative to that struct base address. The displacements thus calculated will therefore
        // include padding if any.
        MPI_Aint displacements[2];
        struct Halving_res dummy_halving_res;
        MPI_Aint base_address;
        MPI_Get_address(&dummy_halving_res, &base_address);
        MPI_Get_address(&dummy_halving_res.min, &displacements[0]);
        MPI_Get_address(&dummy_halving_res.candidate, &displacements[1]);
        displacements[0] = MPI_Aint_diff(displacements[0], base_address);
        displacements[1] = MPI_Aint_diff(displacements[1], base_address);

        MPI_Datatype types[2] = {MPI_DOUBLE, MPI_INT};
        MPI_Type_create_struct(2, lengths, displacements, types, halving_res_type);
    }

    inline static void halving_reduce(Halving_res *in, Halving_res *inout, int *len, MPI_Datatype *dptr)
    {
        if (in->min < inout->min)
        {
            inout->min = in->min;
            inout->candidate = in->candidate;
        }
    }

    static void halving_min(Halving_res &a, Halving_res &b)
    {
        if (a.min > b.min)
        {
            a.min = b.min;
            a.candidate = b.candidate;
        }
    }
} Halving_res;