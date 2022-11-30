#include "tree/single_tree.hpp"
#include "tree/single_tree_mpi.hpp"
#include "../product_lattice_model/product_lattice.hpp"
#include "../product_lattice_model/product_lattice_dilution.hpp"
#include "../product_lattice_model/product_lattice_non_dilution.hpp"
#include <chrono>
#include "mpi.h"
#include <omp.h>

void create_halving_res_type(MPI_Datatype* halving_res_type){
    int lengths[2] = { 1, 1 };
 
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

    MPI_Datatype types[2] = { MPI_DOUBLE, MPI_INT };
    MPI_Type_create_struct(2, lengths, displacements, types, halving_res_type);
}

void halving_reduce(Halving_res* in, Halving_res* inout, int* len, MPI_Datatype *dptr){
    if(in->min < inout->min){
        inout->min = in->min;
        inout->candidate = in->candidate;
    }
}

int main(int argc, char* argv[]){
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    // omp_set_num_threads(8);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Create the datatype
    MPI_Datatype halving_res_type;
    create_halving_res_type(&halving_res_type);
    MPI_Type_commit(&halving_res_type);

    MPI_Op halving_op;
    MPI_Op_create((MPI_User_function*)&halving_reduce, true, &halving_op);


    int atom = std::atoi(argv[1]);
    int variant = std::atoi(argv[2]);
    double prior = std::atof(argv[3]);

    double pi0[atom * variant];
    for(int i = 0; i < atom * variant; i++){
        pi0[i] = prior;
    }

    Product_lattice* p = new Product_lattice_dilution(atom, variant, pi0);

    double run_time1 = 0.0 - MPI_Wtime();
    double run_time2 = 0.0 - MPI_Wtime();

    Halving_res halving_res;   
    p->halving(0.25, world_rank, world_size, halving_res);

    run_time1 += MPI_Wtime();
    std::cout << "Time Consumption: " << run_time1 << "s" << std::endl;

    MPI_Allreduce(MPI_IN_PLACE, &halving_res, 2, MPI_DOUBLE, halving_op, MPI_COMM_WORLD);

    run_time2 += MPI_Wtime();
    

    if(world_rank == 0){
        std::cout << "Overall Time Consumption: " << run_time2 << "s" << std::endl;
    }

    // MPI_Op_free(&halving_op);
    // Finalize the MPI environment.
    MPI_Finalize();
}
