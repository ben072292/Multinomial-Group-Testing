#include "tree/single_tree.hpp"
#include "tree/single_tree_mpi.hpp"
#include "../product_lattice_model/product_lattice.hpp"
#include "../product_lattice_model/product_lattice_dilution.hpp"
#include "../product_lattice_model/product_lattice_non_dilution.hpp"
#include <chrono>
#include "/opt/homebrew/Cellar/open-mpi/4.1.4_2/include/mpi.h"

void halving_reduce(double* in, double* inout, int* len, MPI_Datatype *dptr){
    if(in[0] < inout[0]){
        inout[0] = in[0];
        inout[1] = in[1];
    }
}

int main(int argc, char* argv[]){
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

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

    MPI_Op halving_op;

    MPI_Op_create((MPI_User_function*)&halving_reduce, true, &halving_op);


    int atom = std::atoi(argv[1]);
    int variant = std::atoi(argv[2]);
    double prior = std::atof(argv[3]);
    double thres_up = 0.005;
    double thres_lo = 0.005;
    double thres_branch = 0.001;
    int search_depth = 5;

    double pi0[atom * variant];
    for(int i = 0; i < atom * variant; i++){
        pi0[i] = prior;
    }

    Product_lattice* p = new Product_lattice_dilution(atom, variant, pi0);

    double** dilution = p->generate_dilution(0.99, 0.005);

    // Single_tree* tree = new Single_tree(p, -1, -1, 1, 0, thres_up, thres_lo, search_depth, dilution);

    double run_time1 = 0.0 - MPI_Wtime();
    double run_time2 = 0.0 - MPI_Wtime();

    double * halving_res = new double[2];
    halving_res = p->halving(0.25, world_rank, world_size);

    run_time1 += MPI_Wtime();
    std::cout << "Time Consumption: " << run_time1 << "s" << std::endl;

    MPI_Allreduce(MPI_IN_PLACE, halving_res, 2, MPI_DOUBLE, halving_op, MPI_COMM_WORLD);

    run_time2 += MPI_Wtime();
    

    if(world_rank == 0){
        std::cout << "Overall Time Consumption: " << run_time2 << "s" << std::endl;
    }

    // MPI_Op_free(&halving_op);
    // Finalize the MPI environment.
    MPI_Finalize();
}
