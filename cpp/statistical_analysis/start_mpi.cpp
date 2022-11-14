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

    double run_time = 0.0 - MPI_Wtime();

    Product_lattice* p = new Product_lattice_dilution(atom, variant, pi0);

    double** dilution = p->generate_dilution(0.99, 0.005);

    // Single_tree* tree = new Single_tree(p, -1, -1, 1, 0, thres_up, thres_lo, search_depth, dilution);

    Single_tree* tree = new Single_tree_mpi(p, -1, -1, 1, 0, thres_up, thres_lo, search_depth, dilution, world_rank, world_size, &halving_op);

    // std::vector<const Single_tree*> *leaves = new std::vector<const Single_tree*>;
    // Single_tree::find_all(tree, leaves);
    // std::cout << leaves->size() << std::endl;
    
    if(world_rank == 0){
        tree->apply_true_state(p, 0, thres_branch, dilution);

        Tree_stat* prim = new Tree_stat(search_depth, 1);
        Tree_stat* temp = new Tree_stat(search_depth, 1);
        tree->parse(0, p, pi0, thres_branch, 1.0, prim);

        int total_st = p->total_state();
        for(int i = 0; i < total_st; i++){
            tree->apply_true_state(p, i, 0.001, dilution);
            tree->parse(i, p, pi0, thres_branch, 1.0, temp);
            prim->merge(temp);
        }

        std::cout << "N = " << atom << ", k = " << variant << std::endl;
        std::cout << "Prior: ";
        for (int i = 0; i < p->nominal_pool_size(); i++){
            std::cout << pi0[i] << ", ";
        }
        std::cout << std::endl;
        prim->output_detail(p->nominal_pool_size());

        // clean up memory
        delete prim;
        delete temp;
    }

    for(int i = 0; i < atom; i++){
        delete[] dilution[i];
    }
    delete[] dilution;

    delete tree;

    MPI_Barrier(MPI_COMM_WORLD);
    run_time += MPI_Wtime();

    if(world_rank == 0){
        std::cout << "Time Consumption: " << run_time << "s" << std::endl;
    }

    MPI_Op_free(&halving_op);
    // Finalize the MPI environment.
    MPI_Finalize();
}
