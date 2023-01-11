#include "tree/single_tree_mpi.hpp"
#include "../product_lattice_model/product_lattice_dilution.hpp"
#include "../product_lattice_model/product_lattice_non_dilution.hpp"
#include "../product_lattice_model/halving_res/halving_res.hpp"
#include <chrono>
#include <sstream>
#include <string>
#include "mpi.h"

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
    int search_depth = std::atoi(argv[4]);
    double thres_up = 0.005;
    double thres_lo = 0.005;
    double thres_branch = 0.001;

    double pi0[atom * variant];
    for(int i = 0; i < atom * variant; i++){
        pi0[i] = prior;
    }

    double run_time = 0.0 - MPI_Wtime();

    Product_lattice* p = new Product_lattice_dilution(atom, variant, pi0);

    double** dilution = p->generate_dilution(0.99, 0.005);

    Halving_res halving_res;
    Single_tree* tree = new Single_tree_mpi(p, -1, -1, 1, 0, thres_up, thres_lo, search_depth, dilution, world_rank, world_size, &halving_op, &halving_res_type, halving_res);

    // std::vector<const Single_tree*> *leaves = new std::vector<const Single_tree*>;
    // Single_tree::find_all(tree, leaves);
    // std::cout << leaves->size() << std::endl;
    
    if(world_rank == 0){
        // redirect stdout to file
        std::stringstream file_name;
        file_name << "Multinomial-N=" << atom << "-k=" + variant << "-Prior=" << prior << "-Depth=" << search_depth << ".csv";
        freopen(file_name.str().c_str(),"w",stdout);

        tree->apply_true_state(p, 0, thres_branch, dilution);

        Tree_stat* prim = new Tree_stat(search_depth, 1);
        Tree_stat* temp = new Tree_stat(search_depth, 1);

        int total_st = p->total_state();
        for(int i = 0; i < total_st; i++){
            tree->apply_true_state(p, i, 0.001, dilution);
            tree->parse(i, p, pi0, thres_branch, 1.0, temp);
            prim->merge(temp);
        }

        std::cout << "N = " << atom << ", k = " << variant << std::endl;
        std::cout << "Prior: ";
        for (int i = 0; i < p->curr_atoms(); i++){
            std::cout << pi0[i] << ", ";
        }
        std::cout << std::endl;
        prim->output_detail(p->curr_atoms());

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

    // Free datatype
    MPI_Type_free(&halving_res_type);
    // Free reduce op
    MPI_Op_free(&halving_op);
    // Finalize the MPI environment.
    MPI_Finalize();
}
