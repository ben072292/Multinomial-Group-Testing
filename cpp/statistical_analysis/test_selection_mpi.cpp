#include "../core.hpp"
#include "tree/global_tree.hpp"
#include "tree/global_tree_mpi.hpp"
#include "../product_lattice_model/product_lattice.hpp"
#include "../product_lattice_model/product_lattice_dilution.hpp"
#include "../product_lattice_model/product_lattice_non_dilution.hpp"

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

    // Initialize product lattice MPI env
    Product_lattice::MPI_Product_lattice_Initialize();


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

    p->halving_hybrid(0.25);

    run_time1 += MPI_Wtime();
    std::cout << "Time Consumption: " << run_time1 << "s" << std::endl;

    run_time2 += MPI_Wtime();
    

    if(world_rank == 0){
        std::cout << "Overall Time Consumption: " << run_time2 << "s" << std::endl;
    }

    // Free product lattice MPI env
    Product_lattice::MPI_Product_lattice_Free();

    // Finalize the MPI environment.
    MPI_Finalize();
}
