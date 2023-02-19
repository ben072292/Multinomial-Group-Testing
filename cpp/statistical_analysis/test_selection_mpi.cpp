#include "../core.hpp"
#include "../product_lattice_model/product_lattice.hpp"
#include "../product_lattice_model/product_lattice_dilution.hpp"
#include "../product_lattice_model/product_lattice_mp.hpp"
#include "../product_lattice_model/product_lattice_mp_dilution.hpp"
#include "../product_lattice_model/product_lattice_mp_non_dilution.hpp"
#include "../product_lattice_model/product_lattice_non_dilution.hpp"

int main(int argc, char *argv[])
{
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
    for (int i = 0; i < atom * variant; i++)
    {
        pi0[i] = prior;
    }

    Product_lattice_mp *p = new Product_lattice_mp_non_dilution(atom, variant, pi0);

    Product_lattice_mp *p1 = new Product_lattice_mp_non_dilution(*p, SHALLOW_COPY_PROB_DIST);

    std::cout << p1->curr_subjs() << std::endl;

    double run_time1 = 0.0 - MPI_Wtime();
    double run_time2 = 0.0 - MPI_Wtime();

    int selection = p->halving_hybrid(0.25);

    if (world_rank == 0)
    {
        std::cout << "Selection: " << selection << std::endl;
    }

    p->update_probs(15, 3, 0.01, 0.01, NULL);

    std::cout << p->posterior_probs()[0] << std::endl;

    run_time1 += MPI_Wtime();
    // std::cout << "Time Consumption: " << run_time1 << "s" << std::endl;

    run_time2 += MPI_Wtime();

    if (world_rank == 0)
    {
        std::cout << "Overall Time Consumption: " << run_time2 << "s" << std::endl;
    }

    // Free product lattice MPI env
    Product_lattice::MPI_Product_lattice_Free();

    // Finalize the MPI environment.
    MPI_Finalize();
}
