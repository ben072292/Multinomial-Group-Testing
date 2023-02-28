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

    int parallelism_type = std::atoi(argv[1]);
    int omp_enabled = std::atoi(argv[2]);
    int atom = std::atoi(argv[3]);
    int variant = std::atoi(argv[4]);

    double pi0[atom * variant];
    for (int i = 0; i < atom * variant; i++)
    {
        pi0[i] = 0.02;
    }

    Product_lattice *p;
    auto start_lattice_construction = std::chrono::high_resolution_clock::now();
    if (parallelism_type == MP_NON_DILUTION)
        p = new Product_lattice_mp_non_dilution(atom, variant, pi0);
    else if (parallelism_type == MP_DILUTION)
        p = new Product_lattice_mp_dilution(atom, variant, pi0);
    else if (parallelism_type == DP_NON_DILUTION)
        p = new Product_lattice_non_dilution(atom, variant, pi0);
    else if (parallelism_type == DP_DILUTION)
        p = new Product_lattice_dilution(atom, variant, pi0);
    else
        exit(1);

    auto end_lattice_construction = std::chrono::high_resolution_clock::now();
    auto start_halving = std::chrono::high_resolution_clock::now();

    if (omp_enabled)
        p->halving_hybrid(0.25);
    else
        p->halving_mpi(0.25);

    auto end_halving = std::chrono::high_resolution_clock::now();

    if (world_rank == 0)
    {
        std::stringstream file_name;
        file_name << "Multinomial-" << p->type()
                  << "-N=" << atom
                  << "-k=" << variant
                  << "-Processes=" << world_size
                  << "-Threads=" << omp_get_num_threads()
                  << "-" << get_curr_time()
                  << ".csv";
        freopen(file_name.str().c_str(), "w", stdout);

        std::cout << "Construction Time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_lattice_construction - start_lattice_construction).count() / 1e9 << "s\n";
        std::cout << "Halving Time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_halving - start_halving).count() / 1e9 << "s\n";
    }

    // Free product lattice MPI env
    Product_lattice::MPI_Product_lattice_Free();

    // Finalize the MPI environment.
    MPI_Finalize();
}
