#include "core.hpp"
#include "product_lattice.hpp"

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
    int variants = std::atoi(argv[4]);

    double pi0[atom * variants];
    for (int i = 0; i < atom * variants; i++)
    {
        pi0[i] = 0.1;
    }

    Product_lattice *p;
    auto start_lattice_construction = std::chrono::high_resolution_clock::now();
    if (parallelism_type == MP_NON_DILUTION)
        p = new Product_lattice_dist_non_dilution(atom, variants, pi0);
    else if (parallelism_type == MP_DILUTION)
        p = new Product_lattice_dist_dilution(atom, variants, pi0);
    else if (parallelism_type == DP_NON_DILUTION)
        p = new Product_lattice_non_dilution(atom, variants, pi0);
    else if (parallelism_type == DP_DILUTION)
        p = new Product_lattice_dilution(atom, variants, pi0);
    else
        exit(1);

    auto end_lattice_construction = std::chrono::high_resolution_clock::now();
    auto start_halving = std::chrono::high_resolution_clock::now();

    if (omp_enabled)
        p->BBPA_mpi_omp(1.0 / (1 << variants));
    else
        p->BBPA_mpi(1.0 / (1 << variants));

    auto end_halving = std::chrono::high_resolution_clock::now();

    if (world_rank == 0)
    {
        std::stringstream file_name;
        file_name << "TestSelection-" << p->type()
                  << "-N=" << atom
                  << "-k=" << variants
                  << "-Processes=" << world_size
                  << "-Threads=" << omp_get_num_threads()
                  << "-" << get_curr_time()
                  << ".csv";
        freopen(file_name.str().c_str(), "w", stdout);

        std::cout << "N," << atom 
                  << ",k," << variants 
                  << ",parallelism," << ((parallelism_type == MP_NON_DILUTION || parallelism_type == MP_DILUTION) ? "distributed" : "replicated") 
                  << ",dilution," << ((parallelism_type == MP_NON_DILUTION || parallelism_type == DP_NON_DILUTION) ? "no dilution" : "dilution")
                  << ",OpenMP," << omp_enabled
                  << std::endl;
        std::cout << "Model Construction Time," << std::chrono::duration_cast<std::chrono::nanoseconds>(end_lattice_construction - start_lattice_construction).count() / 1e9 << "s\n";
        std::cout << "BBPA Halving Time," << std::chrono::duration_cast<std::chrono::nanoseconds>(end_halving - start_halving).count() / 1e9 << "s\n";
    }

    // Free product lattice MPI env
    Product_lattice::MPI_Product_lattice_Free();

    // Finalize the MPI environment.
    MPI_Finalize();
}
