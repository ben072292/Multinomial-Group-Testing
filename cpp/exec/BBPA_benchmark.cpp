#include "product_lattice.hpp"

EXPORT void run_BBPA_benchmark(int argc, char *argv[])
{
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes and rank
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    if (argc != 4)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " <parallelism_type> <subjs> <variants>\n";
        }
        MPI_Finalize(); // Finalize MPI before exiting
        return;
    }

    // Parse arguments after validating argc
    int parallelism_type = std::atoi(argv[1]);
    int subjs = std::atoi(argv[2]);
    int variants = std::atoi(argv[3]);

    // Prepare variables for processing
    double pi0[subjs * variants];
    for (int i = 0; i < subjs * variants; i++)
    {
        pi0[i] = 0.01;
    }

    Product_lattice *p = nullptr;
    auto start_lattice_construction = std::chrono::high_resolution_clock::now();

    // Choose lattice type based on parallelism_type
    if (parallelism_type == DIST_NON_DILUTION)
    {
        Product_lattice_dist::MPI_Product_lattice_Initialize(subjs, variants);
        p = new Product_lattice_dist_non_dilution(subjs, variants, pi0);
    }
    else if (parallelism_type == DIST_DILUTION)
    {
        Product_lattice_dist::MPI_Product_lattice_Initialize(subjs, variants);
        p = new Product_lattice_dist_dilution(subjs, variants, pi0);
    }
    else if (parallelism_type == REPL_NON_DILUTION)
    {
        Product_lattice::MPI_Product_lattice_Initialize();
        p = new Product_lattice_non_dilution(subjs, variants, pi0);
    }
    else if (parallelism_type == REPL_DILUTION)
    {
        Product_lattice::MPI_Product_lattice_Initialize();
        p = new Product_lattice_dilution(subjs, variants, pi0);
    }
    else
    {
        if (rank == 0)
        {
            std::cerr << "Error: Invalid parallelism type\n";
        }
        MPI_Finalize(); // Finalize MPI before exiting
        return;
    }

    auto end_lattice_construction = std::chrono::high_resolution_clock::now();
    auto start_halving = std::chrono::high_resolution_clock::now();

    // Run the BBPA method
    bin_enc res = p->BBPA(1.0 / (1 << variants));
    auto end_halving = std::chrono::high_resolution_clock::now();

    // Output results if master process
    if (rank == 0)
    {
        std::stringstream file_name;
        file_name << "BBPA-Benchmark-"
#ifdef BBPA_NAIVE
            << "Baseline-"
#elif defined(BBPA_OP1)
            << "OP1-"
#elif defined(BBPA_OP2)
            << "OP2-"
// #elif defined(BBPA_OP3)
//             "BBPA-Benchmark-OP3-"
#elif defined(ENABLE_SIMD) && !defined(ENABLE_OMP)
            << "OP3-"
#elif defined(ENABLE_SIMD) && defined(ENABLE_OMP)
            << "OP4-"
#endif
                  << p->type()
                  << "-N=" << subjs
                  << "-k=" << variants
                  << "-Processes=" << world_size
#ifdef ENABLE_OMP
                  << "-Threads=" << std::stoi(getenv("OMP_NUM_THREADS"))
#endif
                  << "-" << get_curr_time()
                  << ".csv";
        freopen(file_name.str().c_str(), "w", stdout);
        std::cout << hardware_config_summary() << std::endl;
        std::cout << "N," << subjs
                  << ",k," << variants
                  << ",Model Parallelism:," << ((parallelism_type == DIST_NON_DILUTION || parallelism_type == DIST_DILUTION) ? "distributed" : "replicated")
                  << ",Use Dilution Effect:," << ((parallelism_type == DIST_NON_DILUTION || parallelism_type == REPL_NON_DILUTION) ? "no dilution" : "dilution")
#ifdef ENABLE_OMP
                  << ",OpenMP," << "Enabled"
#endif
                  << std::endl;
        std::cout << "BBPA Time," << std::chrono::duration_cast<std::chrono::nanoseconds>(end_halving - start_halving).count() / 1e9 << "s\n";
        std::cout << "Candidate," << res << std::endl;
    }

    // Free product lattice MPI environment
    Product_lattice::MPI_Product_lattice_Finalize();

    // Finalize the MPI environment
    MPI_Finalize();
}