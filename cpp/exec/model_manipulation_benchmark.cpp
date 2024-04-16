#include "product_lattice.hpp"

EXPORT void run_model_manipulation_benchmark(int argc, char *argv[])
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

    int parallelism_type = std::atoi(argv[1]);
    int subjs = std::atoi(argv[2]);
    int variants = std::atoi(argv[3]);

    double pi0[subjs * variants];
    for (int i = 0; i < subjs * variants; i++)
    {
        pi0[i] = 0.01;
    }

    Product_lattice *p;
    auto start_lattice_construction = std::chrono::high_resolution_clock::now();
    if (parallelism_type == DIST_NON_DILUTION){
        Product_lattice_dist::MPI_Product_lattice_Initialize(subjs, variants);
        p = new Product_lattice_dist_non_dilution(subjs, variants, pi0);
    }
    else if (parallelism_type == DIST_DILUTION){
        Product_lattice_dist::MPI_Product_lattice_Initialize(subjs, variants);
        p = new Product_lattice_dist_dilution(subjs, variants, pi0);
    }
    else if (parallelism_type == REPL_NON_DILUTION){
        Product_lattice::MPI_Product_lattice_Initialize();
        p = new Product_lattice_non_dilution(subjs, variants, pi0);
    }
    else if (parallelism_type == REPL_DILUTION){
        Product_lattice::MPI_Product_lattice_Initialize();
        p = new Product_lattice_dilution(subjs, variants, pi0);
    }
    else
        exit(1);

    auto end_lattice_construction = std::chrono::high_resolution_clock::now();

    auto start_updating = std::chrono::high_resolution_clock::now();

    p->update_probs(1, 3, nullptr);

    auto end_updating = std::chrono::high_resolution_clock::now();

    auto start_classification = std::chrono::high_resolution_clock::now();

    p->update_metadata(0.001, 0.001);

    auto end_classification = std::chrono::high_resolution_clock::now();

    if (world_rank == 0)
    {
                std::stringstream file_name;
                file_name << "Model-Manipulation-Benchmark-" << p->type()
                          << "-N=" << subjs
                          << "-k=" << variants
                          << "-Processes=" << world_size
        #ifdef ENABLE_OMP
                          << "-Threads=" << omp_get_num_threads()
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
        std::cout << "Model Construction Time," << std::chrono::duration_cast<std::chrono::nanoseconds>(end_lattice_construction - start_lattice_construction).count() / 1e9 << "s\n";
        std::cout << "Model Update Time," << std::chrono::duration_cast<std::chrono::nanoseconds>(end_updating - start_updating).count() / 1e9 << "s\n";
        std::cout << "Model Classification Identification Time," << std::chrono::duration_cast<std::chrono::nanoseconds>(end_classification - start_classification).count() / 1e9 << "s\n";
    }

    // Free product lattice MPI env
    Product_lattice::MPI_Product_lattice_Finalize();

    // Finalize the MPI environment.
    MPI_Finalize();
}
