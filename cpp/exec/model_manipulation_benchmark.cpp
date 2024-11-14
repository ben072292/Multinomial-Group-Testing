#include "product_lattice.hpp"

EXPORT void run_model_manipulation_benchmark(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    if (argc != 5)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " <parallelism_type> <subjs> <variants> <iterations>\n";
        }
        MPI_Finalize(); // Finalize MPI before exiting
        return;
    }

    int parallelism_type = std::atoi(argv[1]);
    int subjs = std::atoi(argv[2]);
    int variants = std::atoi(argv[3]);
    int iters = std::atoi(argv[4]);

    double pi0[subjs * variants];
    for (int i = 0; i < subjs * variants; i++)
    {
        pi0[i] = 0.01;
    }

    Product_lattice *p = nullptr;

    if (parallelism_type == DIST_NON_DILUTION)
    {
        Product_lattice_dist::MPI_Product_lattice_Initialize(subjs, variants);
    }
    else if (parallelism_type == DIST_DILUTION)
    {
        Product_lattice_dist::MPI_Product_lattice_Initialize(subjs, variants);
    }
    else if (parallelism_type == REPL_NON_DILUTION)
    {
        Product_lattice::MPI_Product_lattice_Initialize();
    }
    else if (parallelism_type == REPL_DILUTION)
    {
        Product_lattice::MPI_Product_lattice_Initialize();
    }
    else
        exit(1);

    auto start_lattice_construction = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
    {
        if (parallelism_type == DIST_NON_DILUTION)
        {
            p = new Product_lattice_dist_non_dilution(subjs, variants, pi0);
        }
        else if (parallelism_type == DIST_DILUTION)
        {
            p = new Product_lattice_dist_dilution(subjs, variants, pi0);
        }
        else if (parallelism_type == REPL_NON_DILUTION)
        {
            p = new Product_lattice_non_dilution(subjs, variants, pi0);
        }
        else if (parallelism_type == REPL_DILUTION)
        {
            p = new Product_lattice_dilution(subjs, variants, pi0);
        }
        else
            exit(1);

        if (p != nullptr && i != iters - 1)
        {
            delete p;
        }
    }

    auto end_lattice_construction = std::chrono::high_resolution_clock::now();

    auto start_updating = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iters; i++)
    {
        p->update_probs_in_place(1, 3, nullptr);
    }

    auto end_updating = std::chrono::high_resolution_clock::now();

    auto start_classification = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iters; i++)
    {
        p->update_metadata(0.001, 0.001);
    }

    auto end_classification = std::chrono::high_resolution_clock::now();

    if (rank == 0)
    {
        std::stringstream file_name;
        file_name << "Model-Manipulation-Benchmark-" << p->type()
                  << "-N=" << subjs
                  << "-k=" << variants
                  << "-i=" << iters
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
        std::cout << "Model Construction Time," << std::chrono::duration_cast<std::chrono::nanoseconds>(end_lattice_construction - start_lattice_construction).count() / 1e9 << "s\n";
        std::cout << "Model Update Time," << std::chrono::duration_cast<std::chrono::nanoseconds>(end_updating - start_updating).count() / 1e9 << "s\n";
        std::cout << "Model Classification Identification Time," << std::chrono::duration_cast<std::chrono::nanoseconds>(end_classification - start_classification).count() / 1e9 << "s\n";
    }

    // Free product lattice MPI env
    Product_lattice::MPI_Product_lattice_Finalize();

    // Finalize the MPI environment.
    MPI_Finalize();
}
