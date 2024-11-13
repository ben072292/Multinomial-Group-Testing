#include "product_lattice.hpp"
#include "tree.hpp"

EXPORT void run_fusion_tree(int argc, char *argv[])
{
    // Initialize the MPI environment
    int provided_thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
    if (provided_thread_level < MPI_THREAD_FUNNELED)
    {
        log_error("The threading support level is lesser than that demanded.");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    if (argc != 6)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " <type> <subjs> <variants> <prior> <search_depth>\n";
        }
        MPI_Finalize(); // Finalize MPI before exiting
        return;
    }

    int type = std::atoi(argv[1]);
    int subjs = std::atoi(argv[2]);
    int variants = std::atoi(argv[3]);
    double prior = std::atof(argv[4]);
    int search_depth = std::atoi(argv[5]);
    double thres_up = 0.01;
    double thres_lo = 0.01;
    double thres_branch = 0.001;

    Tree::thres_up(thres_up);
    Tree::thres_lo(thres_lo);
    Tree::thres_branch(thres_branch);
    Tree::search_depth(search_depth);

    // Initialize product lattice MPI env
    Fusion_tree::MPI_Fusion_tree_Initialize(subjs, 1);

    double pi0[subjs * variants];
    for (int i = 0; i < subjs * variants; i++)
    {
        pi0[i] = prior;
    }

    auto start_lattice_model_construction = std::chrono::high_resolution_clock::now();

    Product_lattice *p;
    if (type == DIST_NON_DILUTION)
    {
        Product_lattice_dist::MPI_Product_lattice_Initialize(subjs, variants);
        p = new Product_lattice_dist_non_dilution(subjs, variants, pi0);
    }
    else if (type == DIST_DILUTION)
    {
        Product_lattice_dist::MPI_Product_lattice_Initialize(subjs, variants);
        p = new Product_lattice_dist_dilution(subjs, variants, pi0);
    }
    else if (type == REPL_NON_DILUTION)
    {
        Product_lattice::MPI_Product_lattice_Initialize();
        p = new Product_lattice_non_dilution(subjs, variants, pi0);
    }
    else if (type == REPL_DILUTION)
    {
        // Initialize product lattice MPI env
        Product_lattice::MPI_Product_lattice_Initialize();
        p = new Product_lattice_dilution(subjs, variants, pi0);
    }
    else
    {
        exit(1);
    }

    auto stop_lattice_model_construction = std::chrono::high_resolution_clock::now();

    double **dilution = generate_dilution(subjs, 0.99, 0.005);

    Tree::dilution(dilution);

    auto start_tree_construction = std::chrono::high_resolution_clock::now();
    /* Fusion tree */
    Tree *tree = new Fusion_tree(p, -1, -1, 1, 0, 0.01, 0.0, 1e-6);

    auto stop_tree_construction = std::chrono::high_resolution_clock::now();

    Tree_stat prim(search_depth, 1);
    Tree_stat temp(search_depth, 1);
    Tree_stat summ(search_depth, 1);

    int total_st = p->total_states();
    for (int i = total_st / world_size * rank; i < total_st / world_size * (rank + 1); i++)
    {
        tree->apply_true_state(p, i);
        tree->parse(i, p, 1.0, &temp);
        prim.merge(&temp);
    }

    MPI_Reduce(&prim, &summ, 1, Global_tree::tree_stat_type, Global_tree::tree_stat_op, 0, MPI_COMM_WORLD);

    if (!rank)
    {
        std::stringstream file_name;
        file_name << "FusionTree-" << p->type()
                  << "-N=" << subjs
                  << "-k=" << variants
                  << "-Prior=" << prior
                  << "-Depth=" << search_depth
                  << "-Processes=" << world_size
#ifdef ENABLE_OMP
                  << "-Threads=" << omp_get_num_threads()
#endif
                  << "-" << get_curr_time()
                  << ".csv";
        freopen(file_name.str().c_str(), "w", stdout);
        std::cout << hardware_config_summary() << std::endl;
        std::cout << "N = " << subjs << ", k = " << variants << std::endl;
        std::cout << "Prior: ";
        for (int i = 0; i < p->curr_atoms(); i++)
        {
            std::cout << pi0[i] << ", ";
        }
        std::cout << "\nNegative classification threshold: " << thres_up << std::endl;
        std::cout << "Positive classification threshold: " << thres_lo << std::endl;
        std::cout << "Branch elimination threshold: " << thres_branch << std::endl;
        summ.output_detail();
        auto stop_statistical_analysis = std::chrono::high_resolution_clock::now();
        std::cout << "\n\nPerformance Statistics\n\n";

        std::cout << tree->shrinking_stat() << std::endl
                  << std::endl;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_lattice_model_construction - start_lattice_model_construction);
        std::cout << "Initial Lattice Model Construction Time: " << duration.count() / 1e6 << "s." << std::endl;
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_tree_construction - start_tree_construction);

        Global_tree::tree_perf->output_verbose();

        std::cout << "Global Tree Construction Time: " << duration.count() / 1e6 << "s." << std::endl;
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_statistical_analysis - stop_tree_construction);
        std::cout << "Statistical Analysis Time: " << duration.count() / 1e6 << "s." << std::endl;
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_statistical_analysis - start_lattice_model_construction);
        std::cout << "Total Time: " << duration.count() / 1e6 << "s." << std::endl;
    }

    for (int i = 0; i < subjs; i++)
    {
        delete[] dilution[i];
    }
    delete[] dilution;
    delete tree;

    // Free product lattice MPI env
    switch (type)
    {
    case DIST_NON_DILUTION:
        Product_lattice_dist::MPI_Product_lattice_Finalize();
        break;
    case DIST_DILUTION:
        Product_lattice_dist::MPI_Product_lattice_Finalize();
        break;
    case REPL_NON_DILUTION:
        Product_lattice::MPI_Product_lattice_Finalize();
        break;
    case REPL_DILUTION:
        Product_lattice::MPI_Product_lattice_Finalize();
        break;
    default:
        throw std::logic_error("Nonexisting product lattice type! Exiting...");
        exit(1);
    }
    Fusion_tree::MPI_Fusion_tree_Free();

    // Finalize MPI
    MPI_Finalize();
}
