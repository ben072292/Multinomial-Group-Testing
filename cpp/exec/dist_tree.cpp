#include "product_lattice.hpp"
#include "tree.hpp"

EXPORT void run_dist_tree(int argc, char* argv[])
{
    auto start_time = std::chrono::high_resolution_clock::now();
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

    if (argc != 8)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " <parallelism_type> <subjs> <variants> <prior> <global_tree_depth> <search_depth> <workload_granularity>\n";
        }
        MPI_Finalize(); // Finalize MPI before exiting
        return;
    }

    int type = std::atoi(argv[1]);
    int subjs = std::atoi(argv[2]);
    int variants = std::atoi(argv[3]);
    double prior = std::atof(argv[4]);
    int search_depth = std::atoi(argv[5]);
    int workload_granularity = std::atoi(argv[6]);
    double thres_up = 0.01;
    double thres_lo = 0.01;
    double thres_branch = 0.001;

    // Initialize product lattice MPI env
    Product_lattice::MPI_Product_lattice_Initialize();

    // Initialize product lattice MPI env
    Distributed_tree::MPI_Distributed_tree_Initialize(subjs, 1, search_depth);

    double pi0[subjs * variants];
    for (int i = 0; i < subjs * variants; i++)
    {
        pi0[i] = prior;
    }

    double **dilution = generate_dilution(subjs, 0.99, 0.005);

    Tree::thres_up(thres_up);
    Tree::thres_lo(thres_lo);
    Tree::thres_branch(thres_branch);
    Tree::search_depth(search_depth);
    Tree::dilution(dilution);

    auto start_tree_construction = std::chrono::high_resolution_clock::now();
    auto stop_tree_construction = start_tree_construction;
    Tree_stat prim(search_depth, 1);
    Tree_stat temp(search_depth, 1);
    Tree_stat summ(search_depth, 1);

    int fin = -1; // flag that mark no more worker tasks

    Product_lattice *p;
    if (!rank) // master
    {
        if (type == REPL_NON_DILUTION)
        {
            p = new Product_lattice_non_dilution();
        }
        else if (type == REPL_DILUTION)
        {
            p = new Product_lattice_dilution();
        }
        else
        {
            log_error("Wrong lattice type specifier! Exiting...");
            exit(1);
        }

        start_tree_construction = std::chrono::high_resolution_clock::now();
        int fin_count = 0;
        int workload_count = 0;
        int worker_rank;
        while (fin_count != world_size - 1)
        {
            if (workload_count < (1 << (subjs * variants)))
            {
                MPI_Recv(&worker_rank, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&workload_count, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
                workload_count += workload_granularity;
                log_info("New workload patches to rank %d. Progress: %f %", worker_rank, (double)workload_count / (1 << (subjs * variants)) * 100);
            }
            else
            {
                MPI_Recv(&worker_rank, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&fin, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
                fin_count++;
                log_info("Signaling Fin to rank %d", worker_rank);
            }
        }
        log_info("All workers finish!");

        stop_tree_construction = std::chrono::high_resolution_clock::now();
    }
    else // workers
    {
        if (type == REPL_NON_DILUTION)
        {
            p = new Product_lattice_non_dilution(subjs, variants, pi0);
        }
        else if (type == REPL_DILUTION)
        {
            p = new Product_lattice_dilution(subjs, variants, pi0);
        }
        else
        {
            log_error("Wrong lattice type specifier! Exiting...");
            exit(1);
        }

        Distributed_tree *tree = new Distributed_tree(p, -1, -1, -1, 0, 1.0);

        int total_st = p->total_states();
        int recv_workload = rank * workload_granularity;
        while (recv_workload != fin)
        {
            for (int i = recv_workload; i < std::min(recv_workload + workload_granularity, total_st); i++)
            {
                tree->lazy_eval(tree, p, i);
                tree->parse(i, p, 1.0, &temp);
                prim.merge(&temp);
            }
            MPI_Send(&rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(&recv_workload, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_workload == fin)
            {
                log_info("Rank %d receives FIN", rank);
            }
        }
        delete tree;
    }

    MPI_Reduce(&prim, &summ, 1, Distributed_tree::tree_stat_type, Distributed_tree::tree_stat_op, 0, MPI_COMM_WORLD);

    if (!rank) // master generates statistics
    {
        std::stringstream file_name;
        file_name << "DistributedTree-" << p->type()
                  << "-N=" << subjs
                  << "-k=" << variants
                  << "-Prior=" << prior
                  << "-Depth=" << search_depth
                  << "-Processes=" << world_size
#ifdef ENABLE_OMP
                  << "-Threads=" << std::stoi(getenv("OMP_NUM_THREADS"))
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
        auto stop_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_tree_construction - start_tree_construction);
        std::cout << "Distributed Tree Construction Time: " << duration.count() / 1e6 << "s." << std::endl;
        std::cout << "Statistical Analysis Time: " << duration.count() / 1e6 << "s." << std::endl;
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
        std::cout << "Total Time: " << duration.count() / 1e6 << "s." << std::endl;
    }

    for (int i = 0; i < subjs; i++)
    {
        delete[] dilution[i];
    }
    delete[] dilution;

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
    Distributed_tree::MPI_Distributed_tree_Free();

    // Finalize MPI
    MPI_Finalize();
}
