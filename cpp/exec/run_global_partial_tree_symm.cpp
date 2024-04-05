#include "core.hpp"
#include "product_lattice.hpp"
#include "tree.hpp"

int main(int argc, char *argv[])
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

    int type = std::atoi(argv[1]);
    int subjs = std::atoi(argv[2]);
    int variants = std::atoi(argv[3]);
    double prior = std::atof(argv[4]);
    int search_depth = std::atoi(argv[5]);
    double thres_up = 0.01;
    double thres_lo = 0.01;
    double thres_branch = 0.001;

    // Initialize product lattice MPI env
    Product_lattice::MPI_Product_lattice_Initialize();

    // Initialize product lattice MPI env
    Global_partial_tree::MPI_Distributed_tree_Initialize(subjs, 1, search_depth);

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

    Tree_stat prim(search_depth, 1);
    Tree_stat temp(search_depth, 1);

    auto start_lattice_gen = std::chrono::high_resolution_clock::now();
    Product_lattice *p;
    if (type == MP_NON_DILUTION)
    {
        Product_lattice_dist::MPI_Product_lattice_Initialize(subjs, variants);
        p = new Product_lattice_dist_non_dilution(subjs, variants, pi0);
    }
    else if (type == MP_DILUTION)
    {
        Product_lattice_dist::MPI_Product_lattice_Initialize(subjs, variants);
        p = new Product_lattice_dist_dilution(subjs, variants, pi0);
    }
    else
    {
        log_error("Wrong lattice type specifier! Exiting...");
        exit(1);
    }

    int symm_size = 0;
    bin_enc *symm_true_states(nullptr);
    int *symm_coefficients(nullptr);
    generate_symmetric_true_states(subjs, variants, symm_size, symm_true_states, symm_coefficients);

    if (!rank)
        log_info("%d true states using symmetric property", symm_size);

    Global_partial_tree *tree = new Global_partial_tree(p, -1, -1, -1, 0, 1.0);
    auto stop_lattice_gen = std::chrono::high_resolution_clock::now();

    auto start_evaluation = std::chrono::high_resolution_clock::now();
    auto stop_evaluation = start_evaluation;

    for (int i = 0; i < symm_size; i++)
    {
        auto start_timer = std::chrono::high_resolution_clock::now();
        tree->lazy_eval(tree, p, symm_true_states[i]);
        tree->parse(symm_true_states[i], p, symm_coefficients[i], &temp);
        prim.merge(&temp);

        auto stop_timer = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(stop_timer - start_timer);
        if (!rank)
        {
            log_info("True state %d / %d finishes, took %.2f s.", i + 1, symm_size, time.count() / 1e6);
            if ((i + 1) % 100 == 0)
            {
                log_info("Rank %d: Tree size is %.2fMB", rank, static_cast<double>(tree->size_estimator()) / 1024 / 1024);
                log_info("%s", tree->shrinking_stat().c_str());
            }
        }
    }

    delete[] symm_true_states;
    delete[] symm_coefficients;

    if (!rank) // master generates statistics
    {
        std::stringstream file_name;
        file_name << "GlobalPartialTreeSymm-" << p->type()
                  << "-N=" << subjs
                  << "-k=" << variants
                  << "-Prior=" << prior
                  << "-Depth=" << search_depth
                  << "-Processes=" << world_size
                  << "-Threads=" << omp_get_num_threads()
                  << "-" << get_curr_time()
                  << ".csv";
        freopen(file_name.str().c_str(), "w", stdout);
        std::cout << "N = " << subjs << ", k = " << variants << std::endl;
        std::cout << "Prior: ";
        for (int i = 0; i < p->curr_atoms(); i++)
        {
            std::cout << pi0[i] << ", ";
        }
        std::cout << "\nNegative classification threshold: " << thres_up << std::endl;
        std::cout << "Positive classification threshold: " << thres_lo << std::endl;
        std::cout << "Branch elimination threshold: " << thres_branch << std::endl;
        std::cout << "Number of true states: " << symm_size << std::endl;
        prim.output_detail();
        auto stop_time = std::chrono::high_resolution_clock::now();
        std::cout << "\n\nPerformance Statistics\n\n";
        std::cout << tree->shrinking_stat() << std::endl
                  << std::endl;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_lattice_gen - start_lattice_gen);
        std::cout << "Initial Lattice Construction Time: " << duration.count() / 1e6 << "s." << std::endl;
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_evaluation - start_evaluation);
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
    delete tree;

    // Free product lattice MPI env
    Product_lattice::MPI_Product_lattice_Free();
    if (type == MP_NON_DILUTION || type == MP_DILUTION)
    {
        Product_lattice_dist::MPI_Product_lattice_Free();
    }
    Global_partial_tree::MPI_Distributed_tree_Free();

    // Finalize MPI
    MPI_Finalize();
}
