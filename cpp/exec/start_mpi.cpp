#include "core.hpp"
#include "halving_res.hpp"
#include "product_lattice_mp_dilution.hpp"
#include "product_lattice_mp_non_dilution.hpp"
#include "global_tree_mpi.hpp"

int main(int argc, char *argv[])
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

    int type = std::atoi(argv[1]);
    int subjs = std::atoi(argv[2]);
    int variant = std::atoi(argv[3]);
    double prior = std::atof(argv[4]);
    int search_depth = std::atoi(argv[5]);
    double thres_up = 0.01;
    double thres_lo = 0.01;
    double thres_branch = 0.001;

    // Initialize product lattice MPI env
    Product_lattice::MPI_Product_lattice_Initialize();

    // Initialize product lattice MPI env
    Global_tree_mpi::MPI_Global_tree_Initialize(subjs, search_depth, 1);

    double pi0[subjs * variant];
    for (int i = 0; i < subjs * variant; i++)
    {
        pi0[i] = prior;
    }

    auto start_lattice_model_construction = std::chrono::high_resolution_clock::now();

    Product_lattice *p;
    if (type == MP_NON_DILUTION)
    {
        Product_lattice_mp::MPI_Product_lattice_Initialize(subjs, variant);
        p = new Product_lattice_mp_non_dilution(subjs, variant, pi0);
    }
    else if (type == MP_DILUTION)
    {
        Product_lattice_mp::MPI_Product_lattice_Initialize(subjs, variant);
        p = new Product_lattice_mp_dilution(subjs, variant, pi0);
    }
    else if (type == DP_NON_DILUTION)
    {
        p = new Product_lattice_non_dilution(subjs, variant, pi0);
    }
    else if (type == DP_DILUTION)
    {
        p = new Product_lattice_dilution(subjs, variant, pi0);
    }
    else
    {
        exit(1);
    }

    auto stop_lattice_model_construction = std::chrono::high_resolution_clock::now();

    double **dilution = p->generate_dilution(0.99, 0.005);

    Halving_res halving_res;

    auto start_tree_construction = std::chrono::high_resolution_clock::now();

    // Fusion tree
    // Global_tree *tree = new Global_tree_mpi(p, -1, -1, 1, 0, thres_up, thres_lo, search_depth, pi0, dilution, halving_times, mp_update_times, dp_update_times, mp_dp_update_times);
    // Global tree
    // Global_tree* tree = new Global_tree_mp(p, -1, -1, 1, 0, thres_up, thres_lo, search_depth, dilution);
    Global_tree *tree = new Global_tree_mpi(p, -1, -1, 1, 0, thres_up, thres_lo, search_depth, dilution, true);

    auto stop_tree_construction = std::chrono::high_resolution_clock::now();

    Tree_stat prim(search_depth, 1);
    Tree_stat temp(search_depth, 1); 
    Tree_stat summ(search_depth, 1);

    int total_st = p->total_state();
    for (int i = total_st / world_size * rank; i < total_st / world_size * (rank + 1); i++)
    {
        tree->apply_true_state(p, i, thres_branch, dilution);
        tree->parse(i, p, pi0, thres_branch, 1.0, &temp);
        prim.merge(&temp);
    }

    MPI_Reduce(&prim, &summ, 1, Global_tree_mpi::tree_stat_type, Global_tree_mpi::tree_stat_op, 0, MPI_COMM_WORLD);

    if (!rank)
    {
        std::stringstream file_name;
        file_name << "Multinomial-" << p->type()
                  << "-N=" << subjs
                  << "-k=" << variant
                  << "-Prior=" << prior
                  << "-Depth=" << search_depth
                  << "-Processes=" << world_size
                  << "-Threads=" << omp_get_num_threads()
                  << "-" << get_curr_time()
                  << ".csv";
        freopen(file_name.str().c_str(), "w", stdout);
        std::cout << "N = " << subjs << ", k = " << variant << std::endl;
        std::cout << "Prior: ";
        for (int i = 0; i < p->curr_atoms(); i++)
        {
            std::cout << pi0[i] << ", ";
        }
        std::cout << "\nNegative classification threshold: " << thres_up << std::endl;
        std::cout << "Positive classification threshold: " << thres_lo << std::endl;
        std::cout << "Branch elimination threshold: " << thres_branch << std::endl;

        summ.output_detail();
    }

    auto stop_statistical_analysis = std::chrono::high_resolution_clock::now();

    if (rank == 0)
    {
        std::cout << "\n\nPerformance Statistics\n\n";

        std::cout << tree->shrinking_stat() << std::endl << std::endl;

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_lattice_model_construction - start_lattice_model_construction);
        std::cout << "Initial Lattice Model Construction Time: " << duration.count() / 1e6 << "s." << std::endl;
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_tree_construction - start_tree_construction);

        Global_tree_mpi::tree_perf->output_verbose();

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
    Product_lattice::MPI_Product_lattice_Free();
    if (type == MP_NON_DILUTION || type == MP_DILUTION)
    {
        Product_lattice_mp::MPI_Product_lattice_Free();
    }
    Global_tree_mpi::MPI_Global_tree_Free();

    // Finalize MPI
    MPI_Finalize();
}
