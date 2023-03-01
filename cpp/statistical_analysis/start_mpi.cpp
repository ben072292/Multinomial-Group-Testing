#include "../core.hpp"
#include "../product_lattice_model/halving_res/halving_res.hpp"
#include "../product_lattice_model/product_lattice_mp_dilution.hpp"
#include "../product_lattice_model/product_lattice_mp_non_dilution.hpp"
#include "tree/global_tree_mpi.hpp"

int main(int argc, char *argv[])
{
    // Initialize the MPI environment
    int provided_thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
    if (provided_thread_level < MPI_THREAD_FUNNELED)
    {
        std::cout << "The threading support level is lesser than that demanded.\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

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

    // Initialize product lattice MPI env
    Global_tree_mpi::MPI_Global_tree_Initialize();

    int type = std::atoi(argv[1]);
    int atom = std::atoi(argv[2]);
    int variant = std::atoi(argv[3]);
    double prior = std::atof(argv[4]);
    int search_depth = std::atoi(argv[5]);
    double thres_up = 0.01;
    double thres_lo = 0.01;
    double thres_branch = 0.001;

    double pi0[atom * variant];
    for (int i = 0; i < atom * variant; i++)
    {
        pi0[i] = prior;
    }

    auto start_lattice_model_construction = std::chrono::high_resolution_clock::now();

    Product_lattice *p;
    if (type == MP_NON_DILUTION)
    {
        Product_lattice_mp::MPI_Product_lattice_Initialize(atom, variant);
        p = new Product_lattice_mp_non_dilution(atom, variant, pi0);
    }
    else if (type == MP_DILUTION)
    {
        Product_lattice_mp::MPI_Product_lattice_Initialize(atom, variant);
        p = new Product_lattice_mp_dilution(atom, variant, pi0);
    }
    else if (type == DP_NON_DILUTION)
    {
        p = new Product_lattice_non_dilution(atom, variant, pi0);
    }
    else if (type == DP_DILUTION)
    {
        p = new Product_lattice_dilution(atom, variant, pi0);
    }
    else
    {
        exit(1);
    }

    auto stop_lattice_model_construction = std::chrono::high_resolution_clock::now();

    double **dilution = p->generate_dilution(0.99, 0.005);

    Halving_res halving_res;

    auto start_tree_construction = std::chrono::high_resolution_clock::now();

    std::chrono::nanoseconds halving_times[atom + 1]{std::chrono::nanoseconds::zero()};
    std::chrono::nanoseconds mp_update_times[atom + 1]{std::chrono::nanoseconds::zero()};
    std::chrono::nanoseconds dp_update_times[atom + 1]{std::chrono::nanoseconds::zero()};
    std::chrono::nanoseconds mp_dp_update_times[atom + 1]{std::chrono::nanoseconds::zero()};
    Global_tree *tree = new Global_tree_mpi(p, -1, -1, 1, 0, thres_up, thres_lo, search_depth, dilution, halving_times, mp_update_times, dp_update_times, mp_dp_update_times);

    // Global_tree* tree = new Global_tree_mp(p, -1, -1, 1, 0, thres_up, thres_lo, search_depth, dilution);

    auto stop_tree_construction = std::chrono::high_resolution_clock::now();
    if (world_rank == 0)
    {
        tree->apply_true_state(p, 0, thres_branch, dilution);

        Tree_stat *prim = new Tree_stat(search_depth, 1);
        Tree_stat *temp = new Tree_stat(search_depth, 1);

        int total_st = p->total_state();
        for (int i = 0; i < total_st; i++)
        {
            tree->apply_true_state(p, i, 0.001, dilution);
            tree->parse(i, p, pi0, thres_branch, 1.0, temp);
            prim->merge(temp);
        }
        std::stringstream file_name;
        file_name << "Multinomial-" << p->type()
                  << "-N=" << atom
                  << "-k=" << variant
                  << "-Prior=" << prior
                  << "-Depth=" << search_depth
                  << "-Processes=" << world_size
                  << "-Threads=" << omp_get_num_threads()
                  << "-" << get_curr_time()
                  << ".csv";
        freopen(file_name.str().c_str(), "w", stdout);
        std::cout << std::endl
                  << std::endl
                  << tree->shrinking_stat();

        std::cout << "N = " << atom << ", k = " << variant << std::endl;
        std::cout << "Prior: ";
        for (int i = 0; i < p->curr_atoms(); i++)
        {
            std::cout << pi0[i] << ", ";
        }

        std::vector<const Global_tree *> *leaves = new std::vector<const Global_tree *>;
        tree->find_all_leaves(tree, leaves);
        std::cout << std::endl
                  << "\nNumber of tree leaves," << leaves->size() << std::endl;
        delete leaves;

        prim->output_detail();

        // clean up memory
        delete prim;
        delete temp;
    }
    auto stop_statistical_analysis = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds total_halving_time = std::chrono::nanoseconds::zero();
    std::chrono::nanoseconds total_mp_update_time = std::chrono::nanoseconds::zero();
    std::chrono::nanoseconds total_dp_update_time = std::chrono::nanoseconds::zero();
    std::chrono::nanoseconds total_mp_dp_update_time = std::chrono::nanoseconds::zero();

    if (world_rank == 0)
    {
        std::cout << "\n\n Performance Statistics\n"
                  << std::endl;
        for (int i = 0; i < atom + 1; i++)
        {
            std::cout << "Model size:," << i << ",Having time:," << halving_times[i].count() / 1e9 << "s,"
                      << "MP Update time:," << mp_update_times[i].count() / 1e9 << "s,"
                      << "DP Update time:," << dp_update_times[i].count() / 1e9 << "s,"
                      << "MP-DP Update time:," << mp_dp_update_times[i].count() / 1e9 << "s" << std::endl;
            total_halving_time += halving_times[i];
            total_mp_update_time += mp_update_times[i];
            total_dp_update_time += dp_update_times[i];
            total_mp_dp_update_time += mp_dp_update_times[i];
        }
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_lattice_model_construction - start_lattice_model_construction);
        std::cout << "Initial Lattice Model Construction Time: " << duration.count() / 1e6 << "s." << std::endl;
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_tree_construction - start_tree_construction);
        std::cout << "Total Halving time," << total_halving_time.count() / 1e9 << "s." << std::endl;
        std::cout << "Total MP Update time," << total_mp_update_time.count() / 1e9 << "s." << std::endl;
        std::cout << "Total DP Update time," << total_dp_update_time.count() / 1e9 << "s." << std::endl;
        std::cout << "Total MP-DP Update time," << total_mp_dp_update_time.count() / 1e9 << "s." << std::endl;
        std::cout << "Total Update time," << ((total_mp_update_time.count() + total_dp_update_time.count() + total_mp_dp_update_time.count()) / 1e9) << "s." << std::endl;
        std::cout << "Extra Time," << (duration.count() / 1e6 - total_halving_time.count() / 1e9 - total_mp_update_time.count() / 1e9 - total_dp_update_time.count() / 1e9 - total_mp_dp_update_time.count() / 1e9) << "s." << std::endl;
        std::cout << "Global Tree Construction Time: " << duration.count() / 1e6 << "s." << std::endl;
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_statistical_analysis - stop_tree_construction);
        std::cout << "Statistical Analysis Time: " << duration.count() / 1e6 << "s." << std::endl;
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_statistical_analysis - start_lattice_model_construction);
        std::cout << "Total Time: " << duration.count() / 1e6 << "s." << std::endl;
    }

    for (int i = 0; i < atom; i++)
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

    // Finalize MPI
    MPI_Finalize();
}
