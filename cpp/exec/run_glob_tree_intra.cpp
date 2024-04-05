#include "core.hpp"
#include "product_lattice.hpp"
#include "tree.hpp"

int main(int argc, char *argv[])
{

    // omp_set_num_threads(8);
    int subjs = std::atoi(argv[1]);
    int variants = std::atoi(argv[2]);
    double prior = std::atof(argv[3]);
    double thres_up = 0.01;
    double thres_lo = 0.01;
    double thres_branch = 0.001;
    int search_depth = std::atoi(argv[4]);

    double pi0[subjs * variants];
    for (int i = 0; i < subjs * variants; i++)
    {
        pi0[i] = prior;
    }
    auto start_lattice_model_construction = std::chrono::high_resolution_clock::now();

    Product_lattice *p = new Product_lattice_non_dilution(subjs, variants, pi0);

    double **dilution = generate_dilution(subjs, 0.99, 0.005);

    Tree::thres_up(thres_up);
    Tree::thres_lo(thres_lo);
    Tree::thres_branch(thres_branch);
    Tree::search_depth(search_depth);
    Tree::dilution(dilution);

    std::chrono::nanoseconds BBPA_times[subjs + 1]{std::chrono::nanoseconds::zero()};

    auto start_tree_construction = std::chrono::high_resolution_clock::now();

    Global_tree_intra *tree = new Global_tree_intra(p, -1, -1, 1, 0, BBPA_times);

    auto stop_tree_construction = std::chrono::high_resolution_clock::now();

    tree->apply_true_state(p, 0);

    Tree_stat *prim = new Tree_stat(search_depth, 1);
    Tree_stat *temp = new Tree_stat(search_depth, 1);

    int total_st = p->total_states();
    for (int i = 0; i < total_st; i++)
    {
        tree->apply_true_state(p, i);
        tree->parse(i, p, 1.0, temp);
        prim->merge(temp);
    }

    std::stringstream file_name;
    file_name << "GlobalTreeIntra-" << p->type() << "-N="
              << subjs << "-k=" << variants
              << "-Prior=" << prior
              << "-Depth=" << search_depth
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

    std::cout << std::endl
              << std::endl
              << tree->shrinking_stat();

    prim->output_detail();

    // clean up memory
    delete prim;
    delete temp;

    for (int i = 0; i < subjs; i++)
    {
        delete[] dilution[i];
    }
    delete[] dilution;

    delete tree;

    auto stop_statistical_analysis = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds total_MPI_time = std::chrono::nanoseconds::zero();

    std::cout << "\n\n Performance Statistics\n"
              << std::endl;
    for (int i = 0; i < subjs + 1; i++)
    {
        std::cout << "Halving Time for lattice model size " << i << "," << BBPA_times[i].count() / 1e9 << " Second." << std::endl;
        total_MPI_time += BBPA_times[i];
    }
    std::cout << "Total Halving time," << total_MPI_time.count() / 1e9 << " Second." << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(start_tree_construction - start_lattice_model_construction);
    std::cout << "Initial Lattice Model Construction Time: " << duration.count() / 1e6 << " Second." << std::endl;
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_tree_construction - start_tree_construction);
    std::cout << "Global Tree Construction Time: " << duration.count() / 1e6 << " Second." << std::endl;
    std::cout << "Non Halving Time in Tree Construction: " << (stop_tree_construction - start_tree_construction - total_MPI_time).count() / 1e9 << " Second" << std::endl;
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_statistical_analysis - stop_tree_construction);
    std::cout << "Statistical Analysis Time: " << duration.count() / 1e6 << " Second." << std::endl;
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_statistical_analysis - start_lattice_model_construction);
    std::cout << "Total Time: " << duration.count() / 1e6 << " Second." << std::endl;
}
