#include "core.hpp"
#include "distributed_tree.hpp"
#include "halving_res.hpp"
#include "product_lattice_mp_dilution.hpp"
#include "product_lattice_mp_non_dilution.hpp"

int main(int argc, char *argv[])
{
    int type = std::atoi(argv[1]);
    int subjs = std::atoi(argv[2]);
    int variant = std::atoi(argv[3]);
    double prior = std::atof(argv[4]);
    int search_depth = std::atoi(argv[5]);
    double thres_up = 0.01;
    double thres_lo = 0.01;
    double thres_branch = 0.001;

    double pi0[subjs * variant];
    for (int i = 0; i < subjs * variant; i++)
    {
        pi0[i] = prior;
    }

    auto start_lattice_model_construction = std::chrono::high_resolution_clock::now();

    Product_lattice *p;

    if (type == DP_NON_DILUTION)
    {
        p = new Product_lattice_non_dilution(subjs, variant, pi0);
    }
    else if (type == DP_DILUTION)
    {
        p = new Product_lattice_dilution(subjs, variant, pi0);
    }
    else
    {
        log_error("Wrong lattice type specifier! Exiting...");
        exit(1);
    }

    auto stop_lattice_model_construction = std::chrono::high_resolution_clock::now();

    double **dilution = generate_dilution(subjs, 0.99, 0.005);

    Tree::thres_up(thres_up);
    Tree::thres_lo(thres_lo);
    Tree::thres_branch(thres_branch);
    Tree::search_depth(search_depth);
    Tree::dilution(dilution);

    Halving_res halving_res;

    auto start_tree_construction = std::chrono::high_resolution_clock::now();

    Distributed_tree *tree = new Distributed_tree(p, -1, -1, -1, 0, 1.0);

    Tree_stat prim(search_depth, 1);
    Tree_stat temp(search_depth, 1);

    for (int i = p->total_states() - 1; i >= 0; i--)
    {   
        tree->lazy_eval(tree, p, i);
        tree->parse(i, p, 1.0, &temp);
        prim.merge(&temp);
    }
    auto stop_tree_construction = std::chrono::high_resolution_clock::now();

    std::stringstream file_name;
    file_name << "DistributedTreeIntra-" << p->type()
              << "-N=" << subjs
              << "-k=" << variant
              << "-Prior=" << prior
              << "-Depth=" << search_depth
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

    prim.output_detail();

    auto stop_statistical_analysis = std::chrono::high_resolution_clock::now();

    std::cout << "\n\nPerformance Statistics\n\n";

    std::cout << tree->shrinking_stat() << std::endl
              << std::endl;

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_lattice_model_construction - start_lattice_model_construction);
    std::cout << "Initial Lattice Model Construction Time: " << duration.count() / 1e6 << "s." << std::endl;
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_tree_construction - start_tree_construction);

    std::cout << "Distributed Tree Construction Time: " << duration.count() / 1e6 << "s." << std::endl;
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_statistical_analysis - stop_tree_construction);
    std::cout << "Statistical Analysis Time: " << duration.count() / 1e6 << "s." << std::endl;
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_statistical_analysis - start_lattice_model_construction);
    std::cout << "Total Time: " << duration.count() / 1e6 << "s." << std::endl;

    for (int i = 0; i < subjs; i++)
    {
        delete[] dilution[i];
    }
    delete[] dilution;
    delete tree;
}