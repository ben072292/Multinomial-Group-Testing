#include "tree/single_tree.hpp"
#include "../product_lattice_model/product_lattice.hpp"
#include "../product_lattice_model/product_lattice_dilution.hpp"
#include "../product_lattice_model/product_lattice_non_dilution.hpp"
#include <chrono>
// #include <omp.h>

int main(int argc, char* argv[]){

    // omp_set_num_threads(8);
    int atom = std::atoi(argv[1]);
    int variant = std::atoi(argv[2]);
    double prior = std::atof(argv[3]);
    double thres_up = 0.005;
    double thres_lo = 0.005;
    double thres_branch = 0.001;
    int search_depth = 5;

    double pi0[atom * variant];
    for(int i = 0; i < atom * variant; i++){
        pi0[i] = prior;
    }

    auto start = std::chrono::high_resolution_clock::now();

    Product_lattice* p = new Product_lattice_dilution(atom, variant, pi0);

    double** dilution = p->generate_dilution(0.99, 0.005);

    Single_tree* tree = new Single_tree(p, -1, -1, 1, 0, thres_up, thres_lo, search_depth, dilution);

    // std::vector<const Single_tree*> *leaves = new std::vector<const Single_tree*>;
    // Single_tree::find_all(tree, leaves);
    // std::cout << leaves->size() << std::endl;
    
    tree->apply_true_state(p, 0, thres_branch, dilution);

    Tree_stat* prim = new Tree_stat(search_depth, 1);
    Tree_stat* temp = new Tree_stat(search_depth, 1);
    tree->parse(0, p, pi0, thres_branch, 1.0, prim);

    int total_st = p->total_state();
    for(int i = 0; i < total_st; i++){
        tree->apply_true_state(p, i, 0.001, dilution);
        tree->parse(i, p, pi0, thres_branch, 1.0, temp);
        prim->merge(temp);
    }

    std::cout << "N = " << atom << ", k = " << variant << std::endl;
    std::cout << "Prior: ";
    for (int i = 0; i < p->nominal_pool_size(); i++){
        std::cout << pi0[i] << ", ";
    }
    std::cout << std::endl;
    prim->output_detail(p->nominal_pool_size());
    
    // clean up memory
    delete prim;
    delete temp;

    for(int i = 0; i < atom; i++){
        delete[] dilution[i];
    }
    delete[] dilution;

    delete tree;

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "\nTime Consumption: " << duration.count()/1e6 << " Second." << std::endl;
}
