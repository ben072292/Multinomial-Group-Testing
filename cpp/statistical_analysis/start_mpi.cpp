#include "tree/global_tree_mpi.hpp"
#include "../product_lattice_model/product_lattice_dilution.hpp"
#include "../product_lattice_model/product_lattice_non_dilution.hpp"
#include "../product_lattice_model/halving_res/halving_res.hpp"
#include "../core.hpp"

int main(int argc, char* argv[]){
    // Initialize the MPI environment
    int provided_thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
    if(provided_thread_level < MPI_THREAD_FUNNELED){
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

     // Create the datatype
    MPI_Datatype halving_res_type;
    Halving_res::create_halving_res_type(&halving_res_type);
    MPI_Type_commit(&halving_res_type);

    MPI_Op halving_op;
    MPI_Op_create((MPI_User_function*)&Halving_res::halving_reduce, true, &halving_op);


    int atom = std::atoi(argv[1]);
    int variant = std::atoi(argv[2]);
    double prior = std::atof(argv[3]);
    int search_depth = std::atoi(argv[4]);
    double thres_up = 0.01;
    double thres_lo = 0.01;
    double thres_branch = 0.001;

    double pi0[atom * variant];
    for(int i = 0; i < atom * variant; i++){
        pi0[i] = prior;
    }

    auto start_lattice_model_construction = std::chrono::high_resolution_clock::now();

    Product_lattice* p = new Product_lattice_non_dilution(atom, variant, pi0);

    auto stop_lattice_model_construction = std::chrono::high_resolution_clock::now();

    double** dilution = p->generate_dilution(0.99, 0.005);

    Halving_res halving_res;

    auto start_tree_construction = std::chrono::high_resolution_clock::now();

    std::chrono::nanoseconds mpi_times[atom + 1]{std::chrono::nanoseconds::zero()};
    Global_tree* tree = new Global_tree_mpi(p, -1, -1, 1, 0, thres_up, thres_lo, search_depth, dilution, world_rank, world_size, &halving_op, &halving_res_type, halving_res, mpi_times);

    // Global_tree* tree = new Global_tree_mpi(p, -1, -1, 1, 0, thres_up, thres_lo, search_depth, dilution, world_rank, world_size, &halving_op, &halving_res_type, halving_res);

    auto stop_tree_construction = std::chrono::high_resolution_clock::now();
    if(world_rank == 0){
        tree->apply_true_state(p, 0, thres_branch, dilution);

        Tree_stat* prim = new Tree_stat(search_depth, 1);
        Tree_stat* temp = new Tree_stat(search_depth, 1);

        int total_st = p->total_state();
        for(int i = 0; i < total_st; i++){
            tree->apply_true_state(p, i, 0.001, dilution);
            tree->parse(i, p, pi0, thres_branch, 1.0, temp);
            prim->merge(temp);
        }

        std::stringstream file_name;
        file_name << "Multinomial-" << p->type() << "-N=" << atom << "-k=" << variant << "-Prior=" << prior << "-Depth=" << search_depth << "-" << get_curr_time() << ".csv";
        freopen(file_name.str().c_str(),"w",stdout);

        std::cout << std::endl << std::endl << tree->shrinking_stat();

        std::cout << "N = " << atom << ", k = " << variant << std::endl;
        std::cout << "Prior: ";
        for (int i = 0; i < p->curr_atoms(); i++){
            std::cout << pi0[i] << ", ";
        }
        
        std::vector<const Global_tree*> *leaves = new std::vector<const Global_tree*>;
        tree->find_all_leaves(tree, leaves);
        std::cout << std::endl << "\nNumber of tree leaves," << leaves->size() << std::endl;
        delete leaves;

        prim->output_detail();

        // clean up memory
        delete prim;
        delete temp;

    }
    auto stop_statistical_analysis = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds total_MPI_time = std::chrono::nanoseconds::zero();
    if(world_rank == 0){
        std::cout << "\n\n Performance Statistics\n" << std::endl;
        for(int i = 0; i < atom + 1; i++){
           std::cout << "MPI Time for lattice model size " << i << "," << mpi_times[i].count()/1e9 << " Second." << std::endl; 
           total_MPI_time += mpi_times[i];
        }
        std::cout << "Total MPI time," << total_MPI_time.count()/1e9 << " Second." << std::endl;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_lattice_model_construction - start_lattice_model_construction);
        std::cout << "Initial Lattice Model Construction Time: " << duration.count()/1e6 << " Second." << std::endl; 
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_tree_construction - start_tree_construction);
        std::cout << "Global Tree Construction Time: " << duration.count()/1e6 << " Second." << std::endl;
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_statistical_analysis - stop_tree_construction);
        std::cout << "Statistical Analysis Time: " << duration.count()/1e6 << " Second." << std::endl;
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_statistical_analysis - start_lattice_model_construction);
        std::cout << "Total Time: " << duration.count()/1e6 << " Second." << std::endl;
    }

    for(int i = 0; i < atom; i++){
        delete[] dilution[i];
    }
    delete[] dilution;
    delete tree;

    // Free datatype
    MPI_Type_free(&halving_res_type);
    // Free reduce op
    MPI_Op_free(&halving_op);
    // Finalize the MPI environment.
    MPI_Finalize();
}
