#include "single_tree_mpi.hpp"

Single_tree_mpi::Single_tree_mpi(Product_lattice* lattice, int ex, int res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double** __restrict__ dilution, int rank, int world_size, MPI_Op* halving_op, MPI_Datatype* halving_res_type, Halving_res& __restrict__ halving_res) : Single_tree_mpi(lattice, ex, res, curr_stage){
    if (!lattice->is_classified() && curr_stage < stage) {
        children_ = new Single_tree*[1 << lattice->variants()];
        // lattice->halving(1.0 / (1 << lattice->variants()), rank, world_size, halving_res);
        lattice->halving_omp(1.0 / (1 << lattice->variants()), rank, world_size, halving_res);
        MPI_Allreduce(MPI_IN_PLACE, &halving_res, 2, *halving_res_type, *halving_op, MPI_COMM_WORLD);
        int halving = halving_res.candidate; // remember test selection as halving_res will change in depth-frist traversal
        int ex = true_ex(halving);
        for(int re = 0; re < (1 << lattice->variants()); re++){
            if(re != (1 << lattice->variants())-1){
                Product_lattice* p = lattice->clone(1);
                p->update_probs(halving, re, thres_up, thres_lo, dilution);
                children_[re] = new Single_tree_mpi(p, ex, re, k, curr_stage_+1, thres_up, thres_lo, stage, dilution, rank, world_size, halving_op, halving_res_type, halving_res);
            }
            else{ // reuse post_prob_ array in child to save memory
                Product_lattice* p = lattice->clone(1);
                lattice_->posterior_probs(nullptr); // detach post_prob_ from current lattice
                p->update_probs_in_place(halving, re, thres_up, thres_lo, dilution);
                children_[re] = new Single_tree_mpi(p, ex, re, k, curr_stage_+1, thres_up, thres_lo, stage, dilution, rank, world_size, halving_op, halving_res_type, halving_res);
            }
        } 
    }
    else{ // clean in advance to save memory
        delete[] lattice->posterior_probs();
        lattice->posterior_probs(nullptr);
    }
}