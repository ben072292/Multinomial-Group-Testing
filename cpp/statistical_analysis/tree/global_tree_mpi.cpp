#include "global_tree_mpi.hpp"

Global_tree_mpi::Global_tree_mpi(Product_lattice* lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double** __restrict__ dilution, int rank, int world_size, MPI_Op* halving_op, MPI_Datatype* halving_res_type, Halving_res& __restrict__ halving_res) : Global_tree_mpi(lattice, ex, res, curr_stage){
    if (!lattice->is_classified() && curr_stage < stage) {
        bin_enc halving = -1;
        _children = new Global_tree*[1 << lattice->variants()];
        if((1 << lattice->curr_subjs()) >= world_size){ // for larger lattice models, need future tuning
            // lattice->halving(1.0 / (1 << lattice->variants()), rank, world_size, halving_res);
            lattice->halving_omp(1.0 / (1 << lattice->variants()), rank, world_size, halving_res);
            MPI_Allreduce(MPI_IN_PLACE, &halving_res, 2, *halving_res_type, *halving_op, MPI_COMM_WORLD);
            halving = halving_res.candidate; // remember test selection as halving_res will change in depth-frist traversal
        }
        else{ // for smaller lattice models
            // halving = lattice->halving(1.0 / (1 << lattice->variants()));
            halving = lattice->halving_omp(1.0 / (1 << lattice->variants()));
            // MPI_Allreduce(MPI_IN_PLACE, &halving, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        }
        bin_enc ex = true_ex(halving);
        for(int re = 0; re < (1 << lattice->variants()); re++){
            if(re != (1 << lattice->variants())-1){
                Product_lattice* p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                p->update_probs(halving, re, thres_up, thres_lo, dilution);
                _children[re] = new Global_tree_mpi(p, ex, re, k, _curr_stage+1, thres_up, thres_lo, stage, dilution, rank, world_size, halving_op, halving_res_type, halving_res);
            }
            else{ // reuse post_prob_ array in child to save memory
                Product_lattice* p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                _lattice->posterior_probs(nullptr); // detach post_prob_ from current lattice
                p->update_probs_in_place(halving, re, thres_up, thres_lo, dilution);
                _children[re] = new Global_tree_mpi(p, ex, re, k, _curr_stage+1, thres_up, thres_lo, stage, dilution, rank, world_size, halving_op, halving_res_type, halving_res);
            }
        } 
    }
    else{ // clean in advance to save memory
        delete[] lattice->posterior_probs();
        lattice->posterior_probs(nullptr);
    }
}

Global_tree_mpi::Global_tree_mpi(Product_lattice* lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double** __restrict__ dilution, int rank, int world_size, MPI_Op* halving_op, MPI_Datatype* halving_res_type, Halving_res& __restrict__ halving_res, std::chrono::nanoseconds mpi_times[]) : Global_tree_mpi(lattice, ex, res, curr_stage){
    auto start = std::chrono::high_resolution_clock::now(), end = start;
    if (!lattice->is_classified() && curr_stage < stage) {
        bin_enc halving = -1;
        _children = new Global_tree*[1 << lattice->variants()];
        if((1 << lattice->curr_subjs()) >= world_size){ // for larger lattice models
            // lattice->halving(1.0 / (1 << lattice->variants()), rank, world_size, halving_res);
            lattice->halving_omp(1.0 / (1 << lattice->variants()), rank, world_size, halving_res);
            MPI_Allreduce(MPI_IN_PLACE, &halving_res, 2, *halving_res_type, *halving_op, MPI_COMM_WORLD);
            end = std::chrono::high_resolution_clock::now();
            mpi_times[_lattice->curr_subjs()] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            halving = halving_res.candidate; // remember test selection as halving_res will change in depth-frist traversal
        }
        else{ // for smaller lattice models
            // halving = lattice->halving(1.0 / (1 << lattice->variants()));
            halving = lattice->halving_omp(1.0 / (1 << lattice->variants()));
            // MPI_Allreduce(MPI_IN_PLACE, &halving, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        }
        bin_enc ex = true_ex(halving);
        for(int re = 0; re < (1 << lattice->variants()); re++){
            if(re != (1 << lattice->variants())-1){
                Product_lattice* p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                p->update_probs(halving, re, thres_up, thres_lo, dilution);
                _children[re] = new Global_tree_mpi(p, ex, re, k, _curr_stage+1, thres_up, thres_lo, stage, dilution, rank, world_size, halving_op, halving_res_type, halving_res, mpi_times);
            }
            else{ // reuse post_prob_ array in child to save memory
                Product_lattice* p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                _lattice->posterior_probs(nullptr); // detach post_prob_ from current lattice
                p->update_probs_in_place(halving, re, thres_up, thres_lo, dilution);
                // only add the time at the last last child to avoid duplicated counting
                _children[re] = new Global_tree_mpi(p, ex, re, k, _curr_stage+1, thres_up, thres_lo, stage, dilution, rank, world_size, halving_op, halving_res_type, halving_res, mpi_times);
            }
        } 
    }
    else{ // clean in advance to save memory
        delete[] lattice->posterior_probs();
        lattice->posterior_probs(nullptr);
    }
}