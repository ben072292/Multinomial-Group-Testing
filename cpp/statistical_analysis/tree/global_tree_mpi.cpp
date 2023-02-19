#include "global_tree_mpi.hpp"

Global_tree_mpi::Global_tree_mpi(Product_lattice* lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double** __restrict__ dilution) : Global_tree_mpi(lattice, ex, res, curr_stage){
    if (!lattice->is_classified() && curr_stage < stage) {
        _children = new Global_tree*[1 << lattice->variants()];
        bin_enc halving = lattice->halving_mp(1.0 / (1 << lattice->variants()));
        bin_enc ex = true_ex(halving);
        for(int re = 0; re < (1 << lattice->variants()); re++){
            if(re != (1 << lattice->variants())-1){
                Product_lattice* p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                p->update_probs(halving, re, thres_up, thres_lo, dilution);
                Product_lattice* p1 = p->clone(SHALLOW_COPY_PROB_DIST); // potentially switch from model parallelism to data parallelism
                p->posterior_probs(nullptr);
                delete p;
                _children[re] = new Global_tree_mpi(p1, ex, re, k, _curr_stage+1, thres_up, thres_lo, stage, dilution);
            }
            else{ // reuse post_prob_ array in child to save memory
                Product_lattice* p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                _lattice->posterior_probs(nullptr); // detach post_prob_ from current lattice
                p->update_probs_in_place(halving, re, thres_up, thres_lo, dilution);
                Product_lattice* p1 = p->clone(SHALLOW_COPY_PROB_DIST); // potentially switch from model parallelism to data parallelism
                p->posterior_probs(nullptr);
                delete p;
                _children[re] = new Global_tree_mpi(p1, ex, re, k, _curr_stage+1, thres_up, thres_lo, stage, dilution);
            }
        } 
    }
    else{ // clean in advance to save memory
        delete[] lattice->posterior_probs();
        lattice->posterior_probs(nullptr);
    }
}

Global_tree_mpi::Global_tree_mpi(Product_lattice* lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double** __restrict__ dilution, std::chrono::nanoseconds halving_times[], std::chrono::nanoseconds mp_update_times[], std::chrono::nanoseconds dp_update_times[], std::chrono::nanoseconds mp_dp_update_times[]) : Global_tree_mpi(lattice, ex, res, curr_stage){
    auto halving_start = std::chrono::high_resolution_clock::now(), halving_end = halving_start;
    if (!lattice->is_classified() && curr_stage < stage) {
        _children = new Global_tree*[1 << lattice->variants()];
        bin_enc halving = lattice->halving_mp(1.0 / (1 << lattice->variants())); // remember test selection as halving_res will change in depth-frist traversal
        halving_end = std::chrono::high_resolution_clock::now();
        halving_times[_lattice->curr_subjs()] += std::chrono::duration_cast<std::chrono::nanoseconds>(halving_end - halving_start);
        bin_enc ex = true_ex(halving);
        for(int re = 0; re < (1 << lattice->variants()); re++){
            if(re != (1 << lattice->variants())-1){
                auto update_start = std::chrono::high_resolution_clock::now();
                Product_lattice* p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                int old_parallelism = p->parallelism();
                p->update_probs(halving, re, thres_up, thres_lo, dilution);
                int new_parallelism = p->parallelism();
                Product_lattice* p1 = p->clone(SHALLOW_COPY_PROB_DIST); // potentially switch from model parallelism to data parallelism
                auto update_end = std::chrono::high_resolution_clock::now();
                if(old_parallelism == MODEL_PARALLELISM && new_parallelism == MODEL_PARALLELISM)
                    mp_update_times[_lattice->curr_subjs()] += std::chrono::duration_cast<std::chrono::nanoseconds>(update_end - update_start);
                else if (old_parallelism == MODEL_PARALLELISM && new_parallelism == DATA_PARALLELISM)
                    mp_dp_update_times[_lattice->curr_subjs()] += std::chrono::duration_cast<std::chrono::nanoseconds>(update_end - update_start);
                else if (old_parallelism == DATA_PARALLELISM)
                    dp_update_times[_lattice->curr_subjs()] += std::chrono::duration_cast<std::chrono::nanoseconds>(update_end - update_start);
                p->posterior_probs(nullptr);
                delete p;
                _children[re] = new Global_tree_mpi(p1, ex, re, k, _curr_stage+1, thres_up, thres_lo, stage, dilution, halving_times, mp_update_times, dp_update_times, mp_dp_update_times);
            }
            else{ // reuse post_prob_ array in child to save memory
                auto update_start = std::chrono::high_resolution_clock::now();
                Product_lattice* p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                _lattice->posterior_probs(nullptr); // detach post_prob_ from current lattice
                int old_parallelism = p->parallelism();
                p->update_probs_in_place(halving, re, thres_up, thres_lo, dilution);
                int new_parallelism = p->parallelism();
                Product_lattice* p1 = p->clone(SHALLOW_COPY_PROB_DIST); // potentially switch from model parallelism to data parallelism
                auto update_end = std::chrono::high_resolution_clock::now();
                if(old_parallelism == MODEL_PARALLELISM && new_parallelism == MODEL_PARALLELISM)
                    mp_update_times[_lattice->curr_subjs()] += std::chrono::duration_cast<std::chrono::nanoseconds>(update_end - update_start);
                else if (old_parallelism == MODEL_PARALLELISM && new_parallelism == DATA_PARALLELISM)
                    mp_dp_update_times[_lattice->curr_subjs()] += std::chrono::duration_cast<std::chrono::nanoseconds>(update_end - update_start);
                else if (old_parallelism == DATA_PARALLELISM)
                    dp_update_times[_lattice->curr_subjs()] += std::chrono::duration_cast<std::chrono::nanoseconds>(update_end - update_start); 
                p->posterior_probs(nullptr);
                delete p;
                _children[re] = new Global_tree_mpi(p1, ex, re, k, _curr_stage+1, thres_up, thres_lo, stage, dilution, halving_times, mp_update_times, dp_update_times, mp_dp_update_times);
            }
        } 
    }
    else{ // clean in advance to save memory
        delete[] lattice->posterior_probs();
        lattice->posterior_probs(nullptr);
    }
}