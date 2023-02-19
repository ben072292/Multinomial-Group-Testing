#include "global_tree_mp.hpp"

Global_tree_mp::Global_tree_mp(Product_lattice* lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double** __restrict__ dilution) : Global_tree_mp(lattice, ex, res, curr_stage){
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
                _children[re] = new Global_tree_mp(p1, ex, re, k, _curr_stage+1, thres_up, thres_lo, stage, dilution);
            }
            else{ // reuse post_prob_ array in child to save memory
                Product_lattice* p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                _lattice->posterior_probs(nullptr); // detach post_prob_ from current lattice
                p->update_probs_in_place(halving, re, thres_up, thres_lo, dilution);
                Product_lattice* p1 = p->clone(SHALLOW_COPY_PROB_DIST); // potentially switch from model parallelism to data parallelism
                p->posterior_probs(nullptr);
                delete p;
                _children[re] = new Global_tree_mp(p1, ex, re, k, _curr_stage+1, thres_up, thres_lo, stage, dilution);
            }
        } 
    }
    else{ // clean in advance to save memory
        delete[] lattice->posterior_probs();
        lattice->posterior_probs(nullptr);
    }
}

Global_tree_mp::Global_tree_mp(Product_lattice* lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double** __restrict__ dilution, std::chrono::nanoseconds mpi_times[]) : Global_tree_mp(lattice, ex, res, curr_stage){
    auto start = std::chrono::high_resolution_clock::now(), end = start;
    if (!lattice->is_classified() && curr_stage < stage) {
        _children = new Global_tree*[1 << lattice->variants()];
        bin_enc halving = lattice->halving_mp(1.0 / (1 << lattice->variants())); // remember test selection as halving_res will change in depth-frist traversal
        end = std::chrono::high_resolution_clock::now();
        mpi_times[_lattice->curr_subjs()] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        bin_enc ex = true_ex(halving);
        for(int re = 0; re < (1 << lattice->variants()); re++){
            if(re != (1 << lattice->variants())-1){
                Product_lattice* p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                p->update_probs(halving, re, thres_up, thres_lo, dilution);
                Product_lattice* p1 = p->clone(SHALLOW_COPY_PROB_DIST); // potentially switch from model parallelism to data parallelism
                p->posterior_probs(nullptr);
                delete p;
                _children[re] = new Global_tree_mp(p1, ex, re, k, _curr_stage+1, thres_up, thres_lo, stage, dilution, mpi_times);
            }
            else{ // reuse post_prob_ array in child to save memory
                Product_lattice* p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                _lattice->posterior_probs(nullptr); // detach post_prob_ from current lattice
                p->update_probs_in_place(halving, re, thres_up, thres_lo, dilution);
                Product_lattice* p1 = p->clone(SHALLOW_COPY_PROB_DIST); // potentially switch from model parallelism to data parallelism
                p->posterior_probs(nullptr);
                delete p;
                _children[re] = new Global_tree_mp(p1, ex, re, k, _curr_stage+1, thres_up, thres_lo, stage, dilution, mpi_times);
            }
        } 
    }
    else{ // clean in advance to save memory
        delete[] lattice->posterior_probs();
        lattice->posterior_probs(nullptr);
    }
}