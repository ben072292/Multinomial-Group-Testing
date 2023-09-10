#include "doubly_linked_global_tree_mpi.hpp"
#include <vector>

/**
 * Single tree without perf
 */
Doubly_linked_global_tree_mpi::Doubly_linked_global_tree_mpi(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double **__restrict__ dilution, Global_tree *parent) : Doubly_linked_global_tree_mpi(lattice, ex, res, curr_stage, parent)
{
    if (!lattice->is_classified() && curr_stage < stage)
    {
        _children = new Global_tree *[1 << lattice->variants()];
        bin_enc halving = lattice->halving(1.0 / (1 << lattice->variants()));
        bin_enc ex = true_ex(halving);
        for (int re = 0; re < (1 << lattice->variants()); re++)
        {
            Product_lattice *p = lattice->clone(SHALLOW_COPY_PROB_DIST);
            if (re != (1 << lattice->variants()) - 1)
            {                                       // reuse _post_probs in child to save memory
                _lattice->posterior_probs(nullptr); // detach _post_probs from current lattice
                p->update_probs_in_place(halving, re, dilution);
            }
            else
                p->update_probs(halving, re, dilution);
            p->update_metadata_with_shrinking(thres_up, thres_lo);
            Product_lattice *p1 = p->clone(SHALLOW_COPY_PROB_DIST); // potentially switch from model parallelism to data parallelism
            p->posterior_probs(nullptr);                            // detach _post_probs
            delete p;
            _children[re] = new Doubly_linked_global_tree_mpi(p1, ex, re, k, _curr_stage + 1, thres_up, thres_lo, stage, dilution, this);
        }
    }
    else
    { // clean in advance to save memory
        delete[] lattice->posterior_probs();
        lattice->posterior_probs(nullptr);
    }
}

/**
 * Single tree with Perf
 */
Doubly_linked_global_tree_mpi::Doubly_linked_global_tree_mpi(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double **__restrict__ dilution, bool perf, Global_tree *parent) : Doubly_linked_global_tree_mpi(lattice, ex, res, curr_stage, parent)
{
    auto halving_start = std::chrono::high_resolution_clock::now(), halving_end = halving_start;
    if (!lattice->is_classified() && curr_stage < stage)
    {
        _children = new Global_tree *[1 << lattice->variants()];
        bin_enc halving = lattice->halving(1.0 / (1 << lattice->variants())); // remember test selection as halving_res will change in depth-frist traversal
        halving_end = std::chrono::high_resolution_clock::now();
        tree_perf->accumulate_halving_time(lattice->curr_subjs(), std::chrono::duration_cast<std::chrono::nanoseconds>(halving_end - halving_start));
        bin_enc ex = true_ex(halving);
        for (int re = 0; re < (1 << lattice->variants()); re++)
        {
            auto update_start = std::chrono::high_resolution_clock::now();
            Product_lattice *p = lattice->clone(SHALLOW_COPY_PROB_DIST);
            int old_parallelism = p->parallelism();
            if (re == (1 << lattice->variants()) - 1)
            {                                       // reuse _post_probs in child to save memory
                _lattice->posterior_probs(nullptr); // detach _post_probs from current lattice
                p->update_probs_in_place(halving, re, dilution);
            }
            else
                p->update_probs(halving, re, dilution);
            auto update_end = std::chrono::high_resolution_clock::now();
            auto shrink_start = std::chrono::high_resolution_clock::now();
            p->update_metadata_with_shrinking(thres_up, thres_lo);
            auto shrink_end = std::chrono::high_resolution_clock::now();
            int new_parallelism = p->parallelism();
            Product_lattice *p1;
            if (old_parallelism != new_parallelism)
            {
                p1 = p->clone(SHALLOW_COPY_PROB_DIST); // switch type
                p->posterior_probs(nullptr);           // detach _post_probs
                delete p;
            }
            else
            {
                p1 = p;
                p = nullptr;
            }
            tree_perf->accumulate_count(_lattice->curr_subjs(), p1->curr_subjs());
            tree_perf->accumulate_update_time(_lattice->curr_subjs(), p1->curr_subjs(), (update_end - update_start));
            if (old_parallelism == MODEL_PARALLELISM && new_parallelism == MODEL_PARALLELISM)
                tree_perf->accumulate_mp_time(_lattice->curr_subjs(), p1->curr_subjs(), (shrink_end - shrink_start));
            else if (old_parallelism == MODEL_PARALLELISM && new_parallelism == DATA_PARALLELISM)
                tree_perf->accumulate_mp_dp_time(_lattice->curr_subjs(), p1->curr_subjs(), (shrink_end - shrink_start));
            else if (old_parallelism == DATA_PARALLELISM)
                tree_perf->accumulate_dp_time(_lattice->curr_subjs(), p1->curr_subjs(), (shrink_end - shrink_start));
            _children[re] = new Doubly_linked_global_tree_mpi(p1, ex, re, k, _curr_stage + 1, thres_up, thres_lo, stage, dilution, true, this);
        }
    }
    else
    { // clean in advance to save memory
        delete[] lattice->posterior_probs();
        lattice->posterior_probs(nullptr);
    }
}

/**
 * Fusion Tree
 */
Doubly_linked_global_tree_mpi::Doubly_linked_global_tree_mpi(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double *pi0, double **__restrict__ dilution, double prun_thres_sum, double curr_prun_thres_sum, double prun_thres, Global_tree *parent) : Doubly_linked_global_tree_mpi(lattice, ex, res, curr_stage, parent)
{
    sequence_tracer[curr_stage] = this;
    // log_debug("sequence: %d %d %d.", curr_stage, sequence_tracer[curr_stage], sequence_tracer[curr_stage]->ex_res());
    auto halving_start = std::chrono::high_resolution_clock::now(), halving_end = halving_start;
    if (!lattice->is_classified() && curr_stage < stage)
    {
        _children = new Global_tree *[1 << lattice->variants()];
        bin_enc halving = lattice->halving(1.0 / (1 << lattice->variants())); // remember test selection as halving_res will change in depth-frist traversal
        halving_end = std::chrono::high_resolution_clock::now();
        tree_perf->accumulate_halving_time(lattice->curr_subjs(), (halving_end - halving_start));
        bin_enc ex = true_ex(halving);
        for (int re = 0; re < (1 << lattice->variants()); re++)
        {
            // Fusion tree pruning process
            double fusion_tree_branch_prob = fusion_branch_prob(ex, re, pi0, dilution);
            MPI_Allreduce(MPI_IN_PLACE, &fusion_tree_branch_prob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            // log_debug("Stage: %d, Prob: %f.", curr_stage, fusion_tree_branch_prob);
            if (curr_prun_thres_sum < prun_thres_sum && fusion_tree_branch_prob < prun_thres && lattice->curr_atoms() >= lattice->orig_atoms()) // can be pruned through fusion tree
            {
                curr_prun_thres_sum += fusion_tree_branch_prob;
                auto update_start = std::chrono::high_resolution_clock::now();
                Product_lattice *p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                if (re == (1 << lattice->variants()) - 1)
                {                                       // reuse _post_probs in child to save memory
                    _lattice->posterior_probs(nullptr); // detach _post_probs from current lattice
                    p->update_probs_in_place(halving, re, dilution);
                }
                else
                    p->update_probs(halving, re, dilution);
                p->update_metadata(thres_up, thres_lo); // no need to shrink as only metadata matters
                delete[] p->posterior_probs();
                p->posterior_probs(nullptr);
                _children[re] = new Doubly_linked_global_tree_mpi(p, ex, re, curr_stage, this);
                auto update_end = std::chrono::high_resolution_clock::now();
                tree_perf->accumulate_update_time(_lattice->curr_subjs(), p->curr_subjs(), (update_end - update_start));
            }
            else // cannot be pruned
            {
                auto update_start = std::chrono::high_resolution_clock::now();
                Product_lattice *p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                int old_parallelism = p->parallelism();
                if (re == (1 << lattice->variants()) - 1)
                {                                       // reuse _post_probs in child to save memory
                    _lattice->posterior_probs(nullptr); // detach _post_probs from current lattice
                    p->update_probs_in_place(halving, re, dilution);
                }
                else
                    p->update_probs(halving, re, dilution);
                auto update_end = std::chrono::high_resolution_clock::now();
                auto shrink_start = std::chrono::high_resolution_clock::now();
                p->update_metadata_with_shrinking(thres_up, thres_lo);
                auto shrink_end = std::chrono::high_resolution_clock::now();
                int new_parallelism = p->parallelism();
                Product_lattice *p1;
                if (old_parallelism != new_parallelism)
                {
                    p1 = p->clone(SHALLOW_COPY_PROB_DIST); // switch type
                    p->posterior_probs(nullptr);           // detach _post_probs
                    delete p;
                }
                else
                {
                    p1 = p;
                    p = nullptr;
                }
                tree_perf->accumulate_count(_lattice->curr_subjs(), p1->curr_subjs());
                tree_perf->accumulate_update_time(_lattice->curr_subjs(), p1->curr_subjs(), (update_end - update_start));
                if (old_parallelism == MODEL_PARALLELISM && new_parallelism == MODEL_PARALLELISM)
                    tree_perf->accumulate_mp_time(_lattice->curr_subjs(), p1->curr_subjs(), (shrink_end - shrink_start));
                else if (old_parallelism == MODEL_PARALLELISM && new_parallelism == DATA_PARALLELISM)
                    tree_perf->accumulate_mp_dp_time(_lattice->curr_subjs(), p1->curr_subjs(), (shrink_end - shrink_start));
                else if (old_parallelism == DATA_PARALLELISM)
                    tree_perf->accumulate_dp_time(_lattice->curr_subjs(), p1->curr_subjs(), (shrink_end - shrink_start));
                _children[re] = new Doubly_linked_global_tree_mpi(p1, ex, re, k, _curr_stage + 1, thres_up, thres_lo, stage, pi0, dilution, prun_thres_sum, curr_prun_thres_sum, prun_thres, this);
            }
        }
    }
    else
    { // clean in advance to save memory
        delete[] lattice->posterior_probs();
        lattice->posterior_probs(nullptr);
        _lattice->posterior_probs(nullptr);
    }
}