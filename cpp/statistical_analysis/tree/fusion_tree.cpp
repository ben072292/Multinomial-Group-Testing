#include "tree.hpp"

Fusion_tree **Fusion_tree::sequence_tracer = nullptr;

/**
 * Fusion Tree
 */
Fusion_tree::Fusion_tree(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double prun_thres_sum, double curr_prun_thres_sum, double prun_thres) : Fusion_tree(lattice, ex, res, curr_stage)
{
    sequence_tracer[curr_stage] = this;
    // log_debug("sequence: %d %d %d.", curr_stage, sequence_tracer[curr_stage], sequence_tracer[curr_stage]->ex_res());
    auto BBPA_start = std::chrono::high_resolution_clock::now(), BBPA_end = BBPA_start;
    if (!lattice->is_classified() && curr_stage < _search_depth)
    {
        _children = new Tree *[1 << variants()]
        { nullptr };
        bin_enc BBPA = lattice->BBPA(1.0 / (1 << variants())); // remember test selection as BBPA_res will change in depth-frist traversal
        BBPA_end = std::chrono::high_resolution_clock::now();
        tree_perf->accumulate_BBPA_time(lattice->curr_subjs(), (BBPA_end - BBPA_start));
        bin_enc ex = true_ex(BBPA);
        for (int re = 0; re < (1 << variants()); re++)
        {
            // Fusion tree pruning process
            double fusion_tree_branch_prob = fusion_branch_prob(ex, re);
            MPI_Allreduce(MPI_IN_PLACE, &fusion_tree_branch_prob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            // log_debug("Stage: %d, Prob: %f.", curr_stage, fusion_tree_branch_prob);
            if (curr_prun_thres_sum < prun_thres_sum && fusion_tree_branch_prob < prun_thres && lattice->curr_atoms() >= lattice->orig_atoms()) // can be pruned through fusion tree
            {
                curr_prun_thres_sum += fusion_tree_branch_prob;
                auto update_start = std::chrono::high_resolution_clock::now();
                Product_lattice *p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                if (re == (1 << variants()) - 1)
                {                                       // reuse _post_probs in child to save memory
                    _lattice->posterior_probs(nullptr); // detach _post_probs from current lattice
                    p->update_probs_in_place(BBPA, re, _dilution);
                }
                else
                    p->update_probs(BBPA, re, _dilution);
                p->update_metadata(_thres_up, _thres_lo); // no need to shrink as only metadata matters
                delete[] p->posterior_probs();
                p->posterior_probs(nullptr);
                _children[re] = new Fusion_tree(p, ex, re, curr_stage);
                auto update_end = std::chrono::high_resolution_clock::now();
                tree_perf->accumulate_update_time(_lattice->curr_subjs(), p->curr_subjs(), (update_end - update_start));
            }
            else // cannot be pruned
            {
                auto update_start = std::chrono::high_resolution_clock::now();
                Product_lattice *p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                int old_parallelism = p->parallelism();
                if (re == (1 << variants()) - 1)
                {                                       // reuse _post_probs in child to save memory
                    _lattice->posterior_probs(nullptr); // detach _post_probs from current lattice
                    p->update_probs_in_place(BBPA, re, _dilution);
                }
                else
                    p->update_probs(BBPA, re, _dilution);
                auto update_end = std::chrono::high_resolution_clock::now();
                auto shrink_start = std::chrono::high_resolution_clock::now();
                if (p->update_metadata_with_shrinking(_thres_up, _thres_lo))
                    p = p->convert_parallelism();
                auto shrink_end = std::chrono::high_resolution_clock::now();
                int new_parallelism = p->parallelism();
                tree_perf->accumulate_count(_lattice->curr_subjs(), p->curr_subjs());
                tree_perf->accumulate_update_time(_lattice->curr_subjs(), p->curr_subjs(), (update_end - update_start));
                if (old_parallelism == DIST_MODEL && new_parallelism == DIST_MODEL)
                    tree_perf->accumulate_mp_time(_lattice->curr_subjs(), p->curr_subjs(), (shrink_end - shrink_start));
                else if (old_parallelism == DIST_MODEL && new_parallelism == REPL_MODEL)
                    tree_perf->accumulate_mp_dp_time(_lattice->curr_subjs(), p->curr_subjs(), (shrink_end - shrink_start));
                else if (old_parallelism == REPL_MODEL)
                    tree_perf->accumulate_dp_time(_lattice->curr_subjs(), p->curr_subjs(), (shrink_end - shrink_start));
                _children[re] = new Fusion_tree(p, ex, re, k, _curr_stage + 1, prun_thres_sum, curr_prun_thres_sum, prun_thres);
            }
        }
    }
    else
    { // clean in advance to save memory
        destroy_posterior_probs();
    }
}

Fusion_tree::Fusion_tree(const Tree &other, bool deep) : Global_tree(other, false)
{
    if (deep)
    {
        if (other.children() != nullptr)
        {
            _children = new Tree *[1 << variants()]
            { nullptr };
            for (int i = 0; i < (1 << variants()); i++)
            {
                _children[i] = new Fusion_tree(*(other.children()[i]), deep); // TBD: this downcast is problematic, use virtual clone and create functions
            }
        }
    }
}

double Fusion_tree::fusion_branch_prob(int ex, int res)
{
    double ret = 0.0;
    for (int i = (1 << _lattice->orig_atoms()) / world_size * rank; i < (1 << _lattice->orig_atoms()) / world_size * (rank + 1); i++)
    {
        double coef = _lattice->prior_prob(i, Product_lattice::pi0());
        double temp_branch_prob = 1.0;
        for (int j = 1; j <= _curr_stage; j++)
        {
            temp_branch_prob *= _lattice->response_prob(sequence_tracer[j]->ex(), sequence_tracer[j]->ex_res(), i, _dilution);
        }
        temp_branch_prob *= _lattice->response_prob(ex, res, i, _dilution);
        ret += temp_branch_prob * coef;
    }
    return ret;
}

void Fusion_tree::MPI_Fusion_tree_Initialize(int subjs, int k)
{
    MPI_Global_tree_Initialize(subjs, k);
    sequence_tracer = new Fusion_tree *[_search_depth + 1];
}

void Fusion_tree::MPI_Fusion_tree_Free()
{
    MPI_Global_tree_Free();
    delete[] sequence_tracer;
    sequence_tracer = nullptr;
}