#include "global_tree_mpi.hpp"

int Global_tree_mpi::rank = -1;
int Global_tree_mpi::world_size = 0;
Tree_perf* Global_tree_mpi::tree_perf;
Global_tree_mpi **Global_tree_mpi::sequence_tracer = nullptr;
MPI_Datatype Global_tree_mpi::tree_stat_type;
MPI_Op Global_tree_mpi::tree_stat_op;

/**
 * Single tree without perf
*/
Global_tree_mpi::Global_tree_mpi(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double **__restrict__ dilution) : Global_tree_mpi(lattice, ex, res, curr_stage)
{
    if (!lattice->is_classified() && curr_stage < stage)
    {
        _children = new Global_tree *[1 << lattice->variants()];
        bin_enc halving = lattice->halving(1.0 / (1 << lattice->variants()));
        bin_enc ex = true_ex(halving);
        for (int re = 0; re < (1 << lattice->variants()); re++)
        {
            if (re != (1 << lattice->variants()) - 1)
            {
                Product_lattice *p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                p->update_probs(halving, re, dilution);
                p->update_metadata_with_shrinking(thres_up, thres_lo);
                Product_lattice *p1 = p->clone(SHALLOW_COPY_PROB_DIST); // potentially switch from model parallelism to data parallelism
                p->posterior_probs(nullptr);
                delete p;
                _children[re] = new Global_tree_mpi(p1, ex, re, k, _curr_stage + 1, thres_up, thres_lo, stage, dilution);
            }
            else
            { // reuse post_prob_ array in child to save memory
                Product_lattice *p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                _lattice->posterior_probs(nullptr); // detach post_prob_ from current lattice
                p->update_probs_in_place(halving, re, dilution);
                p->update_metadata_with_shrinking(thres_up, thres_lo);
                Product_lattice *p1 = p->clone(SHALLOW_COPY_PROB_DIST); // potentially switch from model parallelism to data parallelism
                p->posterior_probs(nullptr);
                delete p;
                _children[re] = new Global_tree_mpi(p1, ex, re, k, _curr_stage + 1, thres_up, thres_lo, stage, dilution);
            }
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
Global_tree_mpi::Global_tree_mpi(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double **__restrict__ dilution, bool perf) : Global_tree_mpi(lattice, ex, res, curr_stage)
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
            if (re != (1 << lattice->variants()) - 1)
            {
                Product_lattice *p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                int old_parallelism = p->parallelism();
                auto update_start = std::chrono::high_resolution_clock::now();
                p->update_probs(halving, re, dilution);
                auto update_end = std::chrono::high_resolution_clock::now();
                auto shrink_start = std::chrono::high_resolution_clock::now();
                p->update_metadata_with_shrinking(thres_up, thres_lo);
                auto shrink_end = std::chrono::high_resolution_clock::now();
                int new_parallelism = p->parallelism();
                if (old_parallelism != new_parallelism)
                {
                    p = p->clone(SHALLOW_COPY_PROB_DIST);
                }
                tree_perf->accumulate_count(_lattice->curr_subjs(), p->curr_subjs());
                tree_perf->accumulate_update_time(_lattice->curr_subjs(), p->curr_subjs(), std::chrono::duration_cast<std::chrono::nanoseconds>(update_end - update_start));
                if (old_parallelism == MODEL_PARALLELISM && new_parallelism == MODEL_PARALLELISM)
                    tree_perf->accumulate_mp_time(_lattice->curr_subjs(), p->curr_subjs(), std::chrono::duration_cast<std::chrono::nanoseconds>(shrink_end - shrink_start));
                else if (old_parallelism == MODEL_PARALLELISM && new_parallelism == DATA_PARALLELISM)
                    tree_perf->accumulate_mp_dp_time(_lattice->curr_subjs(), p->curr_subjs(), std::chrono::duration_cast<std::chrono::nanoseconds>(shrink_end - shrink_start));
                else if (old_parallelism == DATA_PARALLELISM)
                    tree_perf->accumulate_dp_time(_lattice->curr_subjs(), p->curr_subjs(), std::chrono::duration_cast<std::chrono::nanoseconds>(shrink_end - shrink_start));
                _children[re] = new Global_tree_mpi(p, ex, re, k, _curr_stage + 1, thres_up, thres_lo, stage, dilution, true);
            }
            else
            { // reuse post_prob_ array in child to save memory
                auto update_start = std::chrono::high_resolution_clock::now();
                Product_lattice *p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                _lattice->posterior_probs(nullptr); // detach post_prob_ from current lattice
                int old_parallelism = p->parallelism();
                p->update_probs_in_place(halving, re, dilution);
                auto update_end = std::chrono::high_resolution_clock::now();
                auto shrink_start = std::chrono::high_resolution_clock::now();
                p->update_metadata_with_shrinking(thres_up, thres_lo);
                auto shrink_end = std::chrono::high_resolution_clock::now();
                int new_parallelism = p->parallelism();
                if (old_parallelism != new_parallelism)
                {
                    p = p->clone(SHALLOW_COPY_PROB_DIST);
                }
                tree_perf->accumulate_count(_lattice->curr_subjs(), p->curr_subjs());
                tree_perf->accumulate_update_time(_lattice->curr_subjs(), p->curr_subjs(), std::chrono::duration_cast<std::chrono::nanoseconds>(update_end - update_start));
                if (old_parallelism == MODEL_PARALLELISM && new_parallelism == MODEL_PARALLELISM)
                    tree_perf->accumulate_mp_time(_lattice->curr_subjs(), p->curr_subjs(), std::chrono::duration_cast<std::chrono::nanoseconds>(shrink_end - shrink_start));
                else if (old_parallelism == MODEL_PARALLELISM && new_parallelism == DATA_PARALLELISM)
                    tree_perf->accumulate_mp_dp_time(_lattice->curr_subjs(), p->curr_subjs(), std::chrono::duration_cast<std::chrono::nanoseconds>(shrink_end - shrink_start));
                else if (old_parallelism == DATA_PARALLELISM)
                    tree_perf->accumulate_dp_time(_lattice->curr_subjs(), p->curr_subjs(), std::chrono::duration_cast<std::chrono::nanoseconds>(shrink_end - shrink_start));
                _children[re] = new Global_tree_mpi(p, ex, re, k, _curr_stage + 1, thres_up, thres_lo, stage, dilution, true);
            }
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
Global_tree_mpi::Global_tree_mpi(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double *pi0, double **__restrict__ dilution) : Global_tree_mpi(lattice, ex, res, curr_stage)
{
    sequence_tracer[curr_stage] = this;
    log_debug("sequence: %d %d %d.", curr_stage, sequence_tracer[curr_stage], sequence_tracer[curr_stage]->ex_res());
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
            log_debug("Stage: %d, Prob: %f.", curr_stage, fusion_tree_branch_prob);
            if (fusion_tree_branch_prob < (1e-4 / (std::pow(4, curr_stage))))
            {
                _children[re] = nullptr;
                // clean ``this'' if its last child has been evaluated
                if (re == (1 << lattice->variants()) - 1)
                {
                    delete[] lattice->posterior_probs();
                    lattice->posterior_probs(nullptr);
                    _lattice->posterior_probs(nullptr);
                }
                continue;
            }

            if (re != (1 << lattice->variants()) - 1)
            {
                auto update_start = std::chrono::high_resolution_clock::now();
                Product_lattice *p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                int old_parallelism = p->parallelism();
                p->update_probs(halving, re, dilution);
                auto update_end = std::chrono::high_resolution_clock::now();
                auto shrink_start = std::chrono::high_resolution_clock::now();
                p->update_metadata_with_shrinking(thres_up, thres_lo);
                auto shrink_end = std::chrono::high_resolution_clock::now();
                int new_parallelism = p->parallelism();
                if (old_parallelism != new_parallelism)
                {
                    p = p->clone(SHALLOW_COPY_PROB_DIST);
                }
                tree_perf->accumulate_count(_lattice->curr_subjs(), p->curr_subjs());
                tree_perf->accumulate_update_time(_lattice->curr_subjs(), p->curr_subjs(), (update_end - update_start));
                if (old_parallelism == MODEL_PARALLELISM && new_parallelism == MODEL_PARALLELISM)
                    tree_perf->accumulate_mp_time(_lattice->curr_subjs(), p->curr_subjs(), (shrink_end - shrink_start));
                else if (old_parallelism == MODEL_PARALLELISM && new_parallelism == DATA_PARALLELISM)
                    tree_perf->accumulate_mp_dp_time(_lattice->curr_subjs(), p->curr_subjs(), (shrink_end - shrink_start));
                else if (old_parallelism == DATA_PARALLELISM)
                    tree_perf->accumulate_dp_time(_lattice->curr_subjs(), p->curr_subjs(), (shrink_end - shrink_start));
                _children[re] = new Global_tree_mpi(p, ex, re, k, _curr_stage + 1, thres_up, thres_lo, stage, pi0, dilution);
            }
            else
            { // reuse post_prob_ array in child to save memory
                auto update_start = std::chrono::high_resolution_clock::now();
                Product_lattice *p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                _lattice->posterior_probs(nullptr); // detach post_prob_ from current lattice
                int old_parallelism = p->parallelism();
                p->update_probs(halving, re, dilution);
                auto update_end = std::chrono::high_resolution_clock::now();
                auto shrink_start = std::chrono::high_resolution_clock::now();
                p->update_metadata_with_shrinking(thres_up, thres_lo);
                auto shrink_end = std::chrono::high_resolution_clock::now();
                int new_parallelism = p->parallelism();
                if (old_parallelism != new_parallelism)
                {
                    p = p->clone(SHALLOW_COPY_PROB_DIST);
                }
                tree_perf->accumulate_count(_lattice->curr_subjs(), p->curr_subjs());
                tree_perf->accumulate_update_time(_lattice->curr_subjs(), p->curr_subjs(), (update_end - update_start));
                if (old_parallelism == MODEL_PARALLELISM && new_parallelism == MODEL_PARALLELISM)
                    tree_perf->accumulate_mp_time(_lattice->curr_subjs(), p->curr_subjs(), (shrink_end - shrink_start));
                else if (old_parallelism == MODEL_PARALLELISM && new_parallelism == DATA_PARALLELISM)
                    tree_perf->accumulate_mp_dp_time(_lattice->curr_subjs(), p->curr_subjs(), (shrink_end - shrink_start));
                else if (old_parallelism == DATA_PARALLELISM)
                    tree_perf->accumulate_dp_time(_lattice->curr_subjs(), p->curr_subjs(), (shrink_end - shrink_start));
                _children[re] = new Global_tree_mpi(p, ex, re, k, _curr_stage + 1, thres_up, thres_lo, stage, pi0, dilution);
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

double Global_tree_mpi::fusion_branch_prob(int ex, int res, double *pi0, double **dilution)
{
    double ret = 0.0;
    for (int i = (1 << _lattice->orig_atoms()) / world_size * rank; i < (1 << _lattice->orig_atoms()) / world_size * (rank + 1); i++)
    {
        double coef = _lattice->prior_prob(i, pi0);
        double temp_branch_prob = 1.0;
        for (int j = 1; j <= _curr_stage; j++)
        {
            temp_branch_prob *= _lattice->response_prob(sequence_tracer[j]->ex(), sequence_tracer[j]->ex_res(), i, dilution);
        }
        temp_branch_prob *= _lattice->response_prob(ex, res, i, dilution);
        ret += temp_branch_prob * coef;
    }
    return ret;
}

void Global_tree_mpi::MPI_Global_tree_Initialize(int subjs, int depth, int k)
{
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Initialize node tracer
    sequence_tracer = new Global_tree_mpi *[depth + 1];

    Tree_stat::create_tree_stat_type(&tree_stat_type, depth, k);
    tree_perf = new Tree_perf(subjs);
    MPI_Type_commit(&tree_stat_type);
    MPI_Op_create((MPI_User_function *)&Tree_stat::tree_stat_reduce, true, &tree_stat_op);
}

void Global_tree_mpi::MPI_Global_tree_Free()
{
    // Free datatype
    MPI_Type_free(&tree_stat_type);
    // Free tree stat reduce op
    MPI_Op_free(&tree_stat_op);
    // Delete node tracer
    delete[] sequence_tracer;
    sequence_tracer = nullptr;

    delete tree_perf;
}