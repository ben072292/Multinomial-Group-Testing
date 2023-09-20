#include "global_tree.hpp"

int Global_tree::rank = -1;
int Global_tree::world_size = 0;
Tree_perf *Global_tree::tree_perf;
MPI_Datatype Global_tree::tree_stat_type;
MPI_Op Global_tree::tree_stat_op;

/**
 * Single tree without perf
 */
Global_tree::Global_tree(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage) : Global_tree(lattice, ex, res, curr_stage)
{
    if (!lattice->is_classified() && curr_stage < _search_depth)
    {
        _children = new Tree *[1 << variants()]
        { nullptr };
        bin_enc halving = lattice->halving(1.0 / (1 << variants()));
        bin_enc ex = true_ex(halving);
        for (int re = 0; re < (1 << variants()); re++)
        {
            Product_lattice *p = lattice->clone(SHALLOW_COPY_PROB_DIST);
            if (re == (1 << variants()) - 1)
            {                                       // reuse _post_probs in child to save memory
                _lattice->posterior_probs(nullptr); // detach _post_probs from current lattice
                p->update_probs_in_place(halving, re, _dilution);
            }
            else
                p->update_probs(halving, re, _dilution);
            p->update_metadata_with_shrinking(_thres_up, _thres_lo);
            Product_lattice *p1 = p->clone(SHALLOW_COPY_PROB_DIST); // potentially switch from model parallelism to data parallelism
            p->posterior_probs(nullptr);                            // detach _post_probs
            delete p;
            _children[re] = new Global_tree(p1, ex, re, k, _curr_stage + 1);
        }
    }
    else
    { // clean in advance to save memory
        destroy_posterior_probs();
    }
}

/**
 * Single tree with Perf
 */
Global_tree::Global_tree(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, bool perf) : Global_tree(lattice, ex, res, curr_stage)
{
    auto halving_start = std::chrono::high_resolution_clock::now(), halving_end = halving_start;
    if (!lattice->is_classified() && curr_stage < _search_depth)
    {
        _children = new Tree *[1 << variants()]
        { nullptr };
        bin_enc halving = lattice->halving(1.0 / (1 << variants())); // remember test selection as halving_res will change in depth-frist traversal
        halving_end = std::chrono::high_resolution_clock::now();
        tree_perf->accumulate_halving_time(lattice->curr_subjs(), std::chrono::duration_cast<std::chrono::nanoseconds>(halving_end - halving_start));
        bin_enc ex = true_ex(halving);
        for (int re = 0; re < (1 << variants()); re++)
        {
            auto update_start = std::chrono::high_resolution_clock::now();
            Product_lattice *p = lattice->clone(SHALLOW_COPY_PROB_DIST);
            int old_parallelism = p->parallelism();
            if (re == (1 << variants()) - 1)
            {                                       // reuse _post_probs in child to save memory
                _lattice->posterior_probs(nullptr); // detach _post_probs from current lattice
                p->update_probs_in_place(halving, re, _dilution);
            }
            else
                p->update_probs(halving, re, _dilution);
            auto update_end = std::chrono::high_resolution_clock::now();
            auto shrink_start = std::chrono::high_resolution_clock::now();
            p->update_metadata_with_shrinking(_thres_up, _thres_lo);
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
            _children[re] = new Global_tree(p1, ex, re, k, _curr_stage + 1, true);
        }
    }
    else
    { // clean in advance to save memory
        destroy_posterior_probs();
    }
}

Global_tree::Global_tree(const Tree &other, bool deep) : Global_tree_intra(other, false)
{
    if (deep)
    {
        if (other.children() != nullptr)
        {
            _children = new Tree *[1 << variants()]
            { nullptr };
            for (int i = 0; i < (1 << variants()); i++)
            {
                _children[i] = new Global_tree(*(other.children()[i]), deep); // TBD: this downcast is problematic, use virtual clone and create functions
            }
        }
    }
}

void Global_tree::MPI_Global_tree_Initialize(int subjs, int k)
{
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Tree_stat::create_tree_stat_type(&tree_stat_type, _search_depth, k);
    tree_perf = new Tree_perf(subjs);
    MPI_Type_commit(&tree_stat_type);
    MPI_Op_create((MPI_User_function *)&Tree_stat::tree_stat_reduce, true, &tree_stat_op);
}

void Global_tree::MPI_Global_tree_Free()
{
    // Free datatype
    MPI_Type_free(&tree_stat_type);
    // Free tree stat reduce op
    MPI_Op_free(&tree_stat_op);
    // Delete node tracer
    delete tree_perf;
}