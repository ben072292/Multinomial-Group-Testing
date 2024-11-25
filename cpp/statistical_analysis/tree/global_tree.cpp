#include "tree.hpp"

int Global_tree::rank = -1;
int Global_tree::world_size = 0;
Tree_perf *Global_tree::tree_perf;
MPI_Datatype Global_tree::tree_stat_type;
MPI_Op Global_tree::tree_stat_op;

Global_tree::Global_tree(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage)
    : Global_tree(lattice, ex, res, curr_stage)
{
#ifdef ENABLE_PERF
    auto BBPA_start = std::chrono::high_resolution_clock::now();
    auto BBPA_end = BBPA_start;
#endif

    if (!lattice->is_classified() && curr_stage < _search_depth)
    {
        _children = new Tree *[1 << variants()]{nullptr};

#ifdef ENABLE_PERF
        bin_enc BBPA = lattice->BBPA(1.0 / (1 << variants()));
        BBPA_end = std::chrono::high_resolution_clock::now();
        tree_perf->accumulate_BBPA_time(
            lattice->curr_subjs(),
            std::chrono::duration_cast<std::chrono::nanoseconds>(BBPA_end - BBPA_start));
#else
        bin_enc BBPA = lattice->BBPA(1.0 / (1 << variants()));
#endif

        bin_enc ex = true_ex(BBPA);

        for (int re = 0; re < (1 << variants()); re++)
        {
#ifdef ENABLE_PERF
            auto update_start = std::chrono::high_resolution_clock::now();
#endif
            Product_lattice *p = lattice->clone(SHALLOW_COPY_PROB_DIST);

#ifdef ENABLE_PERF
            int old_parallelism = p->parallelism();
#endif

            if (re == (1 << variants()) - 1)
            {
                _lattice->posterior_probs(nullptr); // Detach _post_probs from current lattice
                p->update_probs_in_place(BBPA, re, _dilution);
            }
            else
            {
                p->update_probs(BBPA, re, _dilution);
            }

#ifdef ENABLE_PERF
            auto update_end = std::chrono::high_resolution_clock::now();
            auto shrink_start = std::chrono::high_resolution_clock::now();
            auto classification_identification_start = std::chrono::high_resolution_clock::now();
            auto classification_identification_end = std::chrono::high_resolution_clock::now();
#endif
            if (p->update_metadata_with_shrinking(_thres_up, _thres_lo
            #ifdef ENABLE_PERF
, &classification_identification_start, &classification_identification_end
            #endif
            ))
            {
                p = p->to_local();
            }
#ifdef ENABLE_PERF
            auto shrink_end = std::chrono::high_resolution_clock::now();
            int new_parallelism = p->parallelism();

            tree_perf->accumulate_count(_lattice->curr_subjs(), p->curr_subjs());
            tree_perf->accumulate_update_time(
                _lattice->curr_subjs(),
                p->curr_subjs(),
                (update_end - update_start));

            tree_perf->accumulate_classification_identification_time(
                _lattice->curr_subjs(),
                p->curr_subjs(),
                (classification_identification_end - classification_identification_start));

            if (old_parallelism == DIST_MODEL && new_parallelism == DIST_MODEL)
            {
                tree_perf->accumulate_parallel_shrinking_time(
                    _lattice->curr_subjs(),
                    p->curr_subjs(),
                    (shrink_end - shrink_start));
            }
            else if (old_parallelism == DIST_MODEL && new_parallelism == REPL_MODEL)
            {
                tree_perf->accumulate_parallel_to_serial_transition_time(
                    _lattice->curr_subjs(),
                    p->curr_subjs(),
                    (shrink_end - shrink_start));
            }
            else if (old_parallelism == REPL_MODEL)
            {
                tree_perf->accumulate_serial_shrinking_time(
                    _lattice->curr_subjs(),
                    p->curr_subjs(),
                    (shrink_end - shrink_start));
            }
#endif
            _children[re] = new Global_tree(p, ex, re, k, _curr_stage + 1);
        }
    }
    else
    { // Clean in advance to save memory
        destroy_posterior_probs();
    }
}

Global_tree::Global_tree(const Tree &other, bool deep) : Global_tree_intra(other, false)
{
    if (deep)
    {
        if (other.children() != nullptr)
        {
            _children = new Tree *[1 << variants()]{nullptr};
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