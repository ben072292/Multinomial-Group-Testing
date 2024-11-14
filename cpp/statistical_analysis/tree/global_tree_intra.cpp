#include "tree.hpp"

Global_tree_intra::Global_tree_intra(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage) : Global_tree_intra(lattice, ex, res, curr_stage)
{
    if (!lattice->is_classified() && curr_stage < _search_depth)
    {
        _children = new Tree *[1 << variants()]
        { nullptr };
        // int BBPA = lattice->BBPA(1.0 / (1 << lattice->variant()));
        bin_enc BBPA = lattice->BBPA(1.0 / (1 << variants())); // openmp
        bin_enc ex = true_ex(BBPA);                                // full-sized experiment should be generated before posterior probability distribution is updated, because unupdated clas_subj_ should be used to calculate the correct value
        for (bin_enc re = 0; re < (1 << variants()); re++)
        {
            Product_lattice *p = lattice->clone(SHALLOW_COPY_PROB_DIST);
            if (re != (1 << variants()) - 1)
            {
                p->update_probs(BBPA, re, _dilution);
            }
            else
            {                                       // reuse post_prob_ array in child to save memory
                _lattice->posterior_probs(nullptr); // detach post_prob_ from current lattice
                p->update_probs_in_place(BBPA, re, _dilution);
            }
            if (p->update_metadata_with_shrinking(_thres_up, _thres_lo))
                p = p->to_local();
            _children[re] = new Global_tree_intra(p, ex, re, k, _curr_stage + 1);
        }
    }
    else
    { // clean in advance to save memory
        destroy_posterior_probs();
    }
}

Global_tree_intra::Global_tree_intra(Product_lattice *lattice, bin_enc ex, bin_enc res, int k, int curr_stage, std::chrono::nanoseconds BBPA_times[]) : Global_tree_intra(lattice, ex, res, curr_stage)
{
    auto start = std::chrono::high_resolution_clock::now(), end = start;
    if (!lattice->is_classified() && curr_stage < _search_depth)
    {
        _children = new Tree *[1 << variants()]
        { nullptr };
        // int BBPA = lattice->BBPA(1.0 / (1 << lattice->variant()));
        bin_enc BBPA = lattice->BBPA(1.0 / (1 << variants())); // openmp
        end = std::chrono::high_resolution_clock::now();
        BBPA_times[_lattice->curr_subjs()] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        bin_enc ex = true_ex(BBPA); // full-sized experiment should be generated before posterior probability distribution is updated, because unupdated clas_subj_ should be used to calculate the correct value
        for (bin_enc re = 0; re < (1 << variants()); re++)
        {
            Product_lattice *p = lattice->clone(SHALLOW_COPY_PROB_DIST);
            if (re != (1 << variants()) - 1)
            {
                p->update_probs(BBPA, re, _dilution);
            }
            else
            {                                       // reuse post_prob_ array in child to save memory
                _lattice->posterior_probs(nullptr); // detach post_prob_ from current lattice
                p->update_probs_in_place(BBPA, re, _dilution);
            }
            if (p->update_metadata_with_shrinking(_thres_up, _thres_lo))
                p = p->to_local();
            _children[re] = new Global_tree_intra(p, ex, re, k, _curr_stage + 1, BBPA_times);
        }
    }
    else
    { // clean in advance to save memory
        destroy_posterior_probs();
    }
}

Global_tree_intra::Global_tree_intra(const Tree &other, bool deep) : Tree(other, false)
{
    if (deep)
    {
        if (other.children() != nullptr)
        {
            _children = new Tree *[1 << variants()]
            { nullptr };
            for (int i = 0; i < (1 << variants()); i++)
            {
                _children[i] = new Global_tree_intra(*(other.children()[i]), deep);
            }
        }
    }
}