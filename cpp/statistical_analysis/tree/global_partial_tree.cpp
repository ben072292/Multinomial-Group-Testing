#include "tree.hpp"

Global_partial_tree::Global_partial_tree(const Tree &other, bool deep) : Distributed_tree(other, false)
{
    _BBPA = dynamic_cast<const Global_partial_tree &>(other)._BBPA;
    if (deep)
    {
        if (other.children() != nullptr)
        {
            _children = new Tree *[1 << variants()];
            for (int i = 0; i < (1 << variants()); i++)
            {
                _children[i] = new Global_partial_tree(*(other.children()[i]), deep); // TBD: this downcast is problematic, use virtual clone and create functions
            }
        }
    }
}

void Global_partial_tree::eval(Tree *node, Product_lattice *orig_lattice, bin_enc true_state)
{
    if (!node->lattice()->is_classified() && node->curr_stage() < _search_depth && node->branch_prob() > _thres_branch)
    {
        if (node->children() == nullptr) // no child is explored, allocate children and perform computation
        {
            node->children(1 << node->variants());
            bin_enc BBPA = node->lattice()->BBPA(1.0 / (1 << node->variants()));
            bin_enc ex = node->true_ex(BBPA);

            for (int re = 0; re < (1 << node->variants()); re++)
            {
                double child_prob = node->branch_prob() * orig_lattice->response_prob(ex, re, true_state, _dilution);
                Product_lattice *p = node->lattice()->clone(SHALLOW_COPY_PROB_DIST);
                if (re == (1 << variants()) - 1 && node->curr_stage() != 0) // reuse _post_probs in child to save memory
                {
                    node->lattice()->posterior_probs(nullptr); // detach _post_probs from current lattice
                    p->update_probs_in_place(BBPA, re, _dilution);
                }
                else
                    p->update_probs(BBPA, re, _dilution);
                if (p->update_metadata_with_shrinking(_thres_up, _thres_lo))
                    p = p->convert_parallelism();
                node->children()[re] = new Global_partial_tree(p, BBPA, ex, re, node->curr_stage() + 1, child_prob);
                eval(node->children()[re], orig_lattice, true_state);
            }
        }
        else // test selection has been performed, children is (partially) initialized
        {
            // find BBPA and ex
            bin_enc BBPA = -1, ex = -1;
            for (int re = 0; re < (1 << node->variants()); re++)
            {
                if (node->children()[re] != nullptr)
                {
                    BBPA = dynamic_cast<Global_partial_tree *>(node->children()[re])->_BBPA;
                    ex = node->children()[re]->ex();
                    break;
                }
            }
            for (int re = 0; re < (1 << node->variants()); re++)
            {
                double child_prob = node->branch_prob() * orig_lattice->response_prob(ex, re, true_state, _dilution);
                node->children()[re]->branch_prob(child_prob);
                Product_lattice *p = node->lattice()->clone(SHALLOW_COPY_PROB_DIST); // apply calculated branch probability first
                if (re == (1 << variants()) - 1 && node->curr_stage() != 0)          // because of early-stopping, children may be computed out-of-order
                {                                                                    // reuse _post_probs in child to save memory
                    node->lattice()->posterior_probs(nullptr);                       // detach _post_probs from current lattice
                    p->update_probs_in_place(BBPA, re, _dilution);
                }
                else
                    p->update_probs(BBPA, re, _dilution);
                if (p->update_metadata_with_shrinking(_thres_up, _thres_lo))
                    p = p->convert_parallelism();
                node->children()[re]->lattice()->posterior_probs(p->posterior_probs());
                p->posterior_probs(nullptr);
                delete p;
                eval(node->children()[re], orig_lattice, true_state);
            }
        }
    }
    else // clean in advance to save memory
    {
        if (node->lattice()->posterior_probs() != nullptr) // a pre-constructed global tree might delete post_probs in classified node
        {
            node->destroy_posterior_probs();
        }
    }
}

void Global_partial_tree::lazy_eval(Tree *node, Product_lattice *orig_lattice, bin_enc true_state)
{
    backtrack[node->curr_stage()] = dynamic_cast<Global_partial_tree *>(node);
    if (!node->lattice()->is_classified() && node->curr_stage() < _search_depth && node->branch_prob() > _thres_branch)
    {
        if (node->children() == nullptr) // no child is explored, allocate children and perform computation
        {
            if (node->lattice()->posterior_probs() == nullptr)
            {
                int start_stage = 0;
                for (int i = node->curr_stage(); i >= 0; i--)
                {
                    if (backtrack[i]->lattice()->posterior_probs() != nullptr)
                    {
                        start_stage = i;
                        break;
                    }
                }
                for (int i = start_stage; i < node->curr_stage(); i++) // recover posterior probs
                {
                    Product_lattice *p = backtrack[i]->lattice()->clone(SHALLOW_COPY_PROB_DIST);
                    // if (backtrack[i + 1]->ex_res() == (1 << variants()) - 1 && i != 0) // this seems evaluate quicker but uses more space
                    // {
                    //     backtrack[i]->lattice()->posterior_probs(nullptr);
                    //     p->update_probs_in_place(backtrack[i + 1]->_BBPA, backtrack[i + 1]->_res, _dilution);
                    // }
                    if (i != 0)
                    {
                        backtrack[i]->lattice()->posterior_probs(nullptr);
                        p->update_probs_in_place(backtrack[i + 1]->BBPA(), backtrack[i + 1]->ex_res(), _dilution);
                    }
                    else
                    {
                        p->update_probs(backtrack[i + 1]->BBPA(), backtrack[i + 1]->ex_res(), _dilution);
                    }
                    if (p->update_metadata_with_shrinking(_thres_up, _thres_lo))
                        p = p->convert_parallelism();
                    backtrack[i + 1]->lattice()->posterior_probs(p->posterior_probs());
                    p->posterior_probs(nullptr);
                    delete p;
                }
            }

            node->children(1 << node->variants());
            bin_enc BBPA = node->lattice()->BBPA(1.0 / (1 << node->variants()));
            bin_enc ex = node->true_ex(BBPA);

            for (int re = 0; re < (1 << node->variants()); re++)
            {
                double child_prob = node->branch_prob() * orig_lattice->response_prob(ex, re, true_state, _dilution);
                Product_lattice *p = node->lattice()->clone(SHALLOW_COPY_PROB_DIST);
                if (re == (1 << variants()) - 1 && node->curr_stage() != 0) // reuse _post_probs in child to save memory
                {
                    node->lattice()->posterior_probs(nullptr); // detach _post_probs from current lattice
                    p->update_probs_in_place(BBPA, re, _dilution);
                }
                else
                    p->update_probs(BBPA, re, _dilution);
                if (p->update_metadata_with_shrinking(_thres_up, _thres_lo))
                    p = p->convert_parallelism();
                node->children()[re] = new Global_partial_tree(p, BBPA, ex, re, node->curr_stage() + 1, child_prob);
                lazy_eval(node->children()[re], orig_lattice, true_state);
            }
        }
        else // test selection has been performed, children is (partially) initialized
        {
            // find BBPA and ex
            bin_enc ex = -1;
            for (int re = 0; re < (1 << node->variants()); re++)
            {
                if (node->children()[re] != nullptr)
                {
                    ex = node->children()[re]->ex();
                    break;
                }
            }
            for (int re = 0; re < (1 << node->variants()); re++)
            {
                double child_prob = node->branch_prob() * orig_lattice->response_prob(ex, re, true_state, _dilution);
                node->children()[re]->branch_prob(child_prob);
                lazy_eval(node->children()[re], orig_lattice, true_state);
            }
        }
    }
    else // clean in advance to save memory
    {
        if (node->lattice()->posterior_probs() != nullptr) // a pre-constructed global tree might delete post_probs in classified node
        {
            node->destroy_posterior_probs();
        }
    }
}