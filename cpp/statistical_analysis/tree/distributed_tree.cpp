#include "distributed_tree.hpp"

int Distributed_tree::rank = -1;
int Distributed_tree::world_size = 0;
Distributed_tree **Distributed_tree::backtrack;
MPI_Datatype Distributed_tree::tree_stat_type;
MPI_Op Distributed_tree::tree_stat_op;

Distributed_tree::Distributed_tree(Product_lattice *lattice, bin_enc halving, bin_enc ex, bin_enc res, int curr_stage, double prob) : Tree(lattice, ex, res, curr_stage, prob)
{
    _halving = halving;
}

Distributed_tree::Distributed_tree(Product_lattice *lattice, bin_enc halving, bin_enc ex, bin_enc res, int k, int curr_stage, int expansion_depth) : Distributed_tree(lattice, halving, ex, res, curr_stage, 0.0)
{
    if (!_lattice->is_classified() && curr_stage < expansion_depth)
    {
        _children = new Tree *[1 << variants()]
        { nullptr };
        bin_enc halving = _lattice->halving(1.0 / (1 << variants())); // openmp
        bin_enc ex = true_ex(halving);                                // full-sized experiment should be generated before posterior probability distribution is updated, because unupdated clas_subj_ should be used to calculate the correct value
        for (bin_enc re = 0; re < (1 << variants()); re++)
        {
            Product_lattice *p = _lattice->clone(SHALLOW_COPY_PROB_DIST);
            if (re != (1 << variants()) - 1 || curr_stage == 0)
            {
                p->update_probs(halving, re, _dilution);
            }
            else
            {                                       // reuse post_prob_ array in child to save memory
                _lattice->posterior_probs(nullptr); // detach post_prob_ from current lattice
                p->update_probs_in_place(halving, re, _dilution);
            }
            p->update_metadata_with_shrinking(_thres_up, _thres_lo);
            _children[re] = new Distributed_tree(p, halving, ex, re, k, _curr_stage + 1, expansion_depth);
        }
    }
    else
    { // clean in advance to save memory
        destroy_posterior_probs();
    }
}

Distributed_tree::Distributed_tree(const Tree &other, bool deep) : Tree(other, false)
{
    _halving = dynamic_cast<const Distributed_tree &>(other)._halving;
    if (deep)
    {
        if (other.children() != nullptr)
        {
            _children = new Tree *[1 << variants()];
            for (int i = 0; i < (1 << variants()); i++)
            {
                _children[i] = new Distributed_tree(*(other.children()[i]), deep); // TBD: this downcast is problematic, use virtual clone and create functions
            }
        }
    }
}

void Distributed_tree::eval(Tree *node, Product_lattice *orig_lattice, bin_enc true_state)
{
    if (!node->lattice()->is_classified() && node->curr_stage() < _search_depth && node->branch_prob() > _thres_branch)
    {
        if (node->children() == nullptr) // no child is explored, allocate children and perform computation
        {
            node->children(1 << node->variants());
            bin_enc halving = node->lattice()->halving_serial(1.0 / (1 << node->variants()));
            bin_enc ex = node->true_ex(halving);

            for (int re = 0; re < (1 << node->variants()); re++)
            {
                double child_prob = node->branch_prob() * orig_lattice->response_prob(ex, re, true_state, _dilution);
                // if (child_prob < _thres_branch)  // TBD not working for now (double free error)
                // {
                //     if (node->children()[re] != nullptr)
                //     {
                //         node->children()[re]->branch_prob(child_prob);
                //         if (node->children()[re]->lattice()->posterior_probs() != nullptr)
                //         {
                //             node->children()[re]->destroy_posterior_probs();
                //         }
                //         if (re == (1 << variants()) - 1 && node->curr_stage() != 0) // because of early-stopping, children may be computed out-of-order
                //         {
                //             node->destroy_posterior_probs();
                //         }
                //     }
                //     continue; // early-stopping
                // }
                Product_lattice *p = node->lattice()->clone(SHALLOW_COPY_PROB_DIST);
                if (re == (1 << variants()) - 1 && node->curr_stage() != 0) // reuse _post_probs in child to save memory
                {
                    node->lattice()->posterior_probs(nullptr); // detach _post_probs from current lattice
                    p->update_probs_in_place(halving, re, _dilution);
                }
                else
                    p->update_probs(halving, re, _dilution);
                p->update_metadata_with_shrinking(_thres_up, _thres_lo);
                node->children()[re] = new Distributed_tree(p, halving, ex, re, node->curr_stage() + 1, child_prob);
                eval(node->children()[re], orig_lattice, true_state);
            }
        }
        else // test selection has been performed, children is (partially) initialized
        {
            // find halving and ex
            bin_enc halving = -1, ex = -1;
            for (int re = 0; re < (1 << node->variants()); re++)
            {
                if (node->children()[re] != nullptr)
                {
                    halving = dynamic_cast<Distributed_tree *>(node->children()[re])->_halving;
                    ex = node->children()[re]->ex();
                    break;
                }
            }
            if (halving == -1)
            {
                log_error("Logic error");
                exit(1);
            }
            for (int re = 0; re < (1 << node->variants()); re++)
            {
                double child_prob = node->branch_prob() * orig_lattice->response_prob(ex, re, true_state, _dilution);
                // if (child_prob < _thres_branch) // early-stopping
                // {
                //     if (node->children()[re] != nullptr)
                //     {
                //         node->children()[re]->branch_prob(child_prob);
                //         if (node->children()[re]->lattice()->posterior_probs() != nullptr)
                //         {
                //             node->children()[re]->destroy_posterior_probs();
                //         }
                //         if (re == (1 << variants()) - 1 && node->curr_stage() != 0)
                //         {
                //             node->destroy_posterior_probs();
                //         }
                //     }
                //     continue;
                // }
                if (node->children()[re] != nullptr) // fill in only posterior probability and step in
                {
                    node->children()[re]->branch_prob(child_prob);
                    Product_lattice *p = node->lattice()->clone(SHALLOW_COPY_PROB_DIST); // apply calculated branch probability first
                    if (re == (1 << variants()) - 1 && node->curr_stage() != 0)          // because of early-stopping, children may be computed out-of-order
                    {                                                                    // reuse _post_probs in child to save memory
                        node->lattice()->posterior_probs(nullptr);                       // detach _post_probs from current lattice
                        p->update_probs_in_place(halving, re, _dilution);
                    }
                    else
                        p->update_probs(halving, re, _dilution);
                    p->update_metadata_with_shrinking(_thres_up, _thres_lo);
                    node->children()[re]->lattice()->posterior_probs(p->posterior_probs());
                    p->posterior_probs(nullptr);
                    delete p;
                    eval(node->children()[re], orig_lattice, true_state);
                }
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

void Distributed_tree::lazy_eval(Tree *node, Product_lattice *orig_lattice, bin_enc true_state)
{
    backtrack[node->curr_stage()] = dynamic_cast<Distributed_tree *>(node);
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
                    //     p->update_probs_in_place(backtrack[i + 1]->_halving, backtrack[i + 1]->_res, _dilution);
                    // }
                    if (i != 0)
                    {
                        backtrack[i]->lattice()->posterior_probs(nullptr);
                        p->update_probs_in_place(backtrack[i + 1]->_halving, backtrack[i + 1]->_res, _dilution);
                    }
                    else
                    {
                        p->update_probs(backtrack[i + 1]->_halving, backtrack[i + 1]->_res, _dilution);
                    }
                    p->update_metadata_with_shrinking(_thres_up, _thres_lo);
                    backtrack[i + 1]->lattice()->posterior_probs(p->posterior_probs());
                    p->posterior_probs(nullptr);
                    delete p;
                }
            }

            node->children(1 << node->variants());
            bin_enc halving = node->lattice()->halving_serial(1.0 / (1 << node->variants()));
            bin_enc ex = node->true_ex(halving);

            for (int re = 0; re < (1 << node->variants()); re++)
            {
                double child_prob = node->branch_prob() * orig_lattice->response_prob(ex, re, true_state, _dilution);
                Product_lattice *p = node->lattice()->clone(SHALLOW_COPY_PROB_DIST);
                if (re == (1 << variants()) - 1 && node->curr_stage() != 0) // reuse _post_probs in child to save memory
                {
                    node->lattice()->posterior_probs(nullptr); // detach _post_probs from current lattice
                    p->update_probs_in_place(halving, re, _dilution);
                }
                else
                    p->update_probs(halving, re, _dilution);
                if (p->update_metadata_with_shrinking(_thres_up, _thres_lo))
                    p = p->convert_parallelism();
                node->children()[re] = new Distributed_tree(p, halving, ex, re, node->curr_stage() + 1, child_prob);
                lazy_eval(node->children()[re], orig_lattice, true_state);
            }
        }
        else // test selection has been performed, children is (partially) initialized
        {
            // find halving and ex
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

void Distributed_tree::MPI_Distributed_tree_Initialize(int subjs, int k, int search_depth)
{
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    backtrack = new Distributed_tree *[search_depth + 1];

    Tree_stat::create_tree_stat_type(&tree_stat_type, _search_depth, k);
    MPI_Type_commit(&tree_stat_type);
    MPI_Op_create((MPI_User_function *)&Tree_stat::tree_stat_reduce, true, &tree_stat_op);
}

void Distributed_tree::MPI_Distributed_tree_Free()
{
    // Free datatype
    MPI_Type_free(&tree_stat_type);
    // Free tree stat reduce op
    MPI_Op_free(&tree_stat_op);

    delete[] backtrack;
}