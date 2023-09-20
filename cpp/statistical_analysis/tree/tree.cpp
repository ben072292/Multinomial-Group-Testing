#include "tree.hpp"

int Tree::_search_depth = 0;
double Tree::_thres_up = 0.01;
double Tree::_thres_lo = 0.01;
double Tree::_thres_branch = 0.001;
double **Tree::_dilution;

Tree::Tree(Product_lattice *lattice, bin_enc ex, bin_enc res, int curr_stage, double prob)
{
    _lattice = lattice;
    _ex = ex;
    _res = res;
    _curr_stage = curr_stage;
    _children = nullptr;
    _branch_prob = prob;
}

Tree::Tree(const Tree &other, bool deep)
{
    _lattice = other._lattice->clone(NO_COPY_PROB_DIST);
    _ex = other._ex;
    _res = other._res;
    _branch_prob = other._branch_prob;
    _curr_stage = other._curr_stage;
    _children = nullptr;
    if (deep)
    {
        if (other._children != nullptr)
        {
            _children = new Tree *[1 << variants()]
            { nullptr };
            for (int i = 0; i < (1 << variants()); i++)
            {
                _children[i] = new Tree(*other.children()[i], deep); // TBD: this downcast is problematic, use virtual clone and create functions
            }
        }
    }
}

Tree::~Tree()
{
    // recursive dtor
    int variants = Product_lattice::variants();
    if (_lattice != nullptr)
        delete _lattice;
    if (_children != nullptr)
    {
        for (int i = 0; i < (1 << variants); i++)
        {
            if (_children[i] != nullptr)
                delete _children[i];
        }
        delete[] _children;
    }
}

// Convert halving selection to full-size experiment (because of lattice shrinking).
// This function should be called inside the tree before posterior probability update.
// Also note it is recommended ot not use this function if lattice shrinking is disabled,
// though initial evaluation shows no statistical changes.
bin_enc Tree::true_ex(bin_enc halving)
{
    bin_enc ret = 0, pos = 0;
    for (int i = 0; i < Product_lattice::orig_subjs(); i++)
    {
        if (!(_lattice->clas_subjs() & (1 << i)))
        {
            if ((halving & (1 << pos)))
                ret |= (1 << i);
            ++pos;
        }
    }
    return ret;
}

void Tree::parse(bin_enc true_state, const Product_lattice *org_lattice, double sym_coef, Tree_stat *stat) const
{
    stat->clear();
    double coef = org_lattice->prior_prob(true_state, Product_lattice::pi0()) * sym_coef;
    std::vector<const Tree *> *leaves = new std::vector<const Tree *>;
    find_all_stat(this, leaves);
    int size = leaves->size(), k = stat->k();

    for (int i = 0; i < size; i++)
    {
        const Tree *leaf = (*leaves)[i];
        int index = leaf->ex_count();
        if (leaf->is_classified() && leaf->is_correct_clas(true_state))
        {
            stat->correct()[index] += leaf->_branch_prob * coef;
        }
        else if (leaf->is_classified() && !leaf->is_correct_clas(true_state))
        {
            stat->incorrect()[index] += leaf->_branch_prob * coef;
            stat->fp()[index] += leaf->fp(true_state) * coef * leaf->_branch_prob;
            stat->fn()[index] += leaf->fn(true_state) * coef * leaf->_branch_prob;
        }
        else if (!leaf->is_classified())
        {
            stat->unclassified(stat->unclassified() + leaf->_branch_prob);
        }
        stat->expected_stage(stat->expected_stage() + std::ceil((double)index / (double)k) * leaf->_branch_prob);
        stat->expected_test(stat->expected_test() + index * leaf->_branch_prob);
    }
    for (int i = 0; i < size; i++)
    {
        const Tree *leaf = (*leaves)[i];
        int index = leaf->ex_count();
        stat->stage_sd(std::pow((std::ceil((double)index / (double)k) - stat->expected_stage()), 2) * leaf->_branch_prob);
        stat->test_sd(std::pow(index - stat->expected_test(), 2) * leaf->_branch_prob);
    }
    stat->unclassified(stat->unclassified() * coef);
    stat->stage_sd(std::sqrt(stat->stage_sd()) * coef);
    stat->test_sd(std::sqrt(stat->test_sd()) * coef);
    stat->expected_stage(stat->expected_stage() * coef);
    stat->expected_test(stat->expected_test() * coef);
    delete leaves;
}

bool Tree::is_correct_clas(bin_enc true_state) const
{
    return _lattice->neg_clas_atoms() == true_state;
}

// neg_clas ^ true_state filter out atoms that are wrongly classified
// then & true_state (1 means negative, 0 means positive) filters out wrong positives
// that was suppose to be negatives
double Tree::fp(bin_enc true_state) const
{
    return total_positive() == 0.0 ? 0.0 : __builtin_popcount((_lattice->neg_clas_atoms() ^ true_state) & true_state) / total_positive();
}

// neg_clas ^ true_state filter out atoms that are wrongly classified
// then & ~true_state (0 means negative, 1 means positive) filters out wrong negatives
// that was suppose to be positives
double Tree::fn(bin_enc true_state) const
{
    return total_negative() == 0.0 ? 0.0 : __builtin_popcount((_lattice->neg_clas_atoms() ^ true_state) & (~true_state)) / total_negative();
}

void Tree::find_all_leaves(const Tree *node, std::vector<const Tree *> *leaves)
{
    if (node == nullptr)
        return;
    if (node->_children == nullptr)
    {
        leaves->push_back(node);
    }
    else
    {
        for (int i = 0; i < (1 << node->variants()); i++)
        {
            if (node->children()[i] != nullptr)
            {
                find_all_leaves(node->_children[i], leaves);
            }
        }
    }
}

void Tree::find_all_stat(const Tree *node, std::vector<const Tree *> *leaves)
{
    if (node == nullptr || node->_branch_prob < _thres_branch)
        return;
    if (node->_children == nullptr)
    {
        leaves->push_back(node);
    }
    else
    {
        for (int i = 0; i < (1 << node->variants()); i++)
        {
            if (node->children()[i] != nullptr)
            {
                find_all_stat(node->_children[i], leaves);
            }
        }
    }
}

void Tree::find_clas_stat(const Tree *node, std::vector<const Tree *> *leaves)
{
    if (node == nullptr || node->_branch_prob < _thres_branch)
        return;
    if (node->is_classified())
    {
        leaves->push_back(node);
    }
    else
    {
        if (node->_children != nullptr)
        {
            for (int i = 0; i < (1 << node->variants()); i++)
            {
                if (node->children()[i] != nullptr)
                {
                    find_clas_stat(node->_children[i], leaves);
                }
            }
        }
    }
}

void Tree::find_unclas_stat(const Tree *node, std::vector<const Tree *> *leaves)
{
    if (node == nullptr || node->is_classified() || node->_branch_prob < _thres_branch)
        return;
    if (node->_children == nullptr && !node->is_classified())
    {
        leaves->push_back(node);
    }
    else if (node->_children != nullptr && !node->is_classified())
    {
        for (int i = 0; i < (1 << node->variants()); i++)
        {
            if (node->children()[i] != nullptr)
            {
                find_unclas_stat(node->_children[i], leaves);
            }
        }
    }
}

void apply_true_state_helper(const Product_lattice *__restrict__ org_lattice, Tree *__restrict__ node, bin_enc true_state, double prob, double thres_branch, double **dilution)
{
    if (node == nullptr)
        return;
    node->branch_prob(prob);
    if (node->children() != nullptr)
    {
        for (int i = 0; i < (1 << node->variants()); i++)
        {
            if (node->children()[i] != nullptr)
            {

                double child_prob = prob * org_lattice->response_prob(node->children()[i]->ex(), node->children()[i]->ex_res(), true_state, dilution);
                if (child_prob > thres_branch)
                {
                    apply_true_state_helper(org_lattice, node->children()[i], true_state, child_prob, thres_branch, dilution);
                }
                else
                {
                    node->children()[i]->branch_prob(0.0); // otherwise parsing stat tree will lead to incorrect result
                }
            }
        }
    }
}

void Tree::apply_true_state(const Product_lattice *org_lattice, bin_enc true_state)
{
    apply_true_state_helper(org_lattice, this, true_state, 1.0, _thres_branch, _dilution);
}

void shrinking_stat_helper(const Tree *node, int *stat)
{
    if (node == nullptr)
        return;
    else
    {
        stat[node->lattice()->curr_subjs()]++;
        if (node->children() != nullptr)
        {
            for (int i = 0; i < (1 << node->variants()); i++)
            {
                if (node->children()[i] != nullptr)
                {
                    shrinking_stat_helper(node->children()[i], stat);
                }
            }
        }
    }
}

std::string Tree::shrinking_stat() const
{
    std::string ret = "Subject Size, Count, Percentage\n";
    int *stat = new int[_lattice->curr_subjs() + 1]{0};
    int total = 0;
    shrinking_stat_helper(this, stat);
    for (int i = 0; i <= _lattice->curr_subjs(); i++)
    {
        total += stat[i];
    }
    for (int i = 0; i <= _lattice->curr_subjs(); i++)
    {
        ret += std::to_string(i);
        ret += ",";
        ret += std::to_string(stat[i]);
        ret += ",";
        ret += std::to_string((double)stat[i] / total * 100);
        ret += "%\n";
    }
    ret += "Total,";
    ret += std::to_string(total);
    ret += ",100%\n";
    delete[] stat;
    return ret;
}

unsigned long Tree::size_estimator()
{
    unsigned long my_size = sizeof(*this) + sizeof(*_lattice) + (_lattice->posterior_probs() == nullptr ? 0UL : sizeof(double) * (1 << (_lattice->curr_atoms())));
    if (_children == nullptr)
        return my_size;
    else
    {
        for (int i = 0; i < (1 << variants()); i++)
        {
            if (_children[i] != nullptr)
                my_size += _children[i]->size_estimator();
        }
    }
    return my_size;
}