#include "global_tree.hpp"

int Global_tree::rank = -1;
int Global_tree::world_size = 0;

Global_tree::Global_tree(Product_lattice* lattice, bin_enc ex, bin_enc res, int curr_stage){
    _lattice = lattice;
    _is_clas = lattice->is_classified();
    _ex = ex;
    _res = res;
    _curr_stage = curr_stage;
    _children = nullptr;
    _branch_prob = 0.0; // must be 0.0 so that stat tree can generate correct info
}

Global_tree::Global_tree(Product_lattice* lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double** dilution) : Global_tree(lattice, ex, res, curr_stage){
    if (!lattice->is_classified() && curr_stage < stage) {
        _children = new Global_tree*[1 << lattice->variants()];
        // int halving = lattice->halving(1.0 / (1 << lattice->variant()));
        bin_enc halving = lattice->halving_omp(1.0 / (1 << lattice->variants())); // openmp
        bin_enc ex = true_ex(halving); // full-sized experiment should be generated before posterior probability distribution is updated, because unupdated clas_subj_ should be used to calculate the correct value
        for(bin_enc re = 0; re < (1 << lattice->variants()); re++){
            if(re != (1 << lattice->variants())-1){
                Product_lattice* p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                p->update_probs(halving, re, thres_up, thres_lo, dilution);
                _children[re] = new Global_tree(p, ex, re, k, _curr_stage+1, thres_up, thres_lo, stage, dilution);
            }
            else{ // reuse post_prob_ array in child to save memory
                Product_lattice* p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                _lattice->posterior_probs(nullptr); // detach post_prob_ from current lattice
                p->update_probs_in_place(halving, re, thres_up, thres_lo, dilution);
                _children[re] = new Global_tree(p, ex, re, k, _curr_stage+1, thres_up, thres_lo, stage, dilution);
            }
        }
    }
    else{ // clean in advance to save memory
        delete[] lattice->posterior_probs();
        lattice->posterior_probs(nullptr);
    }
}

Global_tree::Global_tree(Product_lattice* lattice, bin_enc ex, bin_enc res, int k, int curr_stage, double thres_up, double thres_lo, int stage, double** dilution, std::chrono::nanoseconds halving_times[]) : Global_tree(lattice, ex, res, curr_stage){
    auto start = std::chrono::high_resolution_clock::now(), end = start;
    if (!lattice->is_classified() && curr_stage < stage) {
        _children = new Global_tree*[1 << lattice->variants()];
        // int halving = lattice->halving(1.0 / (1 << lattice->variant()));
        bin_enc halving = lattice->halving_omp(1.0 / (1 << lattice->variants())); // openmp
        end = std::chrono::high_resolution_clock::now();
        halving_times[_lattice->curr_subjs()] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        bin_enc ex = true_ex(halving); // full-sized experiment should be generated before posterior probability distribution is updated, because unupdated clas_subj_ should be used to calculate the correct value
        for(bin_enc re = 0; re < (1 << lattice->variants()); re++){
            if(re != (1 << lattice->variants())-1){
                Product_lattice* p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                p->update_probs(halving, re, thres_up, thres_lo, dilution);
                _children[re] = new Global_tree(p, ex, re, k, _curr_stage+1, thres_up, thres_lo, stage, dilution, halving_times);
            }
            else{ // reuse post_prob_ array in child to save memory
                Product_lattice* p = lattice->clone(SHALLOW_COPY_PROB_DIST);
                _lattice->posterior_probs(nullptr); // detach post_prob_ from current lattice
                p->update_probs_in_place(halving, re, thres_up, thres_lo, dilution);
                _children[re] = new Global_tree(p, ex, re, k, _curr_stage+1, thres_up, thres_lo, stage, dilution, halving_times);
            }
        }
    }
    else{ // clean in advance to save memory
        delete[] lattice->posterior_probs();
        lattice->posterior_probs(nullptr);
    }
}

Global_tree::Global_tree(const Global_tree &other, bool deep){
    _lattice = other._lattice->clone(NO_COPY_PROB_DIST);
    _is_clas = other._is_clas;
    _ex = other._ex;
    _res = other._res;
    _branch_prob = other._branch_prob;
    _curr_stage = other._curr_stage;
    _children = nullptr;
    if(deep){
        if(other._children != nullptr){
            _children = new Global_tree*[1 << _lattice->variants()];
            for(int i = 0; i < (1 << _lattice->variants()); i++){
                _children[i] = new Global_tree(*other._children[i], deep);
            }
        }
    }
}

Global_tree::~Global_tree(){
    // recursive dtor
    int variants = _lattice->variants();
    if(_lattice != nullptr) delete _lattice;
    if(_children != nullptr){
        for(int i = 0; i < (1 << variants); i++){
            if(_children[i] != nullptr){
                delete _children[i];
            }
        }
        delete[] _children;
    }
}

// Convert halving selection to full-size experiment (because of lattice shrinking).
// This function should be called inside the tree before posterior probability update.
// Also note it is recommended ot not use this function if lattice shrinking is disabled,
// though initial evaluation shows no statistical changes.
bin_enc Global_tree::true_ex(bin_enc halving){
    bin_enc ret = 0, pos = 0;
    for(int i = 0; i < _lattice->orig_subjs(); i++){
        if(!(_lattice->clas_subjs() & (1 << i))){
            if((halving & (1 << pos))) ret |= (1 << i);
            ++pos;
        }
    }
    return ret;
}

void Global_tree::parse(bin_enc true_state, const Product_lattice* org_lattice, double* pi0, double thres_branch, double sym_coef, Tree_stat* stat) const {
    stat->clear();
    double coef = org_lattice->prior_prob(true_state, pi0) * sym_coef;
    std::vector<const Global_tree*> *leaves = new std::vector<const Global_tree*>;
    find_all_stat(this, leaves, thres_branch);
    int size = leaves->size(), k = stat->k();

    for(int i = 0; i < size; i++){
        const Global_tree* leaf = (*leaves)[i];
        int index = leaf->ex_count();
        if(leaf->_is_clas && leaf->is_correct_clas(true_state)){
            stat->correct()[index] += leaf->_branch_prob * coef;
        }
        else if (leaf->_is_clas && !leaf->is_correct_clas(true_state)){
            stat->incorrect()[index] += leaf->_branch_prob * coef;
            stat->fp()[index] += leaf->fp(true_state) * coef * leaf->_branch_prob;
            stat->fn()[index] += leaf->fn(true_state) * coef * leaf->_branch_prob;
        }
        else if (!leaf->_is_clas){
            stat->unclassified(stat->unclassified() + leaf->_branch_prob * coef);
        }
        stat->expected_stage(stat->expected_stage() + std::ceil((double) index / (double) k) * leaf->_branch_prob);
        stat->expected_test(stat->expected_test() + index * leaf->_branch_prob);
    }
    for(int i = 0; i < size; i++){
        const Global_tree* leaf = (*leaves)[i];
        int index = leaf->ex_count();
        stat->stage_sd(std::pow((std::ceil((double) index / (double) k) - stat->expected_stage()), 2) * leaf->_branch_prob);
        stat->test_sd(std::pow(index - stat->expected_test(), 2) * leaf->_branch_prob);
    }
    stat->stage_sd(std::sqrt(stat->stage_sd()) * coef);
    stat->test_sd(std::sqrt(stat->test_sd()) * coef);
    stat->expected_stage(stat->expected_stage() * coef);
    stat->expected_test(stat->expected_test() * coef);
    delete leaves;
}

bool Global_tree::is_correct_clas(bin_enc true_state) const{
    return _lattice->neg_clas_atoms() == true_state;
}

// neg_clas ^ true_state filter out atoms that are wrongly classified
// then & true_state (1 means negative, 0 means positive) filters out wrong positives
// that was suppose to be negatives
double Global_tree::fp(bin_enc true_state) const{
    return total_positive() == 0.0 ? 0.0 : __builtin_popcount((_lattice->neg_clas_atoms() ^ true_state) & true_state) / total_positive();
}

// neg_clas ^ true_state filter out atoms that are wrongly classified
// then & ~true_state (0 means negative, 1 means positive) filters out wrong negatives
// that was suppose to be positives
double Global_tree::fn(bin_enc true_state) const{
    return total_negative() == 0.0 ? 0.0 : __builtin_popcount((_lattice->neg_clas_atoms() ^ true_state) & (~true_state)) / total_negative();
}

void Global_tree::find_all_leaves(const Global_tree* node, std::vector<const Global_tree*> *leaves){
    if(node == nullptr) return;
    if(node->_children == nullptr){
        leaves->push_back(node);
    }
    else{
        for(int i = 0; i < (1 << node->variants()); i++){
            find_all_leaves(node->_children[i], leaves);
        }
    }
}

void Global_tree::find_all_stat(const Global_tree* node, std::vector<const Global_tree*>* leaves, double thres_branch){
    if(node == nullptr || node->_branch_prob < thres_branch) return;
    if(node->_children == nullptr){
        leaves->push_back(node);
    }
    else{
        for(int i = 0; i < (1 << node->variants()); i++){
            find_all_stat(node->_children[i], leaves, thres_branch);
        }
    }
}

void Global_tree::find_clas_stat(const Global_tree* node, std::vector<const Global_tree*>* leaves, double thres_branch){
    if(node == nullptr || node->_branch_prob < thres_branch) return;
    if(node->_is_clas){
        leaves->push_back(node);
    }
    else{
        if(node->_children != nullptr){
            for(int i = 0; i < (1 << node->variants()); i++){
                find_clas_stat(node->_children[i], leaves, thres_branch);
        }
        }
    }
}

void Global_tree::find_unclas_stat(const Global_tree* node, std::vector<const Global_tree*>* leaves, double thres_branch){
    if(node == nullptr || node->_is_clas || node->_branch_prob < thres_branch) return;
    if(node->_children == nullptr && !node->_is_clas){
        leaves->push_back(node);
    }
    else if(node->_children != nullptr && !node->_is_clas){
        for(int i = 0; i < (1 << node->variants()); i++){
            find_unclas_stat(node->_children[i], leaves, thres_branch);
        }  
    }
}

void Global_tree::apply_true_state(const Product_lattice* org_lattice, bin_enc true_state, double thres_branch, double** __restrict__ dilution){
    apply_true_state_helper(org_lattice, this, true_state, 1.0, thres_branch, dilution);
}

void Global_tree::apply_true_state_helper(const Product_lattice* __restrict__ org_lattice, Global_tree* __restrict__ node, bin_enc true_state, double prob, double thres_branch, double** __restrict__ dilution){
    if(node == nullptr) return;
    node->_branch_prob = prob;
    if(node->_children != nullptr){
        for(int i = 0; i < (1 << node->variants()); i++){
            double child_prob = prob * org_lattice->response_prob(node->_children[i]->_ex, node->_children[i]->_res, true_state, dilution);
            if(child_prob > thres_branch){
                apply_true_state_helper(org_lattice, node->_children[i], true_state, child_prob, thres_branch, dilution);
            }
            else{
                node->_children[i]->_branch_prob = 0.0; // otherwise parsing stat tree will lead to incorrect result
            }
        }
    }
}

void shrinking_stat_helper(const Global_tree* node, int* stat){
    if(node == nullptr) return;
    if(node->children() == nullptr){
        stat[node->lattice()->curr_subjs()]++;
    }
    else{
        for(int i = 0; i < (1 << node->variants()); i++){
            shrinking_stat_helper(node->children()[i], stat);
        }
    }
}

std::string Global_tree::shrinking_stat() const {
    std::string ret = "Subject Size, Count, Percentage\n";
    int* stat = new int[_lattice->curr_subjs()+1]{0};
    int total = 0;
    shrinking_stat_helper(this, stat);
    for(int i = 0; i <= _lattice->curr_subjs(); i++){
        total += stat[i];
    }
    for(int i = 0; i <= _lattice->curr_subjs(); i++){
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

void Global_tree::MPI_Global_tree_Initialize(){
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}