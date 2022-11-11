#include "single_tree.hpp"
#include <cmath>

Single_tree::Single_tree(Product_lattice* lattice, int ex, int res, int cur_stage){
    lattice_ = lattice;
    is_clas_ = lattice->is_classified();
    ex_ = ex;
    res_ = res;
    cur_stage_ = cur_stage;
    children_ = nullptr;
    branch_prob_ = 0.0; // must be 0.0 so that stat tree can generate correct info
}

Single_tree::Single_tree(Product_lattice* lattice, int ex, int res, int k, int cur_stage, double thres_up, double thres_lo, int stage, double** dilution) : Single_tree(lattice, ex, res, cur_stage){
    if (!lattice->is_classified() && cur_stage < stage) {
        children_ = new Single_tree*[1 << lattice->variant()];
        int halving = lattice->halving(1.0 / (1 << lattice->variant()));
        for(int re = 0; re < (1 << lattice->variant()); re++){
            if(re != (1 << lattice->variant())-1){
                Product_lattice* p = lattice->clone(1);
                p->update_probs(halving, re, thres_up, thres_lo, dilution);
                children_[re] = new Single_tree(p, halving, re, k, cur_stage_+1, thres_up, thres_lo, stage, dilution);
            }
            else{ // reuse post_prob_ array in child to save memory
                Product_lattice* p = lattice->clone(1);
                lattice_->posterior_probs(nullptr); // detach post_prob_ from current lattice
                p->update_probs_in_place(halving, re, thres_up, thres_lo, dilution);
                children_[re] = new Single_tree(p, halving, re, k, cur_stage_+1, thres_up, thres_lo, stage, dilution);
            }
        } 
    }
    else{ // clean in advance to save memory
        delete[] lattice->posterior_probs();
        lattice->posterior_probs(nullptr);
    }
}

Single_tree::Single_tree(const Single_tree &other, bool deep){
    lattice_ = other.lattice_->clone(2);
    is_clas_ = other.is_clas_;
    ex_ = other.ex_;
    res_ = other.res_;
    branch_prob_ = other.branch_prob_;
    cur_stage_ = other.cur_stage_;
    children_ = nullptr;
    if(deep){
        if(other.children_ != nullptr){
            children_ = new Single_tree*[1 << lattice_->variant()];
            for(int i = 0; i < (1 << lattice_->variant()); i++){
                children_[i] = new Single_tree(*other.children_[i], deep);
            }
        }
    }
}

Single_tree::~Single_tree(){
    // recursive dtor
    if(lattice_ != nullptr) delete lattice_;
    if(children_ != nullptr){
        for(int i = 0; i < 4; i++){
            if(children_[i] != nullptr){
                delete children_[i];
            }
        }
        delete[] children_;
    }
}

void Single_tree::parse(int true_state, const Product_lattice* org_lattice, double* pi0, double sym_coef, Tree_stat* stat) const {
    stat->clear();
    double coef = org_lattice->prior_prob(true_state, pi0) * sym_coef;
    std::vector<const Single_tree*> *leaves = new std::vector<const Single_tree*>;
    find_all(this, leaves);
    int size = leaves->size(), k = stat->k();

    for(int i = 0; i < size; i++){
        const Single_tree* leaf = (*leaves)[i];
        int index = leaf->ex_count();
        if(leaf->is_clas_ && leaf->is_correct_clas(true_state)){
            stat->correct()[index] += leaf->branch_prob_ * coef;
        }
        else if (leaf->is_clas_ && !leaf->is_correct_clas(true_state)){
            stat->incorrect()[index] += leaf->branch_prob_ * coef;
            stat->fp()[index] += leaf->fp(true_state) * coef * leaf->branch_prob_;
            stat->fn()[index] += leaf->fn(true_state) * coef * leaf->branch_prob_;
        }
        else if (!leaf->is_clas_){
            stat->unclassified(stat->unclassified() + leaf->branch_prob_ * coef);
        }
        stat->expected_stage(stat->expected_stage() + std::ceil((double) index / (double) k) * leaf->branch_prob_);
        stat->expected_test(stat->expected_test() + index * leaf->branch_prob_);
    }
    for(int i = 0; i < size; i++){
        const Single_tree* leaf = (*leaves)[i];
        int index = leaf->ex_count();
        stat->stage_sd(std::pow((std::ceil((double) index / (double) k) - stat->expected_stage()), 2) * leaf->branch_prob_);
        stat->test_sd(std::pow(index - stat->expected_test(), 2) * leaf->branch_prob_);
    }
    stat->stage_sd(std::sqrt(stat->stage_sd()) * coef);
    stat->test_sd(std::sqrt(stat->test_sd()) * coef);
    stat->expected_stage(stat->expected_stage() * coef);
    stat->expected_test(stat->expected_test() * coef);
    delete leaves;
}

int Single_tree::actual_true_state() const{
    int actual = 0;
    int index = atom() * variant();
    for(int i = 0; i < index; i++){
        if(classification_stat()[i] == 1) actual += (1 << i);
    }
    return actual;
}

double Single_tree::total_positive() const{
    return atom() * variant() - __builtin_popcount(actual_true_state());
}

double Single_tree::total_negative() const{
    return __builtin_popcount(actual_true_state());
}

bool Single_tree::is_correct_clas(int true_state) const{
    return actual_true_state() == true_state;
}

/**
 *  neg_clas ^ true_state filter out atoms that are wrongly classified
 *  then & true_state (1 means negative, 0 means positive) filters out wrong positives
 *  that was suppose to be negatives
*/
double Single_tree::fp(int true_state) const{
    return total_positive() == 0.0 ? 0.0 : __builtin_popcount((actual_true_state() ^ true_state) & true_state) / total_positive();
}

/**
 *  neg_clas ^ true_state filter out atoms that are wrongly classified
 *  then & ~true_state (0 means negative, 1 means positive) filters out wrong negatives
 *  that was suppose to be positives
*/
double Single_tree::fn(int true_state) const{
    return total_negative() == 0.0 ? 0.0 : __builtin_popcount((actual_true_state() ^ true_state) & (~true_state)) / total_negative();
}

void Single_tree::find_all(const Single_tree* node, std::vector<const Single_tree*>* leaves){
    if(node == nullptr) return;
    if(node->children_ == nullptr){
        leaves->push_back(node);
    }
    else{
        for(int i = 0; i < (1 << node->variant()); i++){
            find_all(node->children_[i], leaves);
        }
    }
}

void Single_tree::find_clas(const Single_tree* node, std::vector<const Single_tree*>* leaves){
    if(node == nullptr) return;
    if(node->is_clas_){
        leaves->push_back(node);
    }
    else{
        if(node->children_ != nullptr){
            for(int i = 0; i < (1 << node->variant()); i++){
                find_clas(node->children_[i], leaves);
        }
        }
    }
}

void Single_tree::find_unclas(const Single_tree* node, std::vector<const Single_tree*>* leaves){
    if(node == nullptr || node->is_clas_) return;
    if(node->children_ == nullptr && !node->is_clas_){
        leaves->push_back(node);
    }
    else if(node->children_ != nullptr && !node->is_clas_){
        for(int i = 0; i < (1 << node->variant()); i++){
            find_unclas(node->children_[i], leaves);
        }  
    }
}

Single_tree* Single_tree::apply_true_state(const Product_lattice* org_lattice, int true_state, double thres_branch, double** dilution) const{
    Single_tree* new_tree = new Single_tree(*this, false);
    apply_true_state_helper(org_lattice, new_tree, this, true_state, 1.0, thres_branch, dilution);
    return new_tree;
}

void Single_tree::apply_true_state_helper(const Product_lattice* org_lattice, Single_tree* ret, const Single_tree* node, int true_state, double prob, double thres_branch, double** dilution){
    if(node == nullptr) return;
    ret->branch_prob_ = prob;
    if(node->children_ != nullptr){
        for(int i = 0; i < (1 << node->variant()); i++){
            if(ret->children_ == nullptr) ret->children_ = new Single_tree*[1 << node->variant()];
            double child_prob = ret->branch_prob_ * org_lattice->response_prob(node->children_[i]->ex_, node->children_[i]->res_, true_state, dilution);
            ret->children_[i] = new Single_tree(*(node->children_[i]), false);
            if(child_prob > thres_branch){
                apply_true_state_helper(org_lattice, ret->children_[i], node->children_[i], true_state, child_prob, thres_branch, dilution);
            }
        }
    }
}

void Single_tree::prep_destroy_stat_tree(){
    lattice_->classification_stat(nullptr);
    if(children_ != nullptr){
        for(int i = 0; i < (1 << lattice_->variant()); i++){
            if(children_[i] != nullptr){
                children_[i]->prep_destroy_stat_tree();
            }
        }
    }
}



