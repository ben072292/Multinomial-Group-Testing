#include "single_tree.hpp"
#include <cmath>

Single_tree::Single_tree(Product_lattice* lattice, int ex, int res, int cur_stage, bool prev_lattice){
    lattice_ = nullptr;
    if(prev_lattice) lattice_ = lattice;
    atom_ = lattice->atom();
    variant_ = lattice->variant();
    is_clas_ = lattice->is_classified();
    clas_stat_ = lattice->classification_stat();
    ex_ = ex;
    res_ = res;
    ex_count_ = lattice->test_count();
    cur_stage_ = cur_stage;
    children_ = nullptr;
}

Single_tree::Single_tree(Product_lattice* lattice, int ex, int res, int k, int cur_stage, double thres_up, double thres_lo, int stage, double** dilution) : Single_tree(lattice, ex, res, cur_stage, false){
    if (!lattice->is_classified() && cur_stage < stage) {
        children_ = new Single_tree*[1 << lattice->variant()];
        int halving = lattice->halving(1.0 / (1 << lattice->variant()));
        for(int re = 0; re < (1 << lattice->variant()); re++){
            Product_lattice* p = lattice->clone(1);
            p->update_probs(halving, re, thres_up, thres_lo, dilution);
            children_[re] = new Single_tree(p, halving, re, k, cur_stage_+1, thres_up, thres_lo, stage, dilution);
        } 
    }
}

Single_tree::Single_tree(const Single_tree &other, bool deep){
    if(other.lattice_ != nullptr) lattice_ = other.lattice_->clone(2);
    ex_count_ = other.ex_count_;
    is_clas_ = other.is_clas_;
    clas_stat_ = other.clas_stat_;
    atom_ = other.atom_;
    variant_ = other.variant_;
    ex_ = other.ex_;
    res_ = other.res_;
    branch_prob_ = other.branch_prob_;
    cur_stage_ = other.cur_stage_;
    if(deep){
        if(other.children_ != nullptr){
            children_ = new Single_tree*[1 << variant_];
            for(int i = 0; i < (1 << variant_); i++){
                children_[i] = new Single_tree(*other.children_[i], deep);
            }
        }
    }
}

Single_tree::~Single_tree(){
    delete lattice_;
}

void Single_tree::parse(int true_state, Product_lattice* org_lattice, double* pi0, double sym_coef, Tree_stat* stat) const {
    stat->clear();
    double coef = org_lattice->prior_prob(true_state, pi0) * sym_coef;
    std::vector<const Single_tree*> *leaves = new std::vector<const Single_tree*>;
    find_all(this, leaves);
    int size = leaves->size(), k = stat->k();

    for(int i = 0; i < size; i++){
        const Single_tree* leaf = (*leaves)[i];
        if(leaf->is_clas_ && leaf->is_correct_clas(true_state)){
            stat->correct()[leaf->ex_count_] += leaf->branch_prob_ * coef;
        }
        else if (leaf->is_clas_ && !leaf->is_correct_clas(true_state)){
            stat->incorrect()[leaf->ex_count_] += leaf->branch_prob_ * coef;
            stat->fp()[leaf->ex_count_] += leaf->fp(true_state) * coef * leaf->branch_prob_;
            stat->fn()[leaf->ex_count_] += leaf->fn(true_state) * coef * leaf->branch_prob_;
        }
        else if (!leaf->is_clas_){
            stat->unclassified(stat->unclassified() + leaf->branch_prob_ * coef);
        }
        stat->expected_stage(stat->expected_stage() + std::ceil((double) leaf->ex_count_ / (double) k) * leaf->branch_prob_);
        stat->expected_test(stat->expected_test() + leaf->ex_count_ * leaf->branch_prob_);
    }

    for(int i = 0; i < size; i++){
        const Single_tree* leaf = (*leaves)[i];
        stat->stage_sd(std::pow((std::ceil((double) leaf->ex_count_ / (double) k) - stat->expected_stage()), 2) * leaf->branch_prob_);
        stat->test_sd(std::pow(leaf->ex_count_ - stat->expected_test(), 2) * leaf->branch_prob_);
    }
    stat->stage_sd(std::sqrt(stat->stage_sd()) * coef);
    stat->test_sd(std::sqrt(stat->test_sd()) * coef);
    stat->expected_stage(stat->expected_stage() * coef);
    stat->expected_test(stat->expected_test() * coef);
    delete leaves;
}

int Single_tree::actual_true_state() const{
    int actual = 0;
    for(int i = 0; i < atom_ * variant_; i++){
        if(clas_stat_[i] == 1) actual += (1 << i);
    }
    return actual;
}

double Single_tree::total_positive() const{
    return atom_ * variant_ - __builtin_popcount(actual_true_state());
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
        for(int i = 0; i < (1 << node->variant_); i++){
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
            for(int i = 0; i < (1 << node->variant_); i++){
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
        for(int i = 0; i < (1 << node->variant_); i++){
            find_unclas(node->children_[i], leaves);
        }  
    }
}

Single_tree* Single_tree::apply_true_state(const Product_lattice* org_lattice, int true_state, double thres_branch) const{
    double** dilution = org_lattice->generate_dilution(0.99, 0.005);
    Single_tree* new_tree = new Single_tree(*this, false);
    apply_true_state_helper(org_lattice, new_tree, this, true_state, 1.0, thres_branch, dilution);
    return new_tree;
}

void Single_tree::apply_true_state_helper(const Product_lattice* org_lattice, Single_tree* ret, const Single_tree* node, int true_state, double prob, double thres_branch, double** dilution){
    if(node == nullptr) return;
    ret->branch_prob_ = prob;
    if(node->children_ != nullptr){
        for(int i = 0; i < (1 << node->variant_); i++){
            if(ret->children_ == nullptr) ret->children_ = new Single_tree*[1 << node->variant_];
            double child_prob = ret->branch_prob_ * org_lattice->response_prob(node->children_[i]->ex_, node->children_[i]->res_, true_state, dilution);
            ret->children_[i] = new Single_tree(*(node->children_[i]), false);
            if(child_prob > thres_branch){
                apply_true_state_helper(org_lattice, ret->children_[i], node->children_[i], true_state, child_prob, thres_branch, dilution);
            }
            else{
                ret->children_[i] = nullptr;
            }
        }
    }
}

// int main(){
//     double pi0[6] = {0.02, 0.02, 0.02, 0.02, 0.02, 0.02};
//     Product_lattice_dilution* p = new Product_lattice_dilution(3, 2, pi0);
// 	std::cout << p->halving(0.25) << std::endl;
//     Single_tree* tree = new Single_tree(p, 0, 0, 1, true);
//     Single_tree* copy = new Single_tree(*tree, true);
//     std::vector<const Single_tree*>* v = new std::vector<const Single_tree*>;
//     std::cout << "size " << v->size() << std::endl;
//     tree->find_all(v);
//     std::cout << "size " << v->size() << std::endl;
//     std::cout << "tree address " << v->front() << std::endl;
//     std::cout << tree << std::endl;
//     std::cout << copy << std::endl;
//     std::cout << copy->lattice() << std::endl;
//     delete p;
//     delete copy;
// }



