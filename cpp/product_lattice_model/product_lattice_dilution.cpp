#include "product_lattice_dilution.hpp"
#include <iostream>

double* Product_lattice_dilution::calc_probs(int experiment, int response, double** dilution){
	int total_stat = 1 << (atom_ * variant_);
	double* ret = new double[total_stat];
	double denominator = 0.0;
	for (int iter = 0; iter < total_stat; iter++) {
		ret[iter] = post_probs_[iter] * response_prob(experiment, response, iter, dilution);
		denominator += ret[iter];
	}
	for (int i = 0; i < total_stat; i++) {
		ret[i] /= denominator;
	}
	return ret;
}

void Product_lattice_dilution::calc_probs_in_place(int experiment, int response, double** dilution){
	double denominator = 0.0;
	int total_stat = 1 << (atom_ * variant_);
	for (int iter = 0; iter < total_stat; iter++) {
		post_probs_[iter] *= response_prob(experiment, response, iter, dilution);
		denominator += post_probs_[iter];
	}
	for (int i = 0; i < total_stat; i++) {
		post_probs_[i] /= denominator;
	}
}

double Product_lattice_dilution::response_prob(int experiment, int response, int true_state, double** dilution) const{
    double ret = 1.0;
	int true_state_per_variant = 0;
	int experimentLength = __builtin_popcount(experiment);
    for (int variant = 0; variant < variant_; variant++) {
		true_state_per_variant = 0;
        for (int l = 0; l < atom_; l++)
			true_state_per_variant += (true_state & (1 << (l * variant_ + variant))) != 0 
			? (1 << l) 
			: 0;
        ret *= (response & (1 << variant)) != 0
			? dilution[experimentLength-1][experimentLength-__builtin_popcount(experiment & true_state_per_variant)] 
			: 1.0 - dilution[experimentLength-1][experimentLength-__builtin_popcount(experiment & true_state_per_variant)];
       
    }
    return ret;
}