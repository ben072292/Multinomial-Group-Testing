#include "product_lattice_non_dilution.hpp"
#include <iostream>


double* Product_lattice_non_dilution::calc_probs(int experiment, int response, double** dilution){
    double* ret = new double[total_st_];
	int partition = 0;
	// borrowed from find halving state function
	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	int state_iter;
	double denominator = 0.0;
	bool is_complement = false;
	for (state_iter = 0; state_iter < total_st_; state_iter++) {
		ret[state_iter] = post_probs_[state_iter];
		partition = 0;
		for (int variant = 0; variant < variant_; variant++) {
			is_complement = false;
			for (int l = 0; l < atom_; l++) {
				if ((experiment & (1 << l)) != 0 && (state_iter & 1 << (l * variant_ + variant)) == 0) {
					is_complement = true;
					break;
				}
			}
			partition |= (is_complement ? 0 : (1 << variant));
		}
		for(int i = 0; i < __builtin_popcount(partition ^ response); i++){
			ret[state_iter] *= 0.005;
		}
		for(int i = 0; i < variant_ - __builtin_popcount(partition ^ response); i++){
			ret[state_iter] *= 0.985;
		}
		denominator += ret[state_iter];
	}
	for (int i = 0; i < total_st_; i++) {
		ret[i] /= denominator;
	}
	return ret;
}

double Product_lattice_non_dilution::response_prob(int experiment, int response, int true_state, double** dilution) const{
    double ret = 1.0;
	int true_state_response = 0;
	int n = __builtin_popcount(experiment);
	for (int i = 0; i < variant_; i++) {
		int temp = 0;
		for (int j = 0; j < atom_; j++) {
			if (!(experiment & (1 << j)))
				continue;
			if ((true_state & (1 << (j * variant_ + i))))
				temp++;
		}
		if (temp == n)
			true_state_response += (1 << i);
	}
	int error = __builtin_popcount(true_state_response ^ response);
	int correct = variant_ - error;
	
	for(int i = 0; i < error; i++) ret *= 0.005;
	for(int i = 0; i < correct; i++) ret *= 0.985;
	return ret;
}

// int main(){
//     double pi0[6] = {0.02, 0.02, 0.02, 0.02, 0.02, 0.02};
//     Product_lattice_non_dilution* p = new Product_lattice_non_dilution(3, 2, pi0);

	
// }