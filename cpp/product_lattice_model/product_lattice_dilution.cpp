#include "product_lattice_dilution.hpp"
#include <iostream>

double* Product_lattice_dilution::calc_probs(int experiment, int response){
    double** dilutionMatrix = generate_dilution(0.99, 0.005);
	double* ret = new double[total_state()];
	// borrowed from find halving state function
	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	int stateIter;
	double denominator = 0.0;
	int complement = 0;
	for (stateIter = 1; stateIter < total_state(); stateIter++) {
        ret[stateIter] = post_probs_[stateIter];
		for (int variant = 0; variant < variant_; variant++) {
			complement = 0;
			for (int l = 0; l < atom_; l++) {
				if ((experiment & (1 << l)) != 0 && (stateIter & 1 << (l * variant_ + variant)) == 0) {
					complement++;
				}
			}
            if((response & (1 << variant)) == 1)
			    ret[stateIter] *= dilutionMatrix[__builtin_popcount(stateIter)-1][complement];
            else
                ret[stateIter] *= (1 - dilutionMatrix[__builtin_popcount(stateIter)-1][complement]);
		}
		denominator += ret[stateIter];
	}
	for (int i = 0; i < total_state(); i++) {
		ret[i] /= denominator;
	}
	return ret;
}

double Product_lattice_dilution::response_prob(int experiment, int response, int true_state, double** dilution) const{
    double ret = 1.0;
	int trueStatePerVariant = 0;
	int experimentLength = __builtin_popcount(experiment);
    for (int variant = 0; variant < variant_; variant++) {
		trueStatePerVariant = 0;
        for (int l = 0; l < atom_; l++)
			trueStatePerVariant += (true_state & (1 << (l * variant_ + variant))) != 0 
			? (1 << l) 
			: 0;
        ret *= (response & (1 << variant)) != 0
			? dilution[experimentLength-1][experimentLength-__builtin_popcount(experiment & trueStatePerVariant)] 
			: 1.0 - dilution[experimentLength-1][experimentLength-__builtin_popcount(experiment & trueStatePerVariant)];
       
    }
    return ret;
}