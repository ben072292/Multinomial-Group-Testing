#include "product_lattice_mp_dilution.hpp"

double Product_lattice_mp_dilution::response_prob(bin_enc experiment, bin_enc response, bin_enc true_state, double** __restrict__ dilution) const{
    double ret = 1.0;
	int experimentLength = __builtin_popcount(experiment);
    for (int variant = 0; variant < _variants; variant++) {
        ret *= (response & (1 << variant)) != 0
			? dilution[experimentLength-1][experimentLength-__builtin_popcount(experiment & (true_state >> (variant * _curr_subjs)))] 
			: 1.0 - dilution[experimentLength-1][experimentLength-__builtin_popcount(experiment & (true_state >> (variant * _curr_subjs)))];
       
    }
    return ret;
}

Product_lattice* Product_lattice_mp_dilution::clone(int assert) const{
	if(_parallelism == DATA_PARALLELISM) return new Product_lattice_dilution(*this, assert);
	else if(_parallelism == MODEL_PARALLELISM) return new Product_lattice_mp_dilution(*this, assert);
	else exit(1);
}