#include "product_lattice_mp_non_dilution.hpp"

double Product_lattice_mp_non_dilution::response_prob(bin_enc experiment, bin_enc response, bin_enc true_state, double** __restrict__ dilution) const{
    double ret = 1.0;
	for (int variant = 0; variant < _variants; variant++) 
		ret *= ((experiment & (true_state >> (variant * _curr_subjs))) == experiment) ? ((response & (1 << variant)) == 0 ? 0.005 : 0.985) : ((response & (1 << variant)) == 0 ? 0.985 : 0.005);
	return ret;
}

Product_lattice* Product_lattice_mp_non_dilution::clone(int assert) const{
	if(_parallelism == DATA_PARALLELISM) return new Product_lattice_non_dilution(*this, assert);
	else if(_parallelism == MODEL_PARALLELISM) return new Product_lattice_mp_non_dilution(*this, assert);
	else exit(1);
}