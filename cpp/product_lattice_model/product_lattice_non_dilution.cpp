#include "product_lattice_non_dilution.hpp"
#include <iostream>

double Product_lattice_non_dilution::response_prob(bin_enc experiment, bin_enc response, bin_enc true_state, double** __restrict__ dilution) const{
    double ret = 1.0;
	for (int variant = 0; variant < _variants; variant++) 
		ret *= ((experiment & (true_state >> (variant * _curr_subjs))) == experiment) ? ((response & (1 << variant)) == 0 ? 0.005 : 0.985) : ((response & (1 << variant)) == 0 ? 0.985 : 0.005);
	return ret;
}