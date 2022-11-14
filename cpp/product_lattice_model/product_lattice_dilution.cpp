#include "product_lattice_dilution.hpp"

double Product_lattice_dilution::response_prob(int experiment, int response, int true_state, double** __restrict__ dilution) const{
    double ret = 1.0;
	int experimentLength = __builtin_popcount(experiment);
    for (int variant = 0; variant < variant_; variant++) {
        ret *= (response & (1 << variant)) != 0
			? dilution[experimentLength-1][experimentLength-__builtin_popcount(experiment & (true_state >> (variant * atom_)))] 
			: 1.0 - dilution[experimentLength-1][experimentLength-__builtin_popcount(experiment & (true_state >> (variant * atom_)))];
       
    }
    return ret;
}