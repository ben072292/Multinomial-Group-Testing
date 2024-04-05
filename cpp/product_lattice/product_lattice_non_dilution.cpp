#include "product_lattice.hpp"

const static double negative_response = 0.99;

double Product_lattice_non_dilution::response_prob(bin_enc experiment, bin_enc response, bin_enc true_state, double **__restrict__ dilution) const
{
	double ret = 1.0;
	for (int variant = 0; variant < _variants; variant++)
		ret *= ((experiment & (true_state >> (variant * _curr_subjs))) == experiment) ? ((response & (1 << variant)) == 0 ? (1.0 - negative_response) : negative_response) : ((response & (1 << variant)) == 0 ? negative_response : (1.0 - negative_response));
	return ret;
}