#pragma once
#include "product_lattice_non_dilution.hpp"

class Product_lattice_dilution : public Product_lattice_non_dilution{
    public:
    Product_lattice_dilution(int n_atom, int n_variant, double* pi0):Product_lattice_non_dilution(n_atom, n_variant, pi0){}

	~Product_lattice_dilution(){}

    double* calc_probs(int experiment, int response);

    double response_prob(int experiment, int response, int true_state, double** dilution) const;
};