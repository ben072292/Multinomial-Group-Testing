#pragma once
#include "product_lattice_mp.hpp"

class Product_lattice_mp_non_dilution : public Product_lattice_mp{
    public:
    Product_lattice_mp_non_dilution(int n_atom, int n_variant, double* pi0) : Product_lattice_mp(n_atom, n_variant, pi0){}

    Product_lattice_mp_non_dilution(Product_lattice_mp const &other, int copy_op) : Product_lattice_mp(other, copy_op){}

    Product_lattice *create(int n_atom, int n_variant, double *pi0) const {return new Product_lattice_mp_non_dilution(n_atom, n_variant, pi0);}

    Product_lattice *clone(int assert) const;

    double response_prob(bin_enc experiment, bin_enc response, bin_enc true_state, double** dilution) const;

    std::string type() const {return "Model-Parallelism-Non-Dilution";}
};