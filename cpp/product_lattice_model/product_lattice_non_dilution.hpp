#pragma once
#include "product_lattice.hpp"

class Product_lattice_non_dilution : public Product_lattice{
    public:
    // Product_lattice_non_dilution(); // default constructor
    Product_lattice_non_dilution(int n_atom, int n_variant, double* pi0) : Product_lattice(n_atom, n_variant, pi0){}

    Product_lattice_non_dilution(Product_lattice_non_dilution const &other, int assert) : Product_lattice(other, assert){}

    Product_lattice_non_dilution *create(int n_atom, int n_variant, double *pi0) const {return new Product_lattice_non_dilution(n_atom, n_variant, pi0);}

    Product_lattice_non_dilution *clone(int assert) const {return new Product_lattice_non_dilution(*this, assert);}

    double* calc_probs(int experiment, int response, double** dilution);

    double response_prob(int experiment, int response, int true_state, double** dilution) const;

    virtual void type(){std::cout << "Lattice Model Non Dilution" << std::endl;}
};