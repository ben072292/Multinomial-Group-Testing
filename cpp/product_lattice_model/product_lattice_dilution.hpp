#pragma once
#include "product_lattice.hpp"

class Product_lattice_dilution : public Product_lattice{
    public:
    Product_lattice_dilution(int n_atom, int n_variant, double* pi0):Product_lattice(n_atom, n_variant, pi0){}

	Product_lattice_dilution(Product_lattice_dilution const &other, int assert) : Product_lattice(other, assert){}

    Product_lattice_dilution *create(int n_atom, int n_variant, double *pi0) const {return new Product_lattice_dilution(n_atom, n_variant, pi0);}

    Product_lattice_dilution *clone(int assert) const {return new Product_lattice_dilution(*this, assert);}

    double response_prob(int experiment, int response, int true_state, double** dilution) const;

    virtual void type(){std::cout << "Lattice Model Dilution" << std::endl;}
};