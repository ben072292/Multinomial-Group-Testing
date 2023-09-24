#pragma once
#include "product_lattice.hpp"

class Product_lattice_non_dilution : public virtual Product_lattice
{
public:
    Product_lattice_non_dilution(){}; // default constructor

    Product_lattice_non_dilution(int n_atom, int n_variant, double *pi0) : Product_lattice(n_atom, n_variant, pi0) {}

    Product_lattice_non_dilution(Product_lattice const &other, copy_op_t op) : Product_lattice(other, op) {}

    Product_lattice *create(int n_atom, int n_variant, double *pi0) const override { return new Product_lattice_non_dilution(n_atom, n_variant, pi0); }

    Product_lattice *clone(copy_op_t op) const override{ return new Product_lattice_non_dilution(*this, op); }

    double response_prob(bin_enc experiment, bin_enc response, bin_enc true_state, double **dilution) const override;

    std::string type() const override { return "DataParallelism-NoDilution"; }
};