#pragma once
#include "product_lattice_mp.hpp"

class Product_lattice_mp_dilution : public Product_lattice_mp, public Product_lattice_dilution{
    public:
    Product_lattice_mp_dilution(int n_atom, int n_variant, double* pi0) : Product_lattice_mp(n_atom, n_variant, pi0){}

    // Note the copy constructor directly calls the grandparent copy constructor, i.e., Product_lattice(ohter, copy, op)
    // Quoted from https://www.geeksforgeeks.org/multiple-inheritance-in-c/
    // "In the above program, constructor of ‘Person’ is called once. One important thing to note in the above output is, 
    // the default constructor of ‘Person’ is called. When we use ‘virtual’ keyword, the default constructor of grandparent 
    // class is called by default even if the parent classes explicitly call parameterized constructor. How to call the 
    // parameterized constructor of the ‘Person’ class? The constructor has to be called in ‘TA’ class. For example, see 
    // the following program. "
	Product_lattice_mp_dilution(Product_lattice_mp const &other, int copy_op) : Product_lattice(other, copy_op){}

    Product_lattice *create(int n_atom, int n_variant, double *pi0) const {return new Product_lattice_mp_dilution(n_atom, n_variant, pi0);}

    Product_lattice *clone(int assert) const;

    std::string type() const {return "Model-Parallelism-Dilution";}
};