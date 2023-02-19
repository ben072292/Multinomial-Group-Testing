#include "product_lattice_mp_dilution.hpp"

Product_lattice* Product_lattice_mp_dilution::clone(int assert) const{
	if(_parallelism == DATA_PARALLELISM) return new Product_lattice_dilution(*this, assert);
	else if(_parallelism == MODEL_PARALLELISM) return new Product_lattice_mp_dilution(*this, assert);
	else exit(1);
}