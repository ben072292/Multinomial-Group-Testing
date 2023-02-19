#pragma once
#include "product_lattice.hpp"
#include "product_lattice_non_dilution.hpp"
#include "product_lattice_dilution.hpp"
/*
 * READ BEFORE USE:
 * In this version of lattice model, state is represented using A0B0A1B1...
 * The input of prior should also follow this pattern.
 * Model parallelism implementation for lattice model
 */
class Product_lattice_mp : public Product_lattice{

	protected:
	inline bin_enc total_state_each() const {return total_state() / world_size;}
    inline int state_to_offset(bin_enc state) const {return state % total_state_each();}
	inline int state_to_rank(bin_enc state) const {return state / total_state_each();}
    inline bin_enc offset_to_state(int offset) const {return offset + total_state_each() * rank;}

	public:
	Product_lattice_mp(int n_atom, int n_variant, double* pi0);
	Product_lattice_mp(const Product_lattice_mp &other, int copy_op) : Product_lattice(other, copy_op){};
	virtual ~Product_lattice_mp();
	double posterior_prob(bin_enc state) const;
	void prior_probs(double* pi0);
	double* calc_probs(bin_enc experiment, bin_enc response, double** dilution);
	void calc_probs_in_place(bin_enc experiment, bin_enc response, double** dilution);
	void update_metadata(double thres_up, double thres_lo);
	void update_metadata_with_shrinking(double thres_up, double thres_lo);
	void shrinking(int orig_subjs, int curr_atoms, int curr_clas_atoms);
	double get_prob_mass(bin_enc state) const {throw std::logic_error("Not Implemented.");}
	double get_atom_prob_mass(bin_enc atom) const;
	bin_enc halving(double prob) const {throw std::logic_error("Not Implemented.");} // serial halving algorithm
	bin_enc halving_omp(double prob) const {throw std::logic_error("Not Implemented.");} // OpenMP halving algorithm
	bin_enc halving_mpi(double prob) const;  // MPI halving algorithm 
	bin_enc halving_hybrid(double prob) const; // hybrid MPI + OpenMP halving algorithm
	inline bin_enc halving_mp(double prob) const {return halving_hybrid(prob);}
};