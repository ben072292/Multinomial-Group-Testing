#pragma once
#include "product_lattice.hpp"
#include "product_lattice_dilution.hpp"
#include "product_lattice_non_dilution.hpp"
/*
 * READ BEFORE USE:
 * In this version of lattice model, state is represented using A0B0A1B1...
 * The input of prior should also follow this pattern.
 * Model parallelism implementation for lattice model
 */
class Product_lattice_mp : public virtual Product_lattice
{

protected:
	static double* temp_post_prob_holder;
	static MPI_Win win;
	inline bin_enc total_states_per_rank() const { return total_states() / world_size; }
	inline int state_to_offset(bin_enc state) const { return state % total_states_per_rank(); }
	inline int state_to_rank(bin_enc state) const { return state / total_states_per_rank(); }
	inline bin_enc offset_to_state(int offset) const { return offset + total_states_per_rank() * rank; }

public:
	Product_lattice_mp() {} // default constructor
	Product_lattice_mp(int n_atom, int n_variant, double *pi0);
	Product_lattice_mp(const Product_lattice &other, copy_op_t op) : Product_lattice(other, op){};
	double posterior_prob(bin_enc state) const override;
	void prior_probs(double *pi0) override;
	void update_probs(bin_enc experiment, bin_enc response, double **dilution) override;
	void update_probs_in_place(bin_enc experiment, bin_enc response, double **dilution) override;
	void update_metadata(double thres_up, double thres_lo) override;
	bool update_metadata_with_shrinking(double thres_up, double thres_lo) override;
	void shrinking(int curr_clas_atoms) override;
	double get_prob_mass(bin_enc state) const override { throw std::logic_error("Not Implemented."); }
	double get_atom_prob_mass(bin_enc atom) const override;
	bin_enc halving_serial(double prob) const override { throw std::logic_error("Serial halving algorithm is not supported in model parallelism."); }	 
	bin_enc halving_omp(double prob) const override { throw std::logic_error("OMP halving algorithm is not supported in model parallelism"); } 
	bin_enc halving_mpi(double prob) const override;												   
	bin_enc halving_mpi_vectorize(double prob) const override;									   
	bin_enc halving_hybrid(double prob) const override;											  
	bin_enc halving(double prob) const override;
	virtual enum parallelism parallelism() const override { return MODEL_PARALLELISM; }
	virtual enum dilution dilution() const override {return NON_DILUTION; }
	static void MPI_Product_lattice_Initialize(int atoms, int variants);
	static void MPI_Product_lattice_Free();
};