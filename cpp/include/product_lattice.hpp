#pragma once
#include "core.hpp"
#include "halving_res.hpp"
#include "lattice_shrinking.hpp"

typedef enum copy_op
{
	NO_COPY_PROB_DIST,
	SHALLOW_COPY_PROB_DIST,
	DEEP_COPY_PROB_DIST
} copy_op_t;

enum parallelism
{
	MODEL_PARALLELISM,
	DATA_PARALLELISM = 2
};

enum dilution
{
	NON_DILUTION,
	DILUTION
};

enum lattice_type // follow old conventions
{
	MP_NON_DILUTION = MODEL_PARALLELISM + NON_DILUTION + 1,
	MP_DILUTION = MODEL_PARALLELISM + DILUTION + 1,
	DP_NON_DILUTION = DATA_PARALLELISM + NON_DILUTION + 1,
	DP_DILUTION = DATA_PARALLELISM + DILUTION + 1
};

/*
 * READ BEFORE USE:
 * In this version of lattice model, state is represented using A0B0A1B1...
 * The input of prior should also follow this pattern.
 */

// Product_lattice parallelism
class Product_lattice
{

protected:
	static int rank, world_size, _orig_subjs, _variants;
	static double *_pi0;
	int _curr_subjs;			 // counter
	bin_enc _pos_clas_atoms = 0; // BE (binary encoding)
	bin_enc _neg_clas_atoms = 0; // BE
	bin_enc _clas_subjs = 0;	 // BE, variables used for lattice shrinking
	double *_post_probs;

public:
	Product_lattice() {}
	Product_lattice(int n_atom, int n_variant, double *pi0);
	Product_lattice(const Product_lattice &other, copy_op_t op);
	virtual ~Product_lattice();
	virtual Product_lattice *create(int n_atom, int n_variant, double *pi0) const = 0;
	virtual Product_lattice *clone(copy_op_t op) const = 0;
	inline int curr_subjs() const { return _curr_subjs; }
	inline static int variants() { return _variants; };
	inline static double *pi0() { return _pi0; }
	inline static void pi0(double *pi0) { _pi0 = pi0; }
	inline bin_enc pos_clas_atoms() const { return _pos_clas_atoms; }
	inline bin_enc neg_clas_atoms() const { return _neg_clas_atoms; }
	inline int curr_atoms() const { return _curr_subjs * _variants; }
	inline int orig_atoms() const { return _orig_subjs * _variants; }
	inline static int orig_subjs() { return _orig_subjs; }
	inline bin_enc clas_subjs() const { return _clas_subjs; }
	inline int total_states() const { return (1 << (_curr_subjs * _variants)); }
	inline double *posterior_probs() const { return _post_probs; };
	virtual double posterior_prob(bin_enc state) const;
	inline void posterior_probs(double *post_probs) { _post_probs = post_probs; }
	inline bool is_classified() const { return __builtin_popcount(_pos_clas_atoms | _neg_clas_atoms) == orig_atoms(); }
	bin_enc *get_up_set(bin_enc state, bin_enc *ret) const;
	void generate_power_set_adder(bin_enc *add_index, int index_len, bin_enc state, bin_enc *ret) const;
	virtual void prior_probs(double *pi0);
	double prior_prob(bin_enc state, double *pi0) const;
	void update_probs(bin_enc experiment, bin_enc response, double **dilution);
	void update_probs_in_place(bin_enc experiment, bin_enc response, double **dilution);
	virtual double *calc_probs(bin_enc experiment, bin_enc response, double **dilution);
	virtual void calc_probs_in_place(bin_enc experiment, bin_enc response, double **dilution);
	virtual void update_metadata(double thres_up, double thres_lo);
	virtual bool update_metadata_with_shrinking(double thres_up, double thres_lo);
	virtual void shrinking(int curr_clas_atoms);
	virtual double get_prob_mass(bin_enc state) const;
	virtual double get_atom_prob_mass(bin_enc atom) const;
	virtual bin_enc halving(double prob) const;
	virtual enum parallelism parallelism() const { return DATA_PARALLELISM; }
	virtual enum dilution dilution() const { return NON_DILUTION; }
	virtual Product_lattice *convert_parallelism() { return this; }

	/**
	 * Serial
	 */
	virtual bin_enc halving_serial(double prob) const;

	/**
	 * Intra-node using OpenMP
	 */
	virtual bin_enc halving_omp(double prob) const;

	/**
	 * Inter-node using MPI
	 */
	virtual bin_enc halving_mpi(double prob) const;

	/**
	 * Inter-node using MPI + vectorization
	 */
	virtual bin_enc halving_mpi_vectorize(double prob) const;

	/**
	 * Inter-node using MPI + intra-node using OpenMP
	 */
	virtual bin_enc halving_hybrid(double prob) const;
	virtual double response_prob(bin_enc experiment, bin_enc response, bin_enc true_state, double **dilution) const = 0;
	virtual std::string type() const = 0;

	static void MPI_Product_lattice_Initialize();
	static void MPI_Product_lattice_Free();
};