#pragma once
#include "../core.hpp"
#include "halving_res/halving_res.hpp"
#include "lattice_shrinking/lattice_shrinking.hpp"
/*
 * READ BEFORE USE:
 * In this version of lattice model, state is represented using A0B0A1B1...
 * The input of prior should also follow this pattern.
 */
class Product_lattice{

	protected:
	static int rank, world_size;
	int _curr_subjs; // counter
	int _variants; // counter
	double* _post_probs; 
	int _test_ct = 0; // counter
	bin_enc _pos_clas_atoms = 0; // BE (binary encoding)
	bin_enc _neg_clas_atoms = 0; // BE
	bin_enc _clas_subjs = 0; // BE, variables used for lattice shrinking

	public:
	Product_lattice(){}
	Product_lattice(int n_atom, int n_variant, double* pi0);
	Product_lattice(const Product_lattice &other, int copy_op);
	virtual ~Product_lattice();
	virtual Product_lattice *create(int n_atom, int n_variant, double *pi9) const = 0;
	virtual Product_lattice *clone(int copy_op) const = 0;
	inline int curr_subjs() const {return _curr_subjs;}
	inline int variants() const {return _variants;};
	inline bin_enc pos_clas_atoms() const {return _pos_clas_atoms;}
	inline bin_enc neg_clas_atoms() const {return _neg_clas_atoms;}
	inline int curr_atoms() const {return _curr_subjs * _variants;}
	inline int orig_atoms() const {return orig_subjs() * _variants;}
	inline int orig_subjs() const {return _curr_subjs + __builtin_popcount(_clas_subjs);}
	inline bin_enc clas_subjs() const {return _clas_subjs;}
	inline int total_state() const {return (1 << (_curr_subjs * _variants));}
	inline double* posterior_probs() const {return _post_probs;};
	inline void posterior_probs(double* post_probs){_post_probs = post_probs;}
	inline int test_count() const {return _test_ct;};
	inline bool is_classified() const {return __builtin_popcount(_pos_clas_atoms | _neg_clas_atoms) == orig_atoms();}
	inline void reset_test_count(){_test_ct = 0;};
	bin_enc* get_up_set (bin_enc state, bin_enc* ret) const;
	void generate_power_set_adder(bin_enc* add_index, int index_len, bin_enc state, bin_enc* ret) const;
	virtual void prior_probs(double* pi0);
	double prior_prob(bin_enc state, double* pi0) const;
	void update_probs(bin_enc experiment, bin_enc response, double thres_up, double thres_lo, double** dilution);
	void update_probs_in_place(bin_enc experiment, bin_enc response, double thres_up, double thres_lo, double** dilution);
	virtual double* calc_probs(bin_enc experiment, bin_enc response, double** dilution);
	virtual void calc_probs_in_place(bin_enc experiment, bin_enc response, double** dilution);
	virtual void update_metadata(double thres_up, double thres_lo);
	virtual void update_metadata_with_shrinking(double thres_up, double thres_lo);
	virtual double get_prob_mass(bin_enc state) const;
	virtual double get_atom_prob_mass(bin_enc atom) const;
	virtual bin_enc halving(double prob) const; // serial halving algorithm
	virtual bin_enc halving_omp(double prob) const; // OpenMP halving algorithm
	virtual bin_enc halving_mpi(double prob) const;  // MPI halving algorithm 
	virtual bin_enc halving_hybrid(double prob) const; // hybrid MPI + OpenMP halving algorithm
	virtual double response_prob(bin_enc experiment, bin_enc response, bin_enc true_state, double** dilution) const = 0;
	double** generate_dilution(double alpha, double h) const;
	virtual std::string type() const = 0;

	static void MPI_Product_lattice_Initialize();
	static void MPI_Product_lattice_Free();
};