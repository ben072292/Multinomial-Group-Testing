#pragma once

#include <iostream>
/*
 * READ BEFORE USE:
 * In this version of lattice model, state is represented using A0B0A1B1...
 * The input of prior should also follow this pattern.
 */
class Product_lattice{

	protected:
	int atom_;
	int variant_;
	double* post_probs_;
	int test_ct_;
	int pos_clas_;
	int neg_clas_;

	public:
	Product_lattice(int n_atom, int n_variant, double* pi0);
	Product_lattice(const Product_lattice &other, int assert);
	virtual ~Product_lattice();
	virtual Product_lattice *create(int n_atom, int n_variant, double *pi9) const = 0;
	virtual Product_lattice *clone(int assert) const = 0;
	inline int atom() const {return atom_;}
	inline void atom(int atom){atom_ = atom;}
	inline int variant() const {return variant_;};
	inline void variant(int variant){variant_ = variant;}
	inline int pos_clas() const {return pos_clas_;}
	inline void pos_clas(int val) {pos_clas_ = val;}
	inline int neg_clas() const {return neg_clas_;}
	inline void neg_clas(int val) {neg_clas_ = val;}
	inline double* posterior_probs() const {return post_probs_;};
	inline void posterior_probs(double* post_probs){post_probs_ = post_probs;}
	void copy_posterior_probs(double* post_probs);
	inline int test_count() const {return test_ct_;};
	inline void test_count(int test_ct){test_ct_ = test_ct;};
	inline int total_state() const {return (1 << (atom_ * variant_));}
	inline int nominal_pool_size() const {return atom_ * variant_;}
	int* get_up_set (int state, int* ret) const;
	int* generate_power_set_adder(int* add_index, int state, int* ret) const;
	void prior_probs(double* pi0);
	double prior_prob(int state, double* pi0) const;
	void reset_test_count(){test_ct_ = 0;};
	void update_probs(int experiment, int response, double thres_up, double thres_lo, double** dilution);
	void update_probs_in_place(int experiment, int response, double thres_up, double thres_lo, double** dilution);
	// void update_probs_parallel(int experiment, int response, double thres_up, double thres_lo);
	// void update_probs_in_place(int experiment, int response, double thres_up, double thres_lo);
	double* calc_probs(int experiment, int response, double** dilution);
	void calc_probs_in_place(int experiment, int response, double** dilution);
	void update_metadata(double thres_up, double thres_lo);
	double get_prob_mass(int state) const;
	bool is_classified() const;
	int halving(double prob) const;
	double* halving(double prob, int rank, int world_size) const; // for MPI split task
	virtual double response_prob(int experiment, int response, int true_state, double** dilution) const = 0;
	double** generate_dilution(double alpha, double h) const;
	virtual void type(){std::cout << "Lattice Model" << std::endl;}
};