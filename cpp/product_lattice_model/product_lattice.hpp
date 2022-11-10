#pragma once

#include <iostream>
class Product_lattice{

	protected:
	int atom_;
	int variant_;
	double* post_probs_;
	int test_ct_;
	int* clas_stat_;
	int total_st_;

	public:
	// product_lattice(); // default constructor
	Product_lattice(int n_atom, int n_variant, double* pi0) : atom_(n_atom), variant_(n_variant){
		total_st_ = 1 << (atom_ * variant_);
		clas_stat_ = new int[nominal_pool_size()]();
        post_probs_ = new double[total_st_];
		prior_probs(pi0);
		test_ct_ = 0;
	}

	Product_lattice(const Product_lattice &other, int assert){
        atom_ = other.atom_;
		variant_ = other.variant_;
		test_ct_ = other.test_ct_;
		total_st_ = other.total_st_;
        if(assert == 0){ // broadcast
			clas_stat_ = new int[nominal_pool_size()];
            copy_classification_stat(other.clas_stat_);
            posterior_probs(other.post_probs_);
        }
        else if (assert == 1){ // statistical analysis
			clas_stat_ = new int[nominal_pool_size()];
            copy_classification_stat(other.clas_stat_);
            posterior_probs(other.post_probs_);
        }
        else if (assert == 2){ // tree internal copy
            // do nithing
        }
	}

	virtual ~Product_lattice(){
		delete[] post_probs_;
		delete[] clas_stat_;
	}

	virtual Product_lattice *create(int n_atom, int n_variant, double *pi9) const = 0;
	virtual Product_lattice *clone(int assert) const = 0;
	int atom() const {return atom_;}
	void atom(int atom){atom_ = atom;}
	int variant() const {return variant_;};
	void variant(int variant){variant_ = variant;}
	int* classification_stat() const {return clas_stat_;}
	void classification_stat(int* clas_stat){clas_stat_ = clas_stat;}
	void copy_classification_stat(int* clas_stat);
	double* posterior_probs() const {return post_probs_;};
	void posterior_probs(double* post_probs){post_probs_ = post_probs;}
	void copy_posterior_probs(double* post_probs);
	int test_count() const {return test_ct_;};
	void test_count(int test_ct){test_ct_ = test_ct;};
	int total_state() const {return total_st_;}
	int nominal_pool_size() const {return atom_ * variant_;}
	int* get_up_set (int state, int* ret) const;
	int* generate_power_set_adder(int* add_index, int state, int* ret) const;
	void prior_probs(double* pi0);
	double prior_prob(int state, double* pi0);
	void reset_test_count(){test_ct_ = 0;};
	void update_probs(int experiment, int response, double thres_up, double thres_lo, double** dilution);
	// void update_probs_parallel(int experiment, int response, double thres_up, double thres_lo);
	// void update_probs_in_place(int experiment, int response, double thres_up, double thres_lo);
	virtual double* calc_probs(int experiment, int response, double** dilution) = 0;
	void update_metadata(double thres_up, double thres_lo);
	double get_prob_mass(int state) const;
	bool is_classified() const;
	int halving(double prob) const;
	int halving_parallel(double prob) const;
	virtual double response_prob(int experiment, int response, int true_state, double** dilution) const = 0;
	double** generate_dilution(double alpha, double h) const;
	virtual void type(){std::cout << "Lattice Model" << std::endl;}

};