#include "product_lattice.hpp"
#include <cstdlib>
#include <limits>

void Product_lattice::copy_classification_stat(int* clas_stat){
    for(int i = 0; i < nominal_pool_size(); i++){
        clas_stat_[i] = clas_stat[i];
    }
}

void Product_lattice::copy_posterior_probs(double* post_probs){
    for(int i = 0; i < total_st_; i++){
        post_probs_[i] = post_probs[i];
    }
}

int* Product_lattice::get_up_set(int state, int* ret) const{
    int *add_index = new int[nominal_pool_size() - __builtin_popcount(state)];
    int counter = 0, i, index;
    for(i = 0; i < nominal_pool_size(); i++){
        index = (1 << i);
        if((state & index) == 0){
            add_index[counter++] = index;
        }
    }
    generate_power_set_adder(add_index, state, ret);
    delete[] add_index;
    return ret;
}

int* Product_lattice::generate_power_set_adder(int* add_index, int state, int* ret) const{
    int n = nominal_pool_size() - __builtin_popcount(state), pow_set_size = 1 << n;
		int i, j, temp;
		for (i = 0; i < pow_set_size; i++) {
			temp = state;
			for (j = 0; j < n; j++) {
				/*
				 * Check if j-th bit in the counter is set If set then print j-th element from
				 * set
				 */
				if ((i & (1 << j))) {
					temp += add_index[j];
				}
			}
			ret[i] = temp;
		}
	return ret;
}

void Product_lattice::prior_probs(double* pi0){
	for (int i = 0; i < total_st_; i++) {
		post_probs_[i] = prior_prob(i, pi0);
	}
}
double Product_lattice::prior_prob(int state, double* pi0) const{
    double prob = 1.0;
	for (int i = 0; i < nominal_pool_size(); i++) {
		if ((state & (1 << i)) == 0)
			prob *= pi0[i];
		else
			prob *= (1.0 - pi0[i]);
	}
	return prob;
}

void Product_lattice::update_probs(int experiment, int response, double thres_up, double thres_lo, double** dilution){
	post_probs_ = calc_probs(experiment, response, dilution);
    update_metadata(thres_up, thres_lo);
    test_ct_++;
}
// void product_lattice::update_probs_parallel(int experiment, int response, double thres_up, double thres_lo);
// void product_lattice::update_probs_in_place(int experiment, int response, double thres_up, double thres_lo);

void Product_lattice::update_metadata(double thres_up, double thres_lo){
    for (int i = 0; i < nominal_pool_size(); i++) {
		if (clas_stat_[i] != 0)
			continue; // skip checking since it's already classified as either positive or negative
		int atom = 1 << i;
		double probMass = get_prob_mass(atom);

		if (probMass < thres_lo)
			clas_stat_[i] = -1; // classified as positive
		else if (probMass > (1 - thres_up))
			clas_stat_[i] = 1; // classified as negative
	}
}
double Product_lattice::get_prob_mass(int state) const{
    double ret = 0.0;
	int n = nominal_pool_size() - __builtin_popcount(state), pow_set_size = 1 << n, j, temp;
    int *add_index = new int[n];
    int counter = 0, i, index;
    for(i = 0; i < nominal_pool_size(); i++){
        index = (1 << i);
        if((state & index) == 0){
            add_index[counter++] = index;
        }
    }

	for (i = 0; i < pow_set_size; i++) {
		temp = state;
		for (j = 0; j < n; j++) {
			/*
			 * Check if j-th bit in the counter is set If set then print j-th element from
			 * set
			 */
			if ((i & (1 << j))) {
				temp += add_index[j];
			}
		}
		ret += post_probs_[temp];
	}

	delete[] add_index;
	return ret;
	
}
bool Product_lattice::is_classified() const{
    for (int i = 0; i < nominal_pool_size(); i++){
		if (clas_stat_[i] == 0){
			return false;
		}
	}	
	return true;
}
int Product_lattice::halving(double prob) const{
	int candidate = 0;
	int s_iter;
	int experiment;
	bool is_complement = false;
	double min = 2.0;
	int partition_id = 0;
	double* partition_mass = new double[(1 << variant_)];
	for (experiment = 0; experiment < (1 << atom_); experiment++) {
		// reset partition_mass
		for (int i = 0; i < (1 << variant_); i++)
			partition_mass[i] = 0.0;
		// tricky: for each state, check each variant of actively
		// pooled subjects to see whether they are all 1.
		for (s_iter = 0; s_iter < total_st_; s_iter++) {
			for (int variant = 0; variant < variant_; variant++) {
				for (int l = 0; l < atom_; l++) {
					if ((experiment & (1 << l)) != 0 && (s_iter & (1 << (l * variant_ + variant))) == 0) {
						is_complement = true;
						break;
					}
				}
				partition_id |= (is_complement ? 0 : (1 << variant));
				is_complement = false; // reset flag
			}
			partition_mass[partition_id] += post_probs_[s_iter];
			partition_id = 0;
		}
		// for (int i = 0; i < totalStates(); i++) {
		// System.out.print(partitionMap[i] + " ");
		// }
		// System.out.println();
		double temp = 0.0;
		for (int i = 0; i < (1 << variant_); i++) {
			temp += std::abs(partition_mass[i] - prob);
		}
		if (temp < min) {
			min = temp;
			candidate = experiment;
		}
	}
	delete[] partition_mass;
	return candidate;
}

// int product_lattice::halving_parallel(double prob);

double** Product_lattice::generate_dilution(double alpha, double h) const{
	double** ret = new double*[atom_];
	int k;
	for (int rk = 1; rk <= atom_; rk++) {
		ret[rk - 1] = new double[rk + 1];
		ret[rk - 1][0] = alpha;
		for (int r = 1; r <= rk; r++) {
			k = rk - r;
			ret[rk - 1][r] = 1 - alpha * r / (k * h + r);
		}
	}
	return ret;
}