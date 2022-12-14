#include "product_lattice.hpp"
#include <cstdlib>
#include <limits>
#include <cmath> // std:abs(double) support for older gcc

Product_lattice::Product_lattice(int n_atom, int n_variant, double* pi0) : atom_(n_atom), variant_(n_variant){
	pos_clas_ = 0;
	neg_clas_ = 0;
    post_probs_ = new double[(1 << (atom_ * variant_))];
	prior_probs(pi0);
	test_ct_ = 0;
}

Product_lattice::Product_lattice(const Product_lattice &other, int assert){
    atom_ = other.atom_;
	variant_ = other.variant_;
	pos_clas_ = other.pos_clas_;
	neg_clas_ = other.neg_clas_;
	test_ct_ = other.test_ct_;
	if(assert == 0){ // broadcast
        posterior_probs(other.post_probs_);
    }
    else if (assert == 1){ // statistical analysis
        posterior_probs(other.post_probs_);
    }
    else if (assert == 2){ // tree internal copy
		post_probs_ = nullptr;
    }
}

Product_lattice::~Product_lattice(){
	if(post_probs_ != nullptr) delete[] post_probs_;
}

void Product_lattice::copy_posterior_probs(double* post_probs){
    int index = total_state();
	for(int i = 0; i < index; i++){
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
					temp |= add_index[j];
				}
			}
			ret[i] = temp;
		}
	return ret;
}

void Product_lattice::prior_probs(double* pi0){
	int index = total_state();
	for (int i = 0; i < index; i++) {
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

void Product_lattice::update_probs_in_place(int experiment, int response, double thres_up, double thres_lo, double** dilution){
	calc_probs_in_place(experiment, response, dilution);
    update_metadata(thres_up, thres_lo);
    test_ct_++;
}

// void Product_lattice::update_metadata(double thres_up, double thres_lo){
// 	int n = nominal_pool_size()-1; // shared variables

// 	#pragma omp parallel for schedule(dynamic)
//     for (int i = 0; i < nominal_pool_size(); i++) {
// 		int placement = (1 << i);
// 		if (((pos_clas_ + neg_clas_) & placement) != 0)
// 			continue; // skip checking since it's already classified as either positive or negative
// 		int atom = 1 << i;

// 		// customized get_prob_mass for atoms
// 		double prob_mass = 0.0;
// 		int add_index[n], counter = 0;
// 		for(int j = 0; j < n+1; j++){
// 			if((atom & (1 << j)) == 0){
// 				add_index[counter++] = (1 << j);
// 			}
// 		}

// 		int temp;
// 		for(int j = 0; j < (1 << n); j++){
// 			temp = atom;
// 			for(int k = 0; k < n; k++){
// 				if((j & (1 << k))){
// 					temp |= add_index[k];
// 				}
// 			}
// 			prob_mass += post_probs_[temp];
// 		}

// 		#pragma omp critical // nominal_pool_size is small so omit contention overhead
//         {
// 		if (prob_mass < thres_lo)
// 			pos_clas_ |= placement; // classified as positive
// 		else if (prob_mass > (1 - thres_up))
// 			neg_clas_ |= placement; // classified as negative
// 		}
// 	}
// }

void Product_lattice::update_metadata(double thres_up, double thres_lo){
	#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nominal_pool_size(); i++) {
		int placement = (1 << i);
		if (((pos_clas_ + neg_clas_) & placement) != 0)
			continue; // skip checking since it's already classified as either positive or negative
		int atom = 1 << i;
		double prob_mass = get_prob_mass(atom);

		#pragma omp critical // nominal_pool_size is small so omit contention overhead
        {
		if (prob_mass < thres_lo)
			pos_clas_ |= placement; // classified as positive
		else if (prob_mass > (1 - thres_up))
			neg_clas_ |= placement; // classified as negative
		}
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
				temp |= add_index[j];
			}
		}
		ret += post_probs_[temp];
	}

	delete[] add_index;
	return ret;
	
}
bool Product_lattice::is_classified() const{
	return __builtin_popcount(pos_clas_ + neg_clas_) == nominal_pool_size();
}

double* Product_lattice::calc_probs(int experiment, int response, double** dilution){
	int total_stat = 1 << (atom_ * variant_);
	double* ret = new double[total_stat];
	double denominator = 0.0;
	// #pragma omp parallel for schedule(static) reduction (+ : denominator) // INVESTIGATE: slowdown than serial
	for (int iter = 0; iter < total_stat; iter++) {
		ret[iter] = post_probs_[iter] * response_prob(experiment, response, iter, dilution);
		denominator += ret[iter];
	}
	double denominator_inv = 1 / denominator; // division has much higher instruction latency and throughput than multiplication
	// #pragma omp parallel for schedule(static)
	for (int i = 0; i < total_stat; i++) {
		ret[i] *= denominator_inv;
	}
	return ret;
}

void Product_lattice::calc_probs_in_place(int experiment, int response, double **__restrict__ dilution){
	double denominator = 0.0;
	int total_stat = 1 << (atom_ * variant_);
	// #pragma omp parallel for schedule(static) reduction (+ : denominator)
	for (int iter = 0; iter < total_stat; iter++) {
		post_probs_[iter] *= response_prob(experiment, response, iter, dilution);
		denominator += post_probs_[iter];
	}
	// #pragma omp parallel for schedule(static)
	for (int i = 0; i < total_stat; i++) {
		post_probs_[i] /= denominator;
	}
}

/**
 * Implementation V2
*/
// int Product_lattice::halving(double prob) const{
// 	int candidate = 0;
// 	int s_iter;
// 	int experiment;
// 	double min = 2.0;
// 	int partition_id = 0;
// 	double partition_mass[(1 << variant_)];
// 	for (experiment = 0; experiment < (1 << atom_); experiment++) {
// 		// reset partition_mass
// 		for (int i = 0; i < (1 << variant_); i++)
// 			partition_mass[i] = 0.0;
// 		// tricky: for each state, check each variant of actively
// 		// pooled subjects to see whether they are all 1.
// 		for (s_iter = 0; s_iter < (1 << (atom_ * variant_)); s_iter++) {
// 			// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
// 			for (int variant = 0; variant < variant_; variant++) {
// 				if ((experiment & (s_iter >> (variant * atom_))) != experiment) {
// 					partition_id |= (1 << variant);
// 				}
// 			}
// 			partition_mass[partition_id] += post_probs_[s_iter];
// 			partition_id = 0;
// 		}
// 		double temp = 0.0;
// 		for (int i = 0; i < (1 << variant_); i++) {
// 			temp += std::abs(partition_mass[i] - prob);
// 		}
// 		if (temp < min) {
// 			min = temp;
// 			candidate = experiment;
// 		}
// 	}
// 	return candidate;
// }

// int Product_lattice::halving(double prob) const{
// 	int candidate = 0;
// 	int s_iter;
// 	int experiment;
// 	double min = 2.0;
// 	int partition_id = 0;
// 	double partition_mass[(1 << variant_)];
// 	for (experiment = 0; experiment < (1 << atom_); experiment++) {
// 		// reset partition_mass
// 		for (int i = 0; i < (1 << variant_); i++)
// 			partition_mass[i] = 0.0;
// 		// tricky: for each state, check each variant of actively
// 		// pooled subjects to see whether they are all 1.
// 		for (s_iter = 0; s_iter < (1 << (atom_ * variant_)); s_iter++) {
// 			// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
// 			for (int variant = 0; variant < variant_; variant++) {
// 				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
// 				//evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way, 
// 				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the 
// 				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0 
// 				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
// 				partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * atom_))) - experiment) >> 31));
				
// 			}
// 			partition_mass[partition_id] += post_probs_[s_iter];
// 			partition_id = 0;
// 		}
// 		double temp = 0.0;
// 		for (int i = 0; i < (1 << variant_); i++) {
// 			temp += std::abs(partition_mass[i] - prob);
// 		}
// 		if (temp < min) {
// 			min = temp;
// 			candidate = experiment;
// 		}
// 	}
// 	return candidate;
// }

int Product_lattice::halving(double prob) const{
	int candidate = 0;
	int s_iter;
	int experiment;
	double min = 2.0;
	int partition_id = 0;
	double partition_mass[(1 << atom_) * (1 << variant_)]{0.0};
	
	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	for (s_iter = 0; s_iter < (1 << (atom_ * variant_)); s_iter++) {
		// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
		for (experiment = 0; experiment < (1 << atom_); experiment++) {
			for (int variant = 0; variant < variant_; variant++) {
				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
				//evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way, 
				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the 
				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0 
				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
				partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * atom_))) - experiment) >> 31));
			
			}
			partition_mass[experiment * (1 << variant_) + partition_id] += post_probs_[s_iter];
			partition_id = 0;
		}
	}
	double temp = 0.0;
	for (experiment = 0; experiment < (1 << atom_); experiment++) {
		for (int i = 0; i < (1 << variant_); i++) {
			temp += std::abs(partition_mass[experiment * (1 << variant_) + i] - prob);
		}
		if (temp < min) {
			min = temp;
			candidate = experiment;
		}
		temp = 0.0;
	}
	
	return candidate;
}

void halving_min(Halving_res& a, Halving_res& b){
	if(a.min > b.min){
		a.min = b.min;
		a.candidate = b.candidate;
	}
}

int Product_lattice::halving_omp(double prob) const{
	double partition_mass[(1 << atom_) * (1 << variant_)]{0.0};
	
	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	#pragma omp parallel for schedule(static) reduction (+ : partition_mass)
	for (int s_iter = 0; s_iter < (1 << (atom_ * variant_)); s_iter++) {
		// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
		for (int experiment = 0; experiment < (1 << atom_); experiment++) {
			int partition_id = 0;
			for (int variant = 0; variant < variant_; variant++) {
				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
				//evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way, 
				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the 
				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0 
				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
				partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * atom_))) - experiment) >> 31));
			
			}
			partition_mass[experiment* (1 << variant_) + partition_id] += post_probs_[s_iter];
		}
	}
	
	Halving_res halving_res;
	#pragma omp declare reduction(Halving_Min : Halving_res : halving_min(omp_out, omp_in)) initializer (omp_priv=Halving_res())
	#pragma omp parallel for schedule(static) reduction (Halving_Min : halving_res)	
	for (int experiment = 0; experiment < (1 << atom_); experiment++) {
		double temp = 0.0;
		for (int i = 0; i < (1 << variant_); i++) {
			temp += std::abs(partition_mass[experiment * (1 << variant_) + i] - prob);
		}
		if (temp < halving_res.min) {
			halving_res.min = temp;
			halving_res.candidate = experiment;
		}
	}
	
	return halving_res.candidate;
}

// void Product_lattice::halving(double prob, int rank, int world_size, Halving_res& halving_res) const{
// 	halving_res.min = 2.0; // reset min
// 	int partition_id = 0;
// 	double partition_mass[(1 << variant_)];
//     int start_experiment = (1 << atom_) / world_size * rank;
//     int stop_experiment = (1 << atom_) / world_size * (rank+1); 

// 	for (int experiment = start_experiment; experiment < stop_experiment; experiment++) {
// 		// reset partition_mass
// 		for (int i = 0; i < (1 << variant_); i++)
// 			partition_mass[i] = 0.0;
// 		// tricky: for each state, check each variant of actively
// 		// pooled subjects to see whether they are all 1.
// 		for (int s_iter = 0; s_iter < (1 << (atom_ * variant_)); s_iter++) {
// 			// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
// 			for (int variant = 0; variant < variant_; variant++) {
// 				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
// 				//evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way, 
// 				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the 
// 				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0 
// 				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
// 				partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * atom_))) - experiment) >> 31));
// 			}
// 			partition_mass[partition_id] += post_probs_[s_iter];
// 			partition_id = 0;
// 		}
// 		double temp = 0.0;
// 		for (int i = 0; i < (1 << variant_); i++) {
// 			temp += std::abs(partition_mass[i] - prob);
// 		}
// 		if (temp < halving_res.min) {
// 			halving_res.min = temp;
// 			halving_res.candidate = experiment;
// 		}
// 	}
// }

void Product_lattice::halving(double prob, int rank, int world_size, Halving_res& halving_res) const{
	halving_res.min = 2.0; // reset min
	int partition_id = 0;
    const int start_experiment = (1 << atom_) / world_size * rank;
    const int stop_experiment = (1 << atom_) / world_size * (rank+1); 
	double partition_mass[(stop_experiment - start_experiment) * (1 << variant_)]{0.0};

	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	for (int s_iter = 0; s_iter < (1 << (atom_ * variant_)); s_iter++) {
		// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
		for (int experiment = start_experiment; experiment < stop_experiment; experiment++) {
			for (int variant = 0; variant < variant_; variant++) {
				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
				//evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way, 
				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the 
				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0 
				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
				partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * atom_))) - experiment) >> 31));
			
			}
			partition_mass[(experiment - start_experiment) * (1 << variant_) + partition_id] += post_probs_[s_iter];
			partition_id = 0;
		}
	}
	double temp = 0.0;
	for (int experiment = start_experiment; experiment < stop_experiment; experiment++) {
		for (int i = 0; i < (1 << variant_); i++) {
			temp += std::abs(partition_mass[(experiment - start_experiment) * (1 << variant_) + i] - prob);
		}
		if (temp < halving_res.min) {
			halving_res.min = temp;
			halving_res.candidate = experiment;
		}
		temp = 0.0;
	}
}

void Product_lattice::halving_omp(double prob, int rank, int world_size, Halving_res& halving_res) const{
	halving_res.min = 2.0; // reset min
    int start_experiment = (1 << atom_) / world_size * rank;
    int stop_experiment = (1 << atom_) / world_size * (rank+1);
	double partition_mass[(stop_experiment - start_experiment) * (1 << variant_)]{0.0}; 

	#pragma omp parallel for schedule(static) reduction (+ : partition_mass)
	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	for (int s_iter = 0; s_iter < (1 << (atom_ * variant_)); s_iter++) {
		int partition_id = 0;
		// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
		for (int experiment = start_experiment; experiment < stop_experiment; experiment++) {
			for (int variant = 0; variant < variant_; variant++) {
				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
				//evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way, 
				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the 
				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0 
				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
				partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * atom_))) - experiment) >> 31));
			
			}
			partition_mass[(experiment - start_experiment) * (1 << variant_) + partition_id] += post_probs_[s_iter];
			partition_id = 0;
		}
	}

	#pragma omp declare reduction(Halving_Min : Halving_res : halving_min(omp_out, omp_in)) initializer (omp_priv=Halving_res())
	#pragma omp parallel for schedule(static) reduction (Halving_Min : halving_res)
	for (int experiment = start_experiment; experiment < stop_experiment; experiment++) {
		double temp = 0.0;
		for (int i = 0; i < (1 << variant_); i++) {
			temp += std::abs(partition_mass[(experiment - start_experiment) * (1 << variant_) + i] - prob);
		}
		if (temp < halving_res.min) {
			halving_res.min = temp;
			halving_res.candidate = experiment;
		}
		temp = 0.0;
	}
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