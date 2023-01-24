#include "product_lattice.hpp"
#include "lattice_shrinking/lattice_shrinking.hpp"

Product_lattice::Product_lattice(int n_subj, int n_variant, double* pi0) : curr_subj_(n_subj), variant_(n_variant){
    post_probs_ = new double[(1 << (curr_subj_ * variant_))];
	prior_probs(pi0);
}

Product_lattice::Product_lattice(const Product_lattice &other, int assert){
    curr_subj_ = other.curr_subj_;
	variant_ = other.variant_;
	pos_clas_ = other.pos_clas_;
	neg_clas_ = other.neg_clas_;
	test_ct_ = other.test_ct_;
	clas_subjs_ = other.clas_subjs_;
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
	int index_len_binary = curr_atoms() - __builtin_popcount(state);
    int *add_index = new int[1 << index_len_binary];
    int counter = 0;
    for(int i = 0; i < curr_atoms(); i++){
        if((state & (1 << i)) == 0){
            add_index[counter++] = (1 << i);
        }
    }
    generate_power_set_adder(add_index, index_len_binary, state, ret);
    delete[] add_index;
    return ret;
}

int* Product_lattice::generate_power_set_adder(int* add_index, int index_len_binary, int state, int* ret) const{
    int pow_set_size = 1 << index_len_binary;
		int i, j, temp;
		for (i = 0; i < pow_set_size; i++) {
			temp = state;
			for (j = 0; j < index_len_binary; j++) {
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
	for (int i = 0; i < curr_atoms(); i++) {
		if ((state & (1 << i)) == 0)
			prob *= pi0[i];
		else
			prob *= (1.0 - pi0[i]);
	}
	return prob;
}

void Product_lattice::update_probs(int experiment, int response, double thres_up, double thres_lo, double** dilution){
	post_probs_ = calc_probs(experiment, response, dilution);
    // update_metadata(thres_up, thres_lo);
	update_metadata_with_shrinking(thres_up, thres_lo);
    test_ct_++;
}

void Product_lattice::update_probs_in_place(int experiment, int response, double thres_up, double thres_lo, double** dilution){
	calc_probs_in_place(experiment, response, dilution);
    // update_metadata(thres_up, thres_lo);
	update_metadata_with_shrinking(thres_up, thres_lo);
    test_ct_++;
}

void Product_lattice::update_metadata(double thres_up, double thres_lo){
	#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < curr_atoms(); i++) {
		int placement = (1 << i);
		if ((pos_clas_ | neg_clas_) & placement)
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

void Product_lattice::update_metadata_with_shrinking(double thres_up, double thres_lo){
	int clas_atoms_bin = (pos_clas_ | neg_clas_); // same size as orig layout
	int orig_subjs = this->orig_subjs(); // called at the beginning to ensure correct value
	int curr_clas_atoms_bin = 0; // same size as curr layout
	int curr_atoms = curr_subj_ * variant_;

	#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < orig_subjs * variant_; i++) {
		int orig_index_bin = (1 << i); // binary index in decimal for original layout
		int curr_index_bin = orig_curr_ind_conv(i, clas_subjs_, orig_subjs, variant_); // binary index in decimal for current layout
		if ((clas_atoms_bin & orig_index_bin)){
			#pragma omp critical
			{
			curr_clas_atoms_bin |= curr_index_bin; // update 
			}
		}
		else{
			double prob_mass = get_prob_mass(curr_index_bin);
			#pragma omp critical // nominal_pool_size is small so omit contention overhead
        	{
			if (prob_mass < thres_lo){ // classified as positive
				pos_clas_ |= orig_index_bin;
				curr_clas_atoms_bin |= curr_index_bin;
			}
			else if (prob_mass > (1 - thres_up)){ // classified as positive
				neg_clas_ |= orig_index_bin;
				curr_clas_atoms_bin |= curr_index_bin;
			}
			}
		}
	}

	curr_clas_atoms_bin = curr_shrinkable_atoms(curr_clas_atoms_bin, curr_subj_, variant_);
	if(!curr_clas_atoms_bin) return; // if no new classifications, we skip the rest
	int reduce_count = __builtin_popcount(curr_clas_atoms_bin);
	int base_count = curr_atoms - reduce_count;
	int* base_index = new int[base_count];
	int* reduce_index = new int[reduce_count];
	base_count = 0;
	reduce_count = 0;
	for(int i = 0; i < curr_atoms; i++){
		if((curr_clas_atoms_bin & (1 << i)) != 0)
			reduce_index[reduce_count++] = (1 << i);
		else
			base_index[base_count++] = (1 << i);
	}

	// Equivalent to:
	// int* base_adder = new int[(1 << base_count)];
	// int* reduce_adder = new int[(1 << reduce_count)];
	// generate_power_set_adder(base_index, base_count, 0, base_adder);
	// generate_power_set_adder(reduce_index, reduce_count, 0, reduce_adder);
	// for(int i = 0; i < base_count; i++){
	// 	 for(int j = 0; j < reduce_count; j++){
	// 		shrinked_post_probs[i] += (post_probs_[base_adder[i] + reduce_adder[j]]);
	// 	 }
	// }
	// But greatly reduce memory access
	int ele_base = 0, ele_reduce = 0;
	for (int i = 0; i < (1 << base_count); i++) {
		ele_base = 0;
		for (int j = 0; j < base_count; j++) {
			/*
			 * Check if j-th bit in the counter is set If set then print j-th element from
			 * set
			 */
			if ((i & (1 << j))) {
				ele_base |= base_index[j];
			}
		}
		for(int k = 0; k < (1 << reduce_count); k++){
			ele_reduce = 0;
			for(int l = 0; l < reduce_count; l++){
				/*
			 	* Check if j-th bit in the counter is set If set then print j-th element from
			 	* set
			 	*/
			 	if((k & (1 << l))) {
					ele_reduce |= reduce_index[l];
				}
			}
			if(i != ele_base + ele_reduce)
				post_probs_[i] += post_probs_[ele_base + ele_reduce]; // in-place modfication because each state is uniquely merged
			// std::cout << i << " " << ele_base + ele_reduce << " " << post_probs_[ele_base + ele_reduce] << std::endl;
		}
	}
	delete[] base_index;
	delete[] reduce_index; 
	clas_subjs_ = update_clas_subj(pos_clas_ | neg_clas_, orig_subjs, variant_);
	curr_subj_ = orig_subjs - __builtin_popcount(clas_subjs_);
}

double Product_lattice::get_prob_mass(int state) const{
    double ret = 0.0;
	int n = curr_atoms() - __builtin_popcount(state), pow_set_size = 1 << n, temp;
    int *add_index = new int[n];
    int counter = 0, index;
    for(int i = 0; i < curr_atoms(); i++){
        index = (1 << i);
        if(!(state & index)){
            add_index[counter++] = index;
        }
    }

	for (int i = 0; i < pow_set_size; i++) {
		temp = state;
		for (int j = 0; j < n; j++) {
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
	return __builtin_popcount(pos_clas_ | neg_clas_) == orig_atoms();
}

/**
 * experiment is an integer with range (0, 2^N)
 * response is an integer with range [0, 2^k)
*/
double* Product_lattice::calc_probs(int experiment, int response, double** dilution){
	int total_state = 1 << (curr_subj_ * variant_);
	double* ret = new double[total_state];
	double denominator = 0.0;
	// #pragma omp parallel for schedule(static) reduction (+ : denominator) // INVESTIGATE: slowdown than serial
	for (int iter = 0; iter < total_state; iter++) {
		ret[iter] = post_probs_[iter] * response_prob(experiment, response, iter, dilution);
		denominator += ret[iter];
	}
	double denominator_inv = 1 / denominator; // division has much higher instruction latency and throughput than multiplication
	// #pragma omp parallel for schedule(static)
	for (int i = 0; i < total_state; i++) {
		ret[i] *= denominator_inv;
	}
	return ret;
}

void Product_lattice::calc_probs_in_place(int experiment, int response, double **__restrict__ dilution){
	double denominator = 0.0;
	int total_state = 1 << (curr_subj_ * variant_);
	// #pragma omp parallel for schedule(static) reduction (+ : denominator)
	for (int iter = 0; iter < total_state; iter++) {
		post_probs_[iter] *= response_prob(experiment, response, iter, dilution);
		denominator += post_probs_[iter];
	}
	// #pragma omp parallel for schedule(static)
	for (int i = 0; i < total_state; i++) {
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
// 	for (experiment = 0; experiment < (1 << curr_subj_); experiment++) {
// 		// reset partition_mass
// 		for (int i = 0; i < (1 << variant_); i++)
// 			partition_mass[i] = 0.0;
// 		// tricky: for each state, check each variant of actively
// 		// pooled subjects to see whether they are all 1.
// 		for (s_iter = 0; s_iter < (1 << (curr_subj_ * variant_)); s_iter++) {
// 			// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
// 			for (int variant = 0; variant < variant_; variant++) {
// 				if ((experiment & (s_iter >> (variant * curr_subj_))) != experiment) {
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
// 	for (experiment = 0; experiment < (1 << curr_subj_); experiment++) {
// 		// reset partition_mass
// 		for (int i = 0; i < (1 << variant_); i++)
// 			partition_mass[i] = 0.0;
// 		// tricky: for each state, check each variant of actively
// 		// pooled subjects to see whether they are all 1.
// 		for (s_iter = 0; s_iter < (1 << (curr_subj_ * variant_)); s_iter++) {
// 			// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
// 			for (int variant = 0; variant < variant_; variant++) {
// 				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
// 				// evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way, 
// 				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the 
// 				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0 
// 				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
// 				partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * curr_subj_))) - experiment) >> 31));
				
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
	double partition_mass[(1 << curr_subj_) * (1 << variant_)]{0.0};
	
	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	for (s_iter = 0; s_iter < (1 << (curr_subj_ * variant_)); s_iter++) {
		// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
		for (experiment = 0; experiment < (1 << curr_subj_); experiment++) {
			for (int variant = 0; variant < variant_; variant++) {
				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
				// evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way, 
				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the 
				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0 
				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
				partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * curr_subj_))) - experiment) >> 31));
			
			}
			partition_mass[experiment * (1 << variant_) + partition_id] += post_probs_[s_iter];
			partition_id = 0;
		}
	}
	double temp = 0.0;
	for (experiment = 0; experiment < (1 << curr_subj_); experiment++) {
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
	double partition_mass[(1 << curr_subj_) * (1 << variant_)]{0.0};
	
	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	#pragma omp parallel for schedule(static) reduction (+ : partition_mass)
	for (int s_iter = 0; s_iter < (1 << (curr_subj_ * variant_)); s_iter++) {
		// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
		for (int experiment = 0; experiment < (1 << curr_subj_); experiment++) {
			int partition_id = 0;
			for (int variant = 0; variant < variant_; variant++) {
				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
				//evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way, 
				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the 
				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0 
				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
				partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * curr_subj_))) - experiment) >> 31));
			
			}
			partition_mass[experiment* (1 << variant_) + partition_id] += post_probs_[s_iter];
		}
	}
	
	Halving_res halving_res;
	#pragma omp declare reduction(Halving_Min : Halving_res : halving_min(omp_out, omp_in)) initializer (omp_priv=Halving_res())
	#pragma omp parallel for schedule(static) reduction (Halving_Min : halving_res)	
	for (int experiment = 0; experiment < (1 << curr_subj_); experiment++) {
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
//     int start_experiment = (1 << curr_subj_) / world_size * rank;
//     int stop_experiment = (1 << curr_subj_) / world_size * (rank+1); 

// 	for (int experiment = start_experiment; experiment < stop_experiment; experiment++) {
// 		// reset partition_mass
// 		for (int i = 0; i < (1 << variant_); i++)
// 			partition_mass[i] = 0.0;
// 		// tricky: for each state, check each variant of actively
// 		// pooled subjects to see whether they are all 1.
// 		for (int s_iter = 0; s_iter < (1 << (curr_subj_ * variant_)); s_iter++) {
// 			// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
// 			for (int variant = 0; variant < variant_; variant++) {
// 				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
// 				//evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way, 
// 				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the 
// 				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0 
// 				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
// 				partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * curr_subj_))) - experiment) >> 31));
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
    const int start_experiment = (1 << curr_subj_) / world_size * rank;
    const int stop_experiment = (1 << curr_subj_) / world_size * (rank+1); 
	double partition_mass[(stop_experiment - start_experiment) * (1 << variant_)]{0.0};

	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	for (int s_iter = 0; s_iter < (1 << (curr_subj_ * variant_)); s_iter++) {
		// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
		for (int experiment = start_experiment; experiment < stop_experiment; experiment++) {
			for (int variant = 0; variant < variant_; variant++) {
				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
				//evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way, 
				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the 
				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0 
				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
				partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * curr_subj_))) - experiment) >> 31));
			
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
	halving_res.reset();
    const int start_experiment = (1 << curr_subj_) / world_size * rank;
    const int stop_experiment = (1 << curr_subj_) / world_size * (rank+1);
	double partition_mass[(stop_experiment - start_experiment) * (1 << variant_)]{0.0}; 

	#pragma omp parallel for schedule(static) reduction (+ : partition_mass)
	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	for (int s_iter = 0; s_iter < (1 << (curr_subj_ * variant_)); s_iter++) {
		int partition_id = 0;
		// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
		for (int experiment = start_experiment; experiment < stop_experiment; experiment++) {
			for (int variant = 0; variant < variant_; variant++) {
				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
				//evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way, 
				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the 
				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0 
				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
				partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * curr_subj_))) - experiment) >> 31));
			
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
	double** ret = new double*[curr_subj_];
	int k;
	for (int rk = 1; rk <= curr_subj_; rk++) {
		ret[rk - 1] = new double[rk + 1];
		ret[rk - 1][0] = alpha;
		for (int r = 1; r <= rk; r++) {
			k = rk - r;
			ret[rk - 1][r] = 1 - alpha * r / (k * h + r);
		}
	}
	return ret;
}