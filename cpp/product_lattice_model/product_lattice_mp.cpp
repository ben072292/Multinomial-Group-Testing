#include "product_lattice_mp.hpp"

Product_lattice_mp::Product_lattice_mp(int subjs, int variants, double* pi0){
	_parallelism = MODEL_PARALLELISM;
    _curr_subjs = subjs;
    _variants = variants;
    _post_probs = new double[total_state_each()];
	prior_probs(pi0);
}

Product_lattice_mp::~Product_lattice_mp(){
	if(_post_probs != nullptr) delete[] _post_probs;
}

// For debugging purpose
double Product_lattice_mp::posterior_prob(bin_enc state) const {
	double ret = -1.0;
	int target_rank = state_to_rank(state);
	int target_offset = state_to_offset(state);
	if(rank == target_rank) ret = _post_probs[target_offset];
	MPI_Allreduce(MPI_IN_PLACE, &ret, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	return ret;
}

void Product_lattice_mp::prior_probs(double* pi0){
	for (int i = 0; i < total_state_each(); i++) {
		_post_probs[i] = prior_prob(offset_to_state(i), pi0);
	}
}

void Product_lattice_mp::update_metadata(double thres_up, double thres_lo){
    for (int i = 0; i < curr_atoms(); i++) {
		bin_enc placement = (1 << i);
		if ((_pos_clas_atoms | _neg_clas_atoms) & placement)
			continue; // skip checking since it's already classified as either positive or negative
		bin_enc atom = 1 << i;
		double prob_mass = get_atom_prob_mass(atom);
		if (prob_mass < thres_lo)
			_pos_clas_atoms |= placement; // classified as positive
		else if (prob_mass > (1 - thres_up))
			_neg_clas_atoms |= placement; // classified as negative
	}
}

// not modified yet
void Product_lattice_mp::update_metadata_with_shrinking(double thres_up, double thres_lo){
	bin_enc clas_atoms = (_pos_clas_atoms | _neg_clas_atoms); // same size as orig layout
	int orig_subjs = this->orig_subjs(); // called at the beginning to ensure correct value
	bin_enc curr_clas_atoms = 0; // same size as curr layout
	int curr_atoms = _curr_subjs * _variants;
	bin_enc new_curr_clas_atoms = 0;
	bin_enc new_pos_clas_atoms = 0;
	bin_enc new_neg_clas_atoms = 0;

    for (int i = 0; i < orig_subjs * _variants; i++) {
		bin_enc orig_index = (1 << i); // binary index in decimal for original layout
		bin_enc curr_index = orig_curr_ind_conv(i, _clas_subjs, orig_subjs, _variants); // binary index in decimal for current layout
		if((clas_atoms & orig_index)){
			new_curr_clas_atoms |= curr_index;
		}
		else{
			double prob_mass = get_atom_prob_mass(curr_index);
			if (prob_mass < thres_lo){ // classified as positive
				new_curr_clas_atoms |= curr_index;
				new_pos_clas_atoms |= orig_index;
			}
			else if (prob_mass > (1 - thres_up)){ // classified as positive
				new_curr_clas_atoms |= curr_index;
				new_neg_clas_atoms |= orig_index;
			}
		}
	}
	curr_clas_atoms = new_curr_clas_atoms;
	_pos_clas_atoms |= new_pos_clas_atoms;
	_neg_clas_atoms |= new_neg_clas_atoms;

	curr_clas_atoms = curr_shrinkable_atoms(curr_clas_atoms, _curr_subjs, _variants);

	int target_prob_size = (1 << ((orig_subjs - __builtin_popcount(update_clas_subj(_pos_clas_atoms | _neg_clas_atoms, orig_subjs, _variants))) * _variants));
	// if data parallelism is achievable, i.e., each process has at least 1 state to work, performing model parallelism shrinking
	if(curr_clas_atoms && target_prob_size / world_size > 0) shrinking(orig_subjs, curr_atoms, curr_clas_atoms);
	// if model parallelism is not achievable, covert model parallelism to data parallelism and then shrinking
	else if(curr_clas_atoms && target_prob_size / world_size == 0){
		double* candidate_post_probs = new double[(1 << curr_atoms)]{0.0};
		MPI_Allgather(_post_probs, total_state_each(), MPI_DOUBLE, candidate_post_probs, total_state_each(), MPI_DOUBLE, MPI_COMM_WORLD);
		delete[] _post_probs;
		_post_probs = candidate_post_probs;
		_parallelism = DATA_PARALLELISM;
		Product_lattice::shrinking(orig_subjs, curr_atoms, curr_clas_atoms);
	}
}

void Product_lattice_mp::shrinking(int orig_subjs, int curr_atoms, int curr_clas_atoms){
	int reduce_count = __builtin_popcount(curr_clas_atoms);
	int base_count = curr_atoms - reduce_count;
	bin_enc* base_index = new bin_enc[base_count];
	bin_enc* reduce_index = new bin_enc[reduce_count];
	base_count = 0;
	reduce_count = 0;
	for(int i = 0; i < curr_atoms; i++){
		if((curr_clas_atoms & (1 << i)) != 0)
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
	double *candidate_post_probs = new double[(1 << base_count) / world_size]{0.0};
	bin_enc ele_base = 0, ele_reduce = 0;
	double rma_val[((1 << base_count) / world_size)][(1 << reduce_count)]{0.0};
	MPI_Win window;
	MPI_Win_create(_post_probs, sizeof(double) * total_state_each(), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &window);
	MPI_Win_fence(MPI_MODE_NOPUT, window);
	for (int i = 0; i < ((1 << base_count) / world_size); i++) {
		ele_base = 0;
		for (int j = 0; j < base_count; j++) {
			// i + (1 << base_count) / world_size * rank is the state's binary encoding for shrinked posterior prob locations
			if (((i + (1 << base_count) / world_size * rank) & (1 << j))) { 
				ele_base |= base_index[j];
			}
		}
		for(int k = 0; k < (1 << reduce_count); k++){
			ele_reduce = 0;
			for(int l = 0; l < reduce_count; l++){
			 	if((k & (1 << l))) {
					ele_reduce |= reduce_index[l];
				}
			}
			// Fetch the value
        	MPI_Get(&rma_val[i][k], 1, MPI_DOUBLE, state_to_rank(ele_base + ele_reduce), state_to_offset(ele_base + ele_reduce), 1, MPI_DOUBLE, window);
		}
	}
	MPI_Win_fence(MPI_MODE_NOPUT, window);
	for (int i = 0; i < ((1 << base_count) / world_size); i++) {
		for(int k = 0; k < (1 << reduce_count); k++){
			candidate_post_probs[i] += rma_val[i][k];
		}
	}
	// Destroy the window
    MPI_Win_free(&window);
	
	delete[] _post_probs;
	delete[] base_index;
	delete[] reduce_index; 
	_post_probs = candidate_post_probs;
	_clas_subjs = update_clas_subj(_pos_clas_atoms | _neg_clas_atoms, orig_subjs, _variants);
	_curr_subjs = orig_subjs - __builtin_popcount(_clas_subjs);
}

// Exhaustive traversal is faster than active generation for atoms
double Product_lattice_mp::get_atom_prob_mass(bin_enc atom) const{
	double ret = 0.0;
	#pragma omp parallel for reduction (+ : ret)
	for(int i = 0; i < total_state_each(); i++){
		if((offset_to_state(i) & atom) == atom){
			ret += _post_probs[i];
		}
	}
    MPI_Allreduce(MPI_IN_PLACE, &ret, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	return ret;
}

/**
 * experiment is an integer with range (0, 2^N)
 * response is an integer with range [0, 2^k)
*/
double* Product_lattice_mp::calc_probs(bin_enc experiment, bin_enc response, double** dilution){
	double* ret = new double[total_state_each()];
	double denominator = 0.0;
	#pragma omp parallel for reduction (+ : denominator) // INVESTIGATE: slowdown than serial
	for (int i = 0; i < total_state_each(); i++) {
		ret[i] = _post_probs[i] * response_prob(experiment, response, offset_to_state(i), dilution);
		denominator += ret[i];
	}
    MPI_Allreduce(MPI_IN_PLACE, &denominator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	double denominator_inv = 1 / denominator; // division has much higher instruction latency and throughput than multiplication (however latency is not that important since we have many independent divison operations)
	#pragma omp parallel for
	for (int i = 0; i < total_state_each(); i++) {
		ret[i] *= denominator_inv;
	}
	return ret;
}

void Product_lattice_mp::calc_probs_in_place(bin_enc experiment, bin_enc response, double **__restrict__ dilution){
	double denominator = 0.0;
	#pragma omp parallel for schedule(static) reduction (+ : denominator)
	for (int i = 0; i < total_state_each(); i++) {
		_post_probs[i] *= response_prob(experiment, response, offset_to_state(i), dilution);
		denominator += _post_probs[i];
	}
    MPI_Allreduce(MPI_IN_PLACE, &denominator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	double denominator_inv = 1 / denominator; // division has much higher instruction latency and throughput than multiplication (however latency is not that important since we have many independent divison operations)
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < total_state_each(); i++) {
		_post_probs[i] *= denominator_inv;
	}
}

bin_enc Product_lattice_mp::halving_mpi(double prob) const{
	int partition_id = 0;
	int partition_size = (1 << _curr_subjs) * (1 << _variants);
	double partition_mass[partition_size]{0.0};

	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	for (int s_iter = 0; s_iter < total_state_each(); s_iter++) {
		// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
		for (bin_enc experiment = 0; experiment < (1 << _curr_subjs); experiment++) {
			for (int variant = 0; variant < _variants; variant++) {
				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
				//evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way, 
				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the 
				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0 
				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
				partition_id |= ((1 << variant) & (((experiment & (offset_to_state(s_iter) >> (variant * _curr_subjs))) - experiment) >> 31));
			
			}
			partition_mass[experiment * (1 << _variants) + partition_id] += _post_probs[s_iter];
			partition_id = 0;
		}
	}
    MPI_Allreduce(MPI_IN_PLACE, partition_mass, partition_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	double temp = 0.0;
    double min = 2.0;
    bin_enc candidate = -1;
	for (bin_enc experiment = 0; experiment < (1 << _curr_subjs); experiment++) {
		for (int i = 0; i < (1 << _variants); i++) {
			temp += std::abs(partition_mass[experiment * (1 << _variants) + i] - prob);
		}
		if (temp < min) {
			min = temp;
			candidate = experiment;
		}
		temp = 0.0;
	}

	return candidate;
}

bin_enc Product_lattice_mp::halving_hybrid(double prob) const{
    Halving_res halving_res;
	int partition_size = (1 << _curr_subjs) * (1 << _variants);
	double partition_mass[partition_size]{0.0};

	#pragma omp parallel for schedule(static) reduction (+ : partition_mass)
	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	for (int s_iter = 0; s_iter < total_state_each(); s_iter++) {
		int partition_id = 0;
		// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
		for (bin_enc experiment = 0; experiment < (1 << _curr_subjs); experiment++) {
			for (int variant = 0; variant < _variants; variant++) {
				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
				//evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way, 
				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the 
				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0 
				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
				partition_id |= ((1 << variant) & (((experiment & (offset_to_state(s_iter) >> (variant * _curr_subjs))) - experiment) >> 31));
			
			}
			partition_mass[experiment * (1 << _variants) + partition_id] += _post_probs[s_iter];
			partition_id = 0;
		}
	}
    MPI_Allreduce(MPI_IN_PLACE, partition_mass, partition_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	#pragma omp declare reduction(Halving_Min : Halving_res : Halving_res::halving_min(omp_out, omp_in)) initializer (omp_priv=Halving_res())
	#pragma omp parallel for schedule(static) reduction (Halving_Min : halving_res)
	for (bin_enc experiment = 0; experiment < (1 << _curr_subjs); experiment++) {
		double temp = 0.0;
		for (bin_enc i = 0; i < (1 << _variants); i++) {
			temp += std::abs(partition_mass[experiment * (1 << _variants) + i] - prob);
		}
		if (temp < halving_res.min) {
			halving_res.min = temp;
			halving_res.candidate = experiment;
		}
		temp = 0.0;
	}
	MPI_Allreduce(MPI_IN_PLACE, &halving_res.candidate, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
	return halving_res.candidate;
}