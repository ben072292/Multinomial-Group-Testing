#include "product_lattice.hpp"

int Product_lattice::rank;
int Product_lattice::world_size;
int Product_lattice::_orig_subjs;
int Product_lattice::_variants;
static MPI_Datatype halving_res_type;
static MPI_Op halving_op;

Product_lattice::Product_lattice(int subjs, int variants, double *pi0)
{
	_curr_subjs = subjs;
	_orig_subjs = subjs;
	_variants = variants;
	_parallelism = DATA_PARALLELISM;
	_post_probs = new double[(1 << (_curr_subjs * _variants))];
	prior_probs(pi0);
}

Product_lattice::Product_lattice(const Product_lattice &other, int copy_op)
{
	_parallelism = other._parallelism;
	_curr_subjs = other._curr_subjs;
	_pos_clas_atoms = other._pos_clas_atoms;
	_neg_clas_atoms = other._neg_clas_atoms;
	_test_ct = other._test_ct;
	_clas_subjs = other._clas_subjs;
	if (copy_op == SHALLOW_COPY_PROB_DIST)
	{
		posterior_probs(other._post_probs);
	}
	else if (copy_op == DEEP_COPY_PROB_DIST)
	{
		posterior_probs(other._post_probs);
	}
	else if (copy_op == NO_COPY_PROB_DIST)
	{
		_post_probs = nullptr;
	}
}

Product_lattice::~Product_lattice()
{
	if (_post_probs != nullptr)
		delete[] _post_probs;
}

double Product_lattice::posterior_prob(bin_enc state) const
{
	return _post_probs[state];
}

bin_enc *Product_lattice::get_up_set(bin_enc state, int *ret) const
{
	int index_len = curr_atoms() - __builtin_popcount(state);
	bin_enc *add_index = new bin_enc[1 << index_len];
	int counter = 0;
	for (int i = 0; i < curr_atoms(); i++)
	{
		if ((state & (1 << i)) == 0)
		{
			add_index[counter++] = (1 << i);
		}
	}
	generate_power_set_adder(add_index, index_len, state, ret);
	delete[] add_index;
	return ret;
}

void Product_lattice::generate_power_set_adder(bin_enc *add_index, int index_len, bin_enc state, bin_enc *ret) const
{
	int pow_set_size = 1 << index_len;
	int i, j, temp;
	for (i = 0; i < pow_set_size; i++)
	{
		temp = state;
		for (j = 0; j < index_len; j++)
		{
			/*
			 * Check if j-th bit in the counter is set If set then print j-th element from
			 * set
			 */
			if ((i & (1 << j)))
			{
				temp |= add_index[j];
			}
		}
		ret[i] = temp;
	}
}

void Product_lattice::prior_probs(double *pi0)
{
	int index = total_state();
	for (int i = 0; i < index; i++)
	{
		_post_probs[i] = prior_prob(i, pi0);
	}
}

double Product_lattice::prior_prob(bin_enc state, double *pi0) const
{
	double prob = 1.0;
	for (int i = 0; i < curr_atoms(); i++)
	{
		if ((state & (1 << i)) == 0)
			prob *= pi0[i];
		else
			prob *= (1.0 - pi0[i]);
	}
	return prob;
}

void Product_lattice::update_probs(bin_enc experiment, bin_enc response, double **dilution)
{
	_post_probs = calc_probs(experiment, response, dilution);
	_test_ct++;
}

void Product_lattice::update_probs_in_place(bin_enc experiment, bin_enc response, double **dilution)
{
	calc_probs_in_place(experiment, response, dilution);
	_test_ct++;
}

void Product_lattice::update_metadata(double thres_up, double thres_lo)
{
	// #pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < curr_atoms(); i++)
	{
		bin_enc placement = (1 << i);
		if ((_pos_clas_atoms | _neg_clas_atoms) & placement)
			continue; // skip checking since it's already classified as either positive or negative
		bin_enc atom = 1 << i;
		double prob_mass = get_atom_prob_mass(atom);

		// #pragma omp critical // nominal_pool_size is small so omit contention overhead
		{
			if (prob_mass < thres_lo)
				_pos_clas_atoms |= placement; // classified as positive
			else if (prob_mass > (1 - thres_up))
				_neg_clas_atoms |= placement; // classified as negative
		}
	}
}

void Product_lattice::update_metadata_with_shrinking(double thres_up, double thres_lo)
{
	bin_enc clas_atoms = (_pos_clas_atoms | _neg_clas_atoms); // same size as orig layout
	bin_enc curr_clas_atoms = 0;							  // same size as curr layout
	bin_enc new_curr_clas_atoms = 0;
	bin_enc new_pos_clas_atoms = 0;
	bin_enc new_neg_clas_atoms = 0;

	// #pragma omp parallel for schedule(dynamic) reduction(+ : new_curr_clas_atoms, new_pos_clas_atoms, new_neg_clas_atoms)
	for (int i = 0; i < _orig_subjs * _variants; i++)
	{
		bin_enc orig_index = (1 << i);													// binary index in decimal for original layout
		bin_enc curr_index = orig_curr_ind_conv(i, _clas_subjs, _orig_subjs, _variants); // binary index in decimal for current layout
		if ((clas_atoms & orig_index))
		{
			new_curr_clas_atoms |= curr_index;
		}
		else
		{
			double prob_mass = get_atom_prob_mass(curr_index);
			if (prob_mass < thres_lo)
			{ // classified as positive
				new_curr_clas_atoms |= curr_index;
				new_pos_clas_atoms |= orig_index;
			}
			else if (prob_mass > (1 - thres_up))
			{ // classified as positive
				new_curr_clas_atoms |= curr_index;
				new_neg_clas_atoms |= orig_index;
			}
		}
	}
	curr_clas_atoms = new_curr_clas_atoms;
	_pos_clas_atoms |= new_pos_clas_atoms;
	_neg_clas_atoms |= new_neg_clas_atoms;

	curr_clas_atoms = curr_shrinkable_atoms(curr_clas_atoms, _curr_subjs, _variants);
	if (curr_clas_atoms)
		shrinking(curr_clas_atoms); // if there's new classifications, we perform the actual shrinkings
}

void Product_lattice::shrinking(int curr_clas_atoms)
{	
	int reduce_count = __builtin_popcount(curr_clas_atoms);
	int base_count = curr_atoms() - reduce_count;
	bin_enc *base_index = new bin_enc[base_count];
	bin_enc *reduce_index = new bin_enc[reduce_count];
	base_count = 0;
	reduce_count = 0;
	for (int i = 0; i < curr_atoms(); i++)
	{
		if ((curr_clas_atoms & (1 << i)) != 0)
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
	bin_enc ele_base = 0, ele_reduce = 0;
	for (int i = 0; i < (1 << base_count); i++)
	{
		ele_base = 0;
		for (int j = 0; j < base_count; j++)
		{
			if ((i & (1 << j)))
			{
				ele_base |= base_index[j];
			}
		}
		for (int k = 0; k < (1 << reduce_count); k++)
		{
			ele_reduce = 0;
			for (int l = 0; l < reduce_count; l++)
			{
				if ((k & (1 << l)))
				{
					ele_reduce |= reduce_index[l];
				}
			}
			if (i != ele_base + ele_reduce)
				_post_probs[i] += _post_probs[ele_base + ele_reduce]; // in-place modfication because each state is uniquely merged
		}
	}
	delete[] base_index;
	delete[] reduce_index;
	_clas_subjs = update_clas_subj(_pos_clas_atoms | _neg_clas_atoms, _orig_subjs, _variants);
	_curr_subjs = _orig_subjs - __builtin_popcount(_clas_subjs);
}

// Active generation
double Product_lattice::get_prob_mass(bin_enc state) const
{
	double ret = 0.0;
	int n = curr_atoms() - __builtin_popcount(state), pow_set_size = 1 << n;
	bin_enc temp;
	bin_enc *add_index = new bin_enc[n];
	int counter = 0;
	bin_enc index;
	for (int i = 0; i < curr_atoms(); i++)
	{
		index = (1 << i);
		if (!(state & index))
		{
			add_index[counter++] = index;
		}
	}

	for (int i = 0; i < pow_set_size; i++)
	{
		temp = state;
		for (int j = 0; j < n; j++)
		{
			if ((i & (1 << j)))
			{
				temp |= add_index[j];
			}
		}
		ret += _post_probs[temp];
	}

	delete[] add_index;
	return ret;
}

// Exhaustive traversal is faster than active generation for atoms
double Product_lattice::get_atom_prob_mass(bin_enc atom) const
{
	double ret = 0.0;
	// #pragma omp parallel for reduction (+ : ret)
	for (int i = 0; i < (1 << curr_atoms()); i++)
	{
		if ((i & atom) == atom)
		{
			ret += _post_probs[i];
		}
	}
	return ret;
}

/**
 * experiment is an integer with range (0, 2^N)
 * response is an integer with range [0, 2^k)
 */
double *Product_lattice::calc_probs(bin_enc experiment, bin_enc response, double **dilution)
{
	bin_enc total_state = 1 << (_curr_subjs * _variants);
	double *ret = new double[total_state];
	double denominator = 0.0;
	// #pragma omp parallel for reduction (+ : denominator) // INVESTIGATE: slowdown than serial
	for (bin_enc i = 0; i < total_state; i++)
	{
		ret[i] = _post_probs[i] * response_prob(experiment, response, i, dilution);
		denominator += ret[i];
	}
	double denominator_inv = 1 / denominator; // division has much higher instruction latency and throughput than multiplication (however latency is not that important since we have many independent divison operations)
	// #pragma omp parallel for
	for (bin_enc i = 0; i < total_state; i++)
	{
		ret[i] *= denominator_inv;
	}
	return ret;
}

void Product_lattice::calc_probs_in_place(bin_enc experiment, bin_enc response, double **__restrict__ dilution)
{
	double denominator = 0.0;
	bin_enc total_state = 1 << (_curr_subjs * _variants);
	// #pragma omp parallel for schedule(static) reduction (+ : denominator)
	for (bin_enc i = 0; i < total_state; i++)
	{
		_post_probs[i] *= response_prob(experiment, response, i, dilution);
		denominator += _post_probs[i];
	}
	// #pragma omp parallel for schedule(static)
	for (bin_enc i = 0; i < total_state; i++)
	{
		_post_probs[i] /= denominator;
	}
}

/**
 * Implementation V2
 */
// int Product_lattice::halving_serial(double prob) const{
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

/**
 * Implementation V3
 */
// int Product_lattice::halving_serial(double prob) const{
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

bin_enc Product_lattice::halving_serial(double prob) const
{
	bin_enc candidate = 0;
	double min = 2.0;
	int partition_id = 0;
	double partition_mass[(1 << _curr_subjs) * (1 << _variants)]{0.0};

	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	for (bin_enc s_iter = 0; s_iter < (1 << (_curr_subjs * _variants)); s_iter++)
	{
		// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
		for (bin_enc experiment = 0; experiment < (1 << _curr_subjs); experiment++)
		{
			for (int variant = 0; variant < _variants; variant++)
			{
				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
				// evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way,
				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the
				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0
				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
				partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * _curr_subjs))) - experiment) >> 31));
			}
			partition_mass[experiment * (1 << _variants) + partition_id] += _post_probs[s_iter];
			partition_id = 0;
		}
	}
	double temp = 0.0;
	for (bin_enc experiment = 0; experiment < (1 << _curr_subjs); experiment++)
	{
		for (bin_enc i = 0; i < (1 << _variants); i++)
		{
			temp += std::abs(partition_mass[experiment * (1 << _variants) + i] - prob);
		}
		if (temp < min)
		{
			min = temp;
			candidate = experiment;
		}
		temp = 0.0;
	}

	return candidate;
}

bin_enc Product_lattice::halving_omp(double prob) const
{
	double partition_mass[(1 << _curr_subjs) * (1 << _variants)]{0.0};

	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	#pragma omp parallel for schedule(static) reduction(+ : partition_mass)
	for (bin_enc s_iter = 0; s_iter < (1 << (_curr_subjs * _variants)); s_iter++)
	{
		// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
		for (bin_enc experiment = 0; experiment < (1 << _curr_subjs); experiment++)
		{
			int partition_id = 0;
			for (int variant = 0; variant < _variants; variant++)
			{
				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
				// evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way,
				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the
				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0
				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
				partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * _curr_subjs))) - experiment) >> 31));
			}
			partition_mass[experiment * (1 << _variants) + partition_id] += _post_probs[s_iter];
		}
	}

	Halving_res halving_res;
	#pragma omp declare reduction(Halving_Min:Halving_res : Halving_res::halving_min(omp_out, omp_in)) initializer(omp_priv = Halving_res())
	#pragma omp parallel for schedule(static) reduction(Halving_Min : halving_res)
	for (bin_enc experiment = 0; experiment < (1 << _curr_subjs); experiment++)
	{
		double temp = 0.0;
		for (int i = 0; i < (1 << _variants); i++)
		{
			temp += std::abs(partition_mass[experiment * (1 << _variants) + i] - prob);
		}
		if (temp < halving_res.min)
		{
			halving_res.min = temp;
			halving_res.candidate = experiment;
		}
	}

	return halving_res.candidate;
}

bin_enc Product_lattice::halving_mpi(double prob) const
{
	Halving_res halving_res;
	int partition_id = 0;
	const bin_enc start_experiment = (1 << _curr_subjs) / world_size * rank;
	const bin_enc stop_experiment = (1 << _curr_subjs) / world_size * (rank + 1);
	double partition_mass[(stop_experiment - start_experiment) * (1 << _variants)]{0.0};

	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	for (bin_enc s_iter = 0; s_iter < (1 << (_curr_subjs * _variants)); s_iter++)
	{
		// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
		for (bin_enc experiment = start_experiment; experiment < stop_experiment; experiment++)
		{
			for (int variant = 0; variant < _variants; variant++)
			{
				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
				// evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way,
				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the
				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0
				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
				partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * _curr_subjs))) - experiment) >> 31));
			}
			partition_mass[(experiment - start_experiment) * (1 << _variants) + partition_id] += _post_probs[s_iter];
			partition_id = 0;
		}
	}
	double temp = 0.0;
	for (bin_enc experiment = start_experiment; experiment < stop_experiment; experiment++)
	{
		for (bin_enc i = 0; i < (1 << _variants); i++)
		{
			temp += std::abs(partition_mass[(experiment - start_experiment) * (1 << _variants) + i] - prob);
		}
		if (temp < halving_res.min)
		{
			halving_res.min = temp;
			halving_res.candidate = experiment;
		}
		temp = 0.0;
	}
	MPI_Allreduce(MPI_IN_PLACE, &halving_res, 2, halving_res_type, halving_op, MPI_COMM_WORLD);

	return halving_res.candidate;
}

bin_enc Product_lattice::halving_mpi_vectorize(double prob) const { throw std::logic_error("Not Implemented."); }

bin_enc Product_lattice::halving_hybrid(double prob) const
{
	Halving_res halving_res;
	const bin_enc start_experiment = (1 << _curr_subjs) / world_size * rank;
	const bin_enc stop_experiment = (1 << _curr_subjs) / world_size * (rank + 1);
	double partition_mass[(stop_experiment - start_experiment) * (1 << _variants)]{0.0};

	#pragma omp parallel for schedule(static) reduction(+ : partition_mass)
	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	for (bin_enc s_iter = 0; s_iter < (1 << (_curr_subjs * _variants)); s_iter++)
	{
		int partition_id = 0;
		// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
		for (bin_enc experiment = start_experiment; experiment < stop_experiment; experiment++)
		{
			for (int variant = 0; variant < _variants; variant++)
			{
				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
				// evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way,
				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the
				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0
				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
				partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * _curr_subjs))) - experiment) >> 31));
			}
			partition_mass[(experiment - start_experiment) * (1 << _variants) + partition_id] += _post_probs[s_iter];
			partition_id = 0;
		}
	}

	#pragma omp declare reduction(Halving_Min:Halving_res : Halving_res::halving_min(omp_out, omp_in)) initializer(omp_priv = Halving_res())
	#pragma omp parallel for schedule(static) reduction(Halving_Min : halving_res)
	for (bin_enc experiment = start_experiment; experiment < stop_experiment; experiment++)
	{
		double temp = 0.0;
		for (bin_enc i = 0; i < (1 << _variants); i++)
		{
			temp += std::abs(partition_mass[(experiment - start_experiment) * (1 << _variants) + i] - prob);
		}
		if (temp < halving_res.min)
		{
			halving_res.min = temp;
			halving_res.candidate = experiment;
		}
		temp = 0.0;
	}
	MPI_Allreduce(MPI_IN_PLACE, &halving_res, 2, halving_res_type, halving_op, MPI_COMM_WORLD);
	return halving_res.candidate;
}

bin_enc Product_lattice::halving(double prob) const
{
	if ((1 << curr_atoms()) >= world_size)
		return halving_hybrid(prob);
	else
		return halving_omp(prob);
}

double **Product_lattice::generate_dilution(double alpha, double h) const
{
	double **ret = new double *[_curr_subjs];
	int k;
	for (int rk = 1; rk <= _curr_subjs; rk++)
	{
		ret[rk - 1] = new double[rk + 1];
		ret[rk - 1][0] = alpha;
		for (int r = 1; r <= rk; r++)
		{
			k = rk - r;
			ret[rk - 1][r] = 1 - alpha * r / (k * h + r);
		}
	}
	return ret;
}

// Assign rank and world size as static member variable,
// initialize MPI datatypes and collective ops for product lattice
void Product_lattice::MPI_Product_lattice_Initialize()
{
	// Get the number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	// Get the rank of the process
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	Halving_res::create_halving_res_type(&halving_res_type);
	MPI_Type_commit(&halving_res_type);
	MPI_Op_create((MPI_User_function *)&Halving_res::halving_reduce, true, &halving_op);
}

// free MPI datatypes and collective ops for product lattice
void Product_lattice::MPI_Product_lattice_Free()
{
	// Free datatype
	MPI_Type_free(&halving_res_type);
	// Free reduce op
	MPI_Op_free(&halving_op);
	// Finalize the MPI environment.
}