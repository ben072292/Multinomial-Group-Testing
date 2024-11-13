#include "product_lattice.hpp"

int Product_lattice::rank;
int Product_lattice::world_size;
int Product_lattice::_orig_subjs;
int Product_lattice::_variants;
double *Product_lattice::_pi0;
static MPI_Datatype BBPA_res_type;
static MPI_Op BBPA_op;

Product_lattice::Product_lattice(int subjs, int variants, double *pi0)
{
	_curr_subjs = subjs;
	_orig_subjs = subjs;
	_pi0 = pi0;
	_variants = variants;
	_post_probs = new double[(1 << (_curr_subjs * _variants))];
	prior_probs(pi0);
}

Product_lattice::Product_lattice(const Product_lattice &other, lattice_copy_op_t op)
{
	_curr_subjs = other._curr_subjs;
	_pos_clas_atoms = other._pos_clas_atoms;
	_neg_clas_atoms = other._neg_clas_atoms;
	_clas_subjs = other._clas_subjs;
	if (op == SHALLOW_COPY_PROB_DIST)
	{
		posterior_probs(other._post_probs);
	}
	else if (op == DEEP_COPY_PROB_DIST)
	{
		posterior_probs(other._post_probs);
	}
	else if (op == NO_COPY_PROB_DIST)
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
	int index = total_states();
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
	_post_probs = ret;
	ret = nullptr;
}

void Product_lattice::update_probs_in_place(bin_enc experiment, bin_enc response, double **dilution)
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

bool Product_lattice::update_metadata_with_shrinking(double thres_up, double thres_lo)
{
	bin_enc clas_atoms = (_pos_clas_atoms | _neg_clas_atoms); // same size as orig layout
	bin_enc curr_clas_atoms = 0;							  // same size as curr layout
	bin_enc new_curr_clas_atoms = 0;
	bin_enc new_pos_clas_atoms = 0;
	bin_enc new_neg_clas_atoms = 0;

	// #pragma omp parallel for schedule(dynamic) reduction(+ : new_curr_clas_atoms, new_pos_clas_atoms, new_neg_clas_atoms)
	for (int i = 0; i < _orig_subjs * _variants; i++)
	{
		bin_enc orig_index = (1 << i);													 // binary index in decimal for original layout
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
	return false;					// data parallelism will not covert parallelism so always return false;
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
	// 		shrinked_post_probs[i] += (_post_probs[base_adder[i] + reduce_adder[j]]);
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

#ifdef BBPA_NAIVE
/**
 * Implementation V1
 */
int Product_lattice::BBPA_serial(double prob) const
{
	int candidate = 0;
	int s_iter;
	int experiment;
	bool is_complement = false;
	double min = 2.0;
	int partition_id = 0;
	double partition_mass[(1 << _variants)];
	for (experiment = 0; experiment < (1 << _curr_subjs); experiment++)
	{
		// reset partition_mass
		for (int i = 0; i < (1 << _variants); i++)
			partition_mass[i] = 0.0;
		// tricky: for each state, check each variant of actively
		// pooled subjects to see whether they are all 1.
		for (s_iter = 0; s_iter < (1 << (_curr_subjs * _variants)); s_iter++)
		{
			for (int variant = 0; variant < _variants; variant++)
			{
				for (int l = 0; l < _curr_subjs; l++)
				{
					if ((experiment & (1 << l)) != 0 && (s_iter & (1 << (l * _variants + variant))) == 0)
					{
						is_complement = true;
						break;
					}
				}
				partition_id |= (is_complement ? 0 : (1 << variant));
				is_complement = false; // reset flag
			}
			partition_mass[partition_id] += _post_probs[s_iter];
			partition_id = 0;
		}
		// for (int i = 0; i < totalStates(); i++) {
		// System.out.print(partitionMap[i] + " ");
		// }
		// System.out.println();
		double temp = 0.0;
		for (int i = 0; i < (1 << _variants); i++)
		{
			temp += std::abs(partition_mass[i] - prob);
		}
		if (temp < min)
		{
			min = temp;
			candidate = experiment;
		}
	}
	return candidate;
}
#elif defined(BBPA_OP1)
/**
 * Implementation V2
 */
int Product_lattice::BBPA_serial(double prob) const
{
	int candidate = 0;
	int s_iter;
	int experiment;
	double min = 2.0;
	int partition_id = 0;
	double partition_mass[(1 << _variants)];
	for (experiment = 0; experiment < (1 << _curr_subjs); experiment++)
	{
		// reset partition_mass
		for (int i = 0; i < (1 << _variants); i++)
			partition_mass[i] = 0.0;
		// tricky: for each state, check each variant of actively
		// pooled subjects to see whether they are all 1.
		for (s_iter = 0; s_iter < (1 << (_curr_subjs * _variants)); s_iter++)
		{
			// __builtin_prefetch((_post_probs + s_iter + 10), 0, 0);
			for (int variant = 0; variant < _variants; variant++)
			{
				if ((experiment & (s_iter >> (variant * _curr_subjs))) != experiment)
				{
					partition_id |= (1 << variant);
				}
			}
			partition_mass[partition_id] += _post_probs[s_iter];
			partition_id = 0;
		}
		double temp = 0.0;
		for (int i = 0; i < (1 << _variants); i++)
		{
			temp += std::abs(partition_mass[i] - prob);
		}
		if (temp < min)
		{
			min = temp;
			candidate = experiment;
		}
	}
	return candidate;
}

#elif defined(BBPA_OP2)
/**
 * Implementation V3
 */
int Product_lattice::BBPA_serial(double prob) const
{
	int candidate = 0;
	int s_iter;
	int experiment;
	double min = 2.0;
	int partition_id = 0;
	double partition_mass[(1 << _variants)];
	for (experiment = 0; experiment < (1 << _curr_subjs); experiment++)
	{
		// reset partition_mass
		for (int i = 0; i < (1 << _variants); i++)
			partition_mass[i] = 0.0;
		// tricky: for each state, check each variant of actively
		// pooled subjects to see whether they are all 1.
		for (s_iter = 0; s_iter < (1 << (_curr_subjs * _variants)); s_iter++)
		{
			// __builtin_prefetch((_post_probs + s_iter + 20), 0, 0);
			for (int variant = 0; variant < _variants; variant++)
			{
				// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
				// evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way,
				// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the
				// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0
				// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
				partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * _curr_subjs))) - experiment) >> 31));
			}
			partition_mass[partition_id] += _post_probs[s_iter];
			partition_id = 0;
		}
		double temp = 0.0;
		for (int i = 0; i < (1 << _variants); i++)
		{
			temp += std::abs(partition_mass[i] - prob);
		}
		if (temp < min)
		{
			min = temp;
			candidate = experiment;
		}
	}
	return candidate;
}
#else
bin_enc Product_lattice::BBPA_serial(double prob) const
{
	bin_enc candidate = 0;
	double min = 2.0;
	int partition_id = 0;
	double partition_mass[(1 << _curr_subjs) * (1 << _variants)]{0.0};

	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	for (bin_enc s_iter = 0; s_iter < (1 << (_curr_subjs * _variants)); s_iter++)
	{
		// __builtin_prefetch((_post_probs + s_iter + 20), 0, 0);
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
#endif

#ifdef ENABLE_OMP
bin_enc Product_lattice::BBPA_omp(double prob) const
{
	double partition_mass[(1 << _curr_subjs) * (1 << _variants)]{0.0};

// tricky: for each state, check each variant of actively
// pooled subjects to see whether they are all 1.
#pragma omp parallel for schedule(static) reduction(+ : partition_mass)
	for (bin_enc s_iter = 0; s_iter < (1 << (_curr_subjs * _variants)); s_iter++)
	{
		// __builtin_prefetch((_post_probs + s_iter + 20), 0, 0);
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

	BBPA_res res;
#pragma omp declare reduction(BBPA_Min:BBPA_res : BBPA_res::BBPA_min(omp_out, omp_in)) initializer(omp_priv = BBPA_res())
#pragma omp parallel for schedule(static) reduction(BBPA_Min : res)
	for (bin_enc experiment = 0; experiment < (1 << _curr_subjs); experiment++)
	{
		double temp = 0.0;
		for (int i = 0; i < (1 << _variants); i++)
		{
			temp += std::abs(partition_mass[experiment * (1 << _variants) + i] - prob);
		}
		if (temp < res.min)
		{
			res.min = temp;
			res.candidate = experiment;
		}
	}

	return res.candidate;
}
#endif

bin_enc Product_lattice::BBPA_mpi(double prob) const
{
	BBPA_res BBPA_res;
	int partition_id = 0;
	const bin_enc start_experiment = (1 << _curr_subjs) / world_size * rank;
	const bin_enc stop_experiment = (1 << _curr_subjs) / world_size * (rank + 1);
	double partition_mass[(stop_experiment - start_experiment) * (1 << _variants)]{0.0};

	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	for (bin_enc s_iter = 0; s_iter < (1 << (_curr_subjs * _variants)); s_iter++)
	{
		// __builtin_prefetch((_post_probs + s_iter + 20), 0, 0);
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
		if (temp < BBPA_res.min)
		{
			BBPA_res.min = temp;
			BBPA_res.candidate = experiment;
		}
		temp = 0.0;
	}
	MPI_Allreduce(MPI_IN_PLACE, &BBPA_res, 1, BBPA_res_type, BBPA_op, MPI_COMM_WORLD);

	return BBPA_res.candidate;
}

#ifdef ENABLE_OMP
bin_enc Product_lattice::BBPA_mpi_omp(double prob) const
{
	BBPA_res res;
	const bin_enc start_experiment = (1 << _curr_subjs) / world_size * rank;
	const bin_enc stop_experiment = (1 << _curr_subjs) / world_size * (rank + 1);
	double partition_mass[(stop_experiment - start_experiment) * (1 << _variants)]{0.0};

#pragma omp parallel for schedule(static) reduction(+ : partition_mass)
	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	for (bin_enc s_iter = 0; s_iter < (1 << (_curr_subjs * _variants)); s_iter++)
	{
		int partition_id = 0;
		// __builtin_prefetch((_post_probs + s_iter + 20), 0, 0);
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

#pragma omp declare reduction(BBPA_Min:BBPA_res : BBPA_res::BBPA_min(omp_out, omp_in)) initializer(omp_priv = BBPA_res())
#pragma omp parallel for schedule(static) reduction(BBPA_Min : res)
	for (bin_enc experiment = start_experiment; experiment < stop_experiment; experiment++)
	{
		double temp = 0.0;
		for (bin_enc i = 0; i < (1 << _variants); i++)
		{
			temp += std::abs(partition_mass[(experiment - start_experiment) * (1 << _variants) + i] - prob);
		}
		if (temp < res.min)
		{
			res.min = temp;
			res.candidate = experiment;
		}
		temp = 0.0;
	}
	MPI_Allreduce(MPI_IN_PLACE, &res, 1, BBPA_res_type, BBPA_op, MPI_COMM_WORLD);
	return res.candidate;
}
#endif

bin_enc Product_lattice::BBPA(double prob) const
{
#ifdef ENABLE_OMP
	return BBPA_omp(prob);
#else
	return BBPA_serial(prob);
#endif
}

// Assign rank and world size as static member variable,
// initialize MPI datatypes and collective ops for product lattice
void Product_lattice::MPI_Product_lattice_Initialize()
{
	// Get the number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	// Get the rank of the process
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	BBPA_res::create_BBPA_res_type(&BBPA_res_type);
	MPI_Type_commit(&BBPA_res_type);
	MPI_Op_create((MPI_User_function *)&BBPA_res::BBPA_reduce, true, &BBPA_op);
}

// free MPI datatypes and collective ops for product lattice
void Product_lattice::MPI_Product_lattice_Finalize()
{
	// Free datatype
	MPI_Type_free(&BBPA_res_type);
	// Free reduce op
	MPI_Op_free(&BBPA_op);
	// Finalize the MPI environment.
}

EXPORT Product_lattice_t create_lattice(lattice_type_t type, int subjs, int variants, double *pi0)
{
	switch (type)
	{
	case DIST_NON_DILUTION:
		return new Product_lattice_dist_non_dilution(subjs, variants, pi0);
		break;
	case DIST_DILUTION:
		return new Product_lattice_dist_dilution(subjs, variants, pi0);
		break;
	case REPL_NON_DILUTION:
		return new Product_lattice_non_dilution(subjs, variants, pi0);
		break;
	case REPL_DILUTION:
		return new Product_lattice_dilution(subjs, variants, pi0);
		break;
	default:
		throw std::logic_error("Nonexisting product lattice type! Exiting...");
		exit(1);
	}
}

EXPORT Product_lattice_t clone_lattice(lattice_type_t type, lattice_copy_op_t op, Product_lattice_t lattice)
{
	switch (type)
	{
	case DIST_NON_DILUTION:
		return new Product_lattice_dist_non_dilution(*lattice, op);
		break;
	case DIST_DILUTION:
		return new Product_lattice_dist_dilution(*lattice, op);
		break;
	case REPL_NON_DILUTION:
		return new Product_lattice_non_dilution(*lattice, op);
		break;
	case REPL_DILUTION:
		return new Product_lattice_dilution(*lattice, op);
		break;
	default:
		throw std::logic_error("Nonexisting product lattice type! Exiting...");
		exit(1);
	}
}

EXPORT void destroy_lattice(Product_lattice_t lattice)
{
	delete lattice;
}

EXPORT void update_lattice_probs(Product_lattice_t lattice, bin_enc experiment, bin_enc responses, double **dilution)
{
	lattice->update_probs_in_place(experiment, responses, dilution);
}

EXPORT bin_enc BBPA(Product_lattice_t lattice)
{
	return lattice->BBPA(1.0 / (1 << lattice->variants()));
}

EXPORT void MPI_Product_lattice_Initialize(lattice_type_t type, int subjs, int variants)
{
	// std::cout << type << std::endl;
	switch (type)
	{
	case DIST_NON_DILUTION:
		Product_lattice_dist::MPI_Product_lattice_Initialize(subjs, variants);
		break;
		// std::cout << type << std::endl;
	case DIST_DILUTION:
		Product_lattice_dist::MPI_Product_lattice_Initialize(subjs, variants);
		break;
	case REPL_NON_DILUTION:
		Product_lattice::MPI_Product_lattice_Initialize();
		break;
	case REPL_DILUTION:
		Product_lattice::MPI_Product_lattice_Initialize();
		break;
	default:
		throw std::logic_error("Nonexisting product lattice type! Exiting...");
		exit(1);
	}
}

EXPORT void MPI_Product_lattice_Finalize(lattice_type_t type)
{
	switch (type)
	{
	case DIST_NON_DILUTION:
		Product_lattice_dist::MPI_Product_lattice_Finalize();
		break;
	case DIST_DILUTION:
		Product_lattice_dist::MPI_Product_lattice_Finalize();
		break;
	case REPL_NON_DILUTION:
		Product_lattice::MPI_Product_lattice_Finalize();
		break;
	case REPL_DILUTION:
		Product_lattice::MPI_Product_lattice_Finalize();
		break;
	default:
		throw std::logic_error("Nonexisting product lattice type! Exiting...");
		exit(1);
	}
}