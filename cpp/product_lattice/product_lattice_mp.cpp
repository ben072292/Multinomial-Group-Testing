#include "product_lattice_mp.hpp"

double *Product_lattice_mp::temp_post_prob_holder = nullptr;
MPI_Win Product_lattice_mp::win;

Product_lattice_mp::Product_lattice_mp(int subjs, int variants, double *pi0)
{
	_parallelism = MODEL_PARALLELISM;
	_curr_subjs = subjs;
	_orig_subjs = subjs;
	_variants = variants;
	_post_probs = new double[total_state_each()];
	prior_probs(pi0);
}

// For debugging purpose
double Product_lattice_mp::posterior_prob(bin_enc state) const
{
	double ret = -1.0;
	int target_rank = state_to_rank(state);
	int target_offset = state_to_offset(state);
	if (rank == target_rank)
		ret = _post_probs[target_offset];
	MPI_Allreduce(MPI_IN_PLACE, &ret, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	return ret;
}

void Product_lattice_mp::prior_probs(double *pi0)
{
	for (int i = 0; i < total_state_each(); i++)
	{
		_post_probs[i] = prior_prob(offset_to_state(i), pi0);
	}
}

void Product_lattice_mp::update_metadata(double thres_up, double thres_lo)
{
	for (int i = 0; i < curr_atoms(); i++)
	{
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

void Product_lattice_mp::update_metadata_with_shrinking(double thres_up, double thres_lo)
{
	bin_enc clas_atoms = (_pos_clas_atoms | _neg_clas_atoms); // same size as orig layout
	bin_enc curr_clas_atoms = 0;							  // same size as curr layout
	for (int i = 0; i < _orig_subjs * _variants; i++)
	{
		bin_enc orig_index = (1 << i);													 // binary index in decimal for original layout
		bin_enc curr_index = orig_curr_ind_conv(i, _clas_subjs, _orig_subjs, _variants); // binary index in decimal for current layout
		if ((clas_atoms & orig_index))
		{
			curr_clas_atoms |= curr_index;
		}
		else
		{
			double prob_mass = get_atom_prob_mass(curr_index);
			if (prob_mass < thres_lo)
			{ // classified as positive
				curr_clas_atoms |= curr_index;
				_pos_clas_atoms |= orig_index;
			}
			else if (prob_mass > (1 - thres_up))
			{ // classified as positive
				curr_clas_atoms |= curr_index;
				_neg_clas_atoms |= orig_index;
			}
		}
	}
	curr_clas_atoms = curr_shrinkable_atoms(curr_clas_atoms, _curr_subjs, _variants);

	const int target_subj_size = _orig_subjs - __builtin_popcount(update_clas_subj(_pos_clas_atoms | _neg_clas_atoms, _orig_subjs, _variants));
	const int target_prob_size = (1 << (target_subj_size * _variants));
	if (is_classified())
	{ // if fully classified, update variables and return;
		_clas_subjs = update_clas_subj(_pos_clas_atoms | _neg_clas_atoms, _orig_subjs, _variants);
		_curr_subjs = _orig_subjs - __builtin_popcount(_clas_subjs);
		delete[] _post_probs;
		_post_probs = nullptr;
		return;
	}
	// if MP is achievable, i.e., each process has at least 1 state to work, performing MP shrinking
	if (curr_clas_atoms && target_prob_size >= world_size)
		shrinking(curr_clas_atoms);
	// if MP is not achievable, first shrinking the lattice model to the minimum achievable size of MP
	// then perform MP-DP conversion, improving performance and scalability than directly performing MP-DP conversion
	else if (curr_clas_atoms && target_prob_size < world_size)
	{
		// Stage 1: MP shrinking to the minimum achievable size
		// Stage 1.1: preparation on _pos_clas_atoms and _neg_clas_atoms
		const int true_pos_clas_atoms = _pos_clas_atoms;
		const int true_neg_clas_atoms = _neg_clas_atoms;
		clas_atoms = (true_pos_clas_atoms | true_neg_clas_atoms);
		const int intermediate_subj_size = (__builtin_popcount(world_size - 1)) / _variants + 1; // store intermediate subject size to shrink to;
		const int subj_size_difference = intermediate_subj_size - target_subj_size;
		int difference_counter = 0;
		if (intermediate_subj_size == _curr_subjs)
		{
			goto stage_2_2;
		}
		for (int i = 0; i < _orig_subjs; i++)
		{
			if (subj_is_classified(clas_atoms, i, _orig_subjs, _variants) && !(_clas_subjs & (1 << i))) // make sure it's not already classified subject
			{
				clas_atoms -= (1 << i);
				difference_counter++;
			}
			if (difference_counter == subj_size_difference)
				break;
		}
		// pos and neg use the same is okay as we usually do pos|neg
		_pos_clas_atoms = clas_atoms;
		_neg_clas_atoms = clas_atoms;

		// Stage 1.2: shrinking
		curr_clas_atoms = 0;
		for (int i = 0; i < _orig_subjs * _variants; i++)
		{
			bin_enc orig_index = (1 << i);													 // binary index in decimal for original layout
			bin_enc curr_index = orig_curr_ind_conv(i, _clas_subjs, _orig_subjs, _variants); // binary index in decimal for current layout
			if ((clas_atoms & orig_index))
				curr_clas_atoms |= curr_index;
		}
		curr_clas_atoms = curr_shrinkable_atoms(curr_clas_atoms, _curr_subjs, _variants);
		shrinking(curr_clas_atoms);

		// Stage 2: real MP-DP conversion
		// Stage 2.1: Preparation (revert true classification profile)
		_pos_clas_atoms = true_pos_clas_atoms;
		_neg_clas_atoms = true_neg_clas_atoms;
		clas_atoms = (_pos_clas_atoms | _neg_clas_atoms);
		curr_clas_atoms = 0;
		for (int i = 0; i < _orig_subjs * _variants; i++)
		{
			bin_enc orig_index = (1 << i);													 // binary index in decimal for original layout
			bin_enc curr_index = orig_curr_ind_conv(i, _clas_subjs, _orig_subjs, _variants); // binary index in decimal for current layout
			if ((clas_atoms & orig_index))
				curr_clas_atoms |= curr_index;
		}
		curr_clas_atoms = curr_shrinkable_atoms(curr_clas_atoms, _curr_subjs, _variants);

	stage_2_2:
	{
		// Stage 2.2: real MP-DP conversion (The old MP-DP)
		double *candidate_post_probs;
		if (rank == 0)
			candidate_post_probs = new double[(1 << curr_atoms())]{0.0};
		else
			candidate_post_probs = new double[target_prob_size]{0.0};
		MPI_Gather(_post_probs, total_state_each(), MPI_DOUBLE, candidate_post_probs, total_state_each(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (rank == 0)
		{
			double *temp = _post_probs;
			_post_probs = candidate_post_probs;
			Product_lattice::shrinking(curr_clas_atoms);
			_post_probs = temp;
		}
		MPI_Bcast(candidate_post_probs, target_prob_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		delete[] _post_probs;
		_post_probs = candidate_post_probs;
		candidate_post_probs = nullptr;
		_parallelism = DATA_PARALLELISM;
		if (rank != 0)
		{
			_clas_subjs = update_clas_subj(_pos_clas_atoms | _neg_clas_atoms, _orig_subjs, _variants);
			_curr_subjs = _orig_subjs - __builtin_popcount(_clas_subjs);
		}
	}
	}
}

void Product_lattice_mp::shrinking(bin_enc curr_clas_atoms)
{
	int reduce_count = __builtin_popcount(curr_clas_atoms);
	int base_count = curr_atoms() - reduce_count;
	int shrinked_total_state_each = (1 << base_count) / world_size;
	MPI_Win_fence(0, win);
	for (int i = 0; i < total_state_each(); i++)
	{
		bin_enc state = offset_to_state(i);
		bin_enc shrinked_state = 0;
		int pos = 0;
		int reduce_index_counter = 0;
		for (int j = 0; j < curr_atoms(); j++)
		{
			if (curr_clas_atoms & (1 << j))
			{
				if (state & (1 << j))
					pos |= (1 << reduce_index_counter);
				reduce_index_counter++;
			}
			else if (state & (1 << j))
			{
				shrinked_state |= (1 << (j - reduce_index_counter));
			}
		}
		MPI_Put(&_post_probs[i], 1, MPI_DOUBLE, shrinked_state / shrinked_total_state_each, (shrinked_state % shrinked_total_state_each) * (1 << reduce_count) + pos, 1, MPI_DOUBLE, win);
	}
	MPI_Win_fence(0, win);

	for (int i = 0; i < shrinked_total_state_each; i++)
	{
		_post_probs[i] = 0.0;
		for (int j = 0; j < (1 << reduce_count); j++)
		{
			_post_probs[i] += temp_post_prob_holder[i * (1 << reduce_count) + j];
		}
	}
	_clas_subjs = update_clas_subj(_pos_clas_atoms | _neg_clas_atoms, _orig_subjs, _variants);
	_curr_subjs = _orig_subjs - __builtin_popcount(_clas_subjs);
}

// Exhaustive traversal is faster than active generation for atoms
double Product_lattice_mp::get_atom_prob_mass(bin_enc atom) const
{
	double ret = 0.0;
	// #pragma omp parallel for reduction(+ : ret)
	for (int i = 0; i < total_state_each(); i++)
	{
		if ((offset_to_state(i) & atom) == atom)
		{
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
double *Product_lattice_mp::calc_probs(bin_enc experiment, bin_enc response, double **dilution)
{
	double *ret = new double[total_state_each()];
	double denominator = 0.0;
	// #pragma omp parallel for reduction(+ : denominator) // INVESTIGATE: slowdown than serial
	for (int i = 0; i < total_state_each(); i++)
	{
		ret[i] = _post_probs[i] * response_prob(experiment, response, offset_to_state(i), dilution);
		denominator += ret[i];
	}
	MPI_Allreduce(MPI_IN_PLACE, &denominator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	double denominator_inv = 1 / denominator; // division has much higher instruction latency and throughput than multiplication (however latency is not that important since we have many independent divison operations)
	// #pragma omp parallel for
	for (int i = 0; i < total_state_each(); i++)
	{
		ret[i] *= denominator_inv;
	}
	return ret;
}

void Product_lattice_mp::calc_probs_in_place(bin_enc experiment, bin_enc response, double **__restrict__ dilution)
{
	double denominator = 0.0;
	// #pragma omp parallel for schedule(static) reduction(+ : denominator)
	for (int i = 0; i < total_state_each(); i++)
	{
		_post_probs[i] *= response_prob(experiment, response, offset_to_state(i), dilution);
		denominator += _post_probs[i];
	}
	MPI_Allreduce(MPI_IN_PLACE, &denominator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	double denominator_inv = 1 / denominator; // division has much higher instruction latency and throughput than multiplication (however latency is not that important since we have many independent divison operations)
	// #pragma omp parallel for schedule(static)
	for (int i = 0; i < total_state_each(); i++)
	{
		_post_probs[i] *= denominator_inv;
	}
}

bin_enc Product_lattice_mp::halving(double prob) const
{
	return halving_hybrid(prob);
}

bin_enc Product_lattice_mp::halving_mpi(double prob) const
{
	int partition_id = 0;
	int partition_size = (1 << _curr_subjs) * (1 << _variants);
	double partition_mass[partition_size]{0.0};

	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	for (int s_iter = 0; s_iter < total_state_each(); s_iter++)
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
	for (bin_enc experiment = 0; experiment < (1 << _curr_subjs); experiment++)
	{
		for (int i = 0; i < (1 << _variants); i++)
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

// only work for k = 2 because of loop unrolling
// SIMD compatible with both AVX2 and AVX-512 using vector type
// MP-DP need to switch when total states <= 256 (2^8)
bin_enc Product_lattice_mp::halving_mpi_vectorize(double prob) const
{
	int partition_size = (1 << _curr_subjs) * (1 << _variants);
	double partition_mass[partition_size]{0.0};
	_mm512_si ex = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
	_mm512_si partition_id = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	for (int s_iter = 0; s_iter < total_state_each(); s_iter++)
	{
		bin_enc state = offset_to_state(s_iter);
		// __builtin_prefetch((post_probs_ + s_iter + 20), 0, 0);
		for (bin_enc experiment = 0; experiment < (1 << _curr_subjs); experiment += 16)
		{
			ex += experiment;
			// https://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord
			// evaluates to sign = v >> 31 for 32-bit integers. This is one operation faster than the obvious way,
			// sign = -(v < 0). This trick works because when signed integers are shifted right, the value of the
			// far left bit is copied to the other bits. The far left bit is 1 when the value is negative and 0
			// otherwise; all 1 bits gives -1. Unfortunately, this behavior is architecture-specific.
			partition_id |= (1 & (((ex & (state >> 0)) - ex) >> 31));
			partition_id |= (2 & (((ex & (state >> _curr_subjs)) - ex) >> 31));

			partition_id += ex * 4;
			for (int i = 0; i < 16; i++)
			{
				partition_mass[partition_id[i]] += _post_probs[s_iter];
			}
			ex -= experiment;
			partition_id *= 0;
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, partition_mass, partition_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	double temp = 0.0;
	double min = 2.0;
	bin_enc candidate = -1;
	for (bin_enc experiment = 0; experiment < (1 << _curr_subjs); experiment++)
	{
		temp += std::abs(partition_mass[experiment * (1 << _variants)] - prob);
		temp += std::abs(partition_mass[experiment * (1 << _variants) + 1] - prob);
		temp += std::abs(partition_mass[experiment * (1 << _variants) + 2] - prob);
		temp += std::abs(partition_mass[experiment * (1 << _variants) + 3] - prob);

		if (temp < min)
		{
			min = temp;
			candidate = experiment;
		}
		temp = 0.0;
	}

	return candidate;
}

bin_enc Product_lattice_mp::halving_hybrid(double prob) const
{
	Halving_res halving_res;
	int partition_size = (1 << _curr_subjs) * (1 << _variants);
	double partition_mass[partition_size]{0.0};

#pragma omp parallel for schedule(static) reduction(+ : partition_mass)
	// tricky: for each state, check each variant of actively
	// pooled subjects to see whether they are all 1.
	for (int s_iter = 0; s_iter < total_state_each(); s_iter++)
	{
		int partition_id = 0;
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
				partition_id |= ((1 << variant) & (((experiment & (offset_to_state(s_iter) >> (variant * _curr_subjs))) - experiment) >> 31));
			}
			partition_mass[experiment * (1 << _variants) + partition_id] += _post_probs[s_iter];
			partition_id = 0;
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, partition_mass, partition_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

#pragma omp declare reduction(Halving_Min:Halving_res : Halving_res::halving_min(omp_out, omp_in)) initializer(omp_priv = Halving_res())
#pragma omp parallel for schedule(static) reduction(Halving_Min : halving_res)
	for (bin_enc experiment = 0; experiment < (1 << _curr_subjs); experiment++)
	{
		double temp = 0.0;
		for (bin_enc i = 0; i < (1 << _variants); i++)
		{
			temp += std::abs(partition_mass[experiment * (1 << _variants) + i] - prob);
		}
		if (temp < halving_res.min)
		{
			halving_res.min = temp;
			halving_res.candidate = experiment;
		}
		temp = 0.0;
	}
	MPI_Allreduce(MPI_IN_PLACE, &halving_res.candidate, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
	return halving_res.candidate;
}

void Product_lattice_mp::MPI_Product_lattice_Initialize(int atoms, int variants)
{
	temp_post_prob_holder = new double[(1 << (atoms * variants)) / world_size];
	MPI_Win_create(temp_post_prob_holder, sizeof(double) * (1 << (atoms * variants)) / world_size, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
}

void Product_lattice_mp::MPI_Product_lattice_Free()
{
	MPI_Win_free(&win);
	delete[] temp_post_prob_holder;
}