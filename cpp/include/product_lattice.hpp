#pragma once
#include "core.hpp"

typedef enum copy_op
{
	NO_COPY_PROB_DIST,
	SHALLOW_COPY_PROB_DIST,
	DEEP_COPY_PROB_DIST
} copy_op_t;

enum parallelism
{
	MODEL_PARALLELISM,
	DATA_PARALLELISM = 2
};

enum dilution
{
	NON_DILUTION,
	DILUTION
};

enum lattice_type // follow old conventions
{
	MP_NON_DILUTION = MODEL_PARALLELISM + NON_DILUTION + 1,
	MP_DILUTION = MODEL_PARALLELISM + DILUTION + 1,
	DP_NON_DILUTION = DATA_PARALLELISM + NON_DILUTION + 1,
	DP_DILUTION = DATA_PARALLELISM + DILUTION + 1
};

/*
 * READ BEFORE USE:
 * In this version of lattice model, state is represented using A0B0A1B1...
 * The input of prior should also follow this pattern.
 */

// Product_lattice parallelism
class Product_lattice
{

protected:
	static int rank, world_size, _orig_subjs, _variants;
	static double *_pi0;
	int _curr_subjs;			 // counter
	bin_enc _pos_clas_atoms = 0; // BE (binary encoding)
	bin_enc _neg_clas_atoms = 0; // BE
	bin_enc _clas_subjs = 0;	 // BE, variables used for lattice shrinking
	double *_post_probs;

public:
	Product_lattice() {}
	Product_lattice(int n_atom, int n_variant, double *pi0);
	Product_lattice(const Product_lattice &other, copy_op_t op);
	virtual ~Product_lattice();
	virtual Product_lattice *create(int n_atom, int n_variant, double *pi0) const = 0;
	virtual Product_lattice *clone(copy_op_t op) const = 0;
	inline int curr_subjs() const { return _curr_subjs; }
	inline static int variants() { return _variants; };
	inline static double *pi0() { return _pi0; }
	inline static void pi0(double *pi0) { _pi0 = pi0; }
	inline bin_enc pos_clas_atoms() const { return _pos_clas_atoms; }
	inline bin_enc neg_clas_atoms() const { return _neg_clas_atoms; }
	inline int curr_atoms() const { return _curr_subjs * _variants; }
	inline int orig_atoms() const { return _orig_subjs * _variants; }
	inline static int orig_subjs() { return _orig_subjs; }
	inline bin_enc clas_subjs() const { return _clas_subjs; }
	inline int total_states() const { return (1 << (_curr_subjs * _variants)); }
	inline double *posterior_probs() const { return _post_probs; };
	virtual double posterior_prob(bin_enc state) const;
	inline void posterior_probs(double *post_probs) { _post_probs = post_probs; }
	inline bool is_classified() const { return __builtin_popcount(_pos_clas_atoms | _neg_clas_atoms) == orig_atoms(); }
	bin_enc *get_up_set(bin_enc state, bin_enc *ret) const;
	void generate_power_set_adder(bin_enc *add_index, int index_len, bin_enc state, bin_enc *ret) const;
	virtual void prior_probs(double *pi0);
	double prior_prob(bin_enc state, double *pi0) const;
	virtual void update_probs(bin_enc experiment, bin_enc response, double **dilution);
	virtual void update_probs_in_place(bin_enc experiment, bin_enc response, double **dilution);
	virtual double response_prob(bin_enc experiment, bin_enc response, bin_enc true_state, double **dilution) const = 0;
	virtual std::string type() const = 0;
	virtual void update_metadata(double thres_up, double thres_lo);
	virtual bool update_metadata_with_shrinking(double thres_up, double thres_lo);
	virtual void shrinking(int curr_clas_atoms);
	virtual double get_prob_mass(bin_enc state) const;
	virtual double get_atom_prob_mass(bin_enc atom) const;
	virtual bin_enc BBPA(double prob) const;
	virtual enum parallelism parallelism() const { return DATA_PARALLELISM; }
	virtual enum dilution dilution() const { return NON_DILUTION; }
	virtual Product_lattice *convert_parallelism() { return this; }

	/**
	 * Serial
	 */
	virtual bin_enc BBPA_serial(double prob) const;

	/**
	 * Intra-node using OpenMP
	 */
	virtual bin_enc BBPA_omp(double prob) const;

	/**
	 * Inter-node using MPI
	 */
	virtual bin_enc BBPA_mpi(double prob) const;

	/**
	 * Inter-node using MPI + OpenMP
	 */
	virtual bin_enc BBPA_mpi_omp(double prob) const;

#ifdef ENABLE_SIMD
	/**
	 * Inter-node using MPI + SIMD
	 */
	virtual bin_enc BBPA_mpi_simd(double prob) const { throw std::logic_error("BBPA: SIMD not implemented."); }
#endif

	static void MPI_Product_lattice_Initialize();
	static void MPI_Product_lattice_Free();
};

class Product_lattice_dilution : public virtual Product_lattice
{
public:
	Product_lattice_dilution() {} // default constructor

	Product_lattice_dilution(int n_atom, int n_variant, double *pi0) : Product_lattice(n_atom, n_variant, pi0) {}

	Product_lattice_dilution(Product_lattice const &other, copy_op_t op) : Product_lattice(other, op) {}

	Product_lattice *create(int n_atom, int n_variant, double *pi0) const override { return new Product_lattice_dilution(n_atom, n_variant, pi0); }

	Product_lattice *clone(copy_op op) const override { return new Product_lattice_dilution(*this, op); }

	double response_prob(bin_enc experiment, bin_enc response, bin_enc true_state, double **dilution) const override;

	enum dilution dilution() const override { return DILUTION; }

	std::string type() const override { return "Replicated-Dilution"; }
};

class Product_lattice_non_dilution : public virtual Product_lattice
{
public:
	Product_lattice_non_dilution(){}; // default constructor

	Product_lattice_non_dilution(int n_atom, int n_variant, double *pi0) : Product_lattice(n_atom, n_variant, pi0) {}

	Product_lattice_non_dilution(Product_lattice const &other, copy_op_t op) : Product_lattice(other, op) {}

	Product_lattice *create(int n_atom, int n_variant, double *pi0) const override { return new Product_lattice_non_dilution(n_atom, n_variant, pi0); }

	Product_lattice *clone(copy_op_t op) const override { return new Product_lattice_non_dilution(*this, op); }

	double response_prob(bin_enc experiment, bin_enc response, bin_enc true_state, double **dilution) const override;

	std::string type() const override { return "Replicated-NoDilution"; }
};

/*
 * READ BEFORE USE:
 * In this version of lattice model, state is represented using A0B0A1B1...
 * The input of prior should also follow this pattern.
 * Model parallelism implementation for lattice model
 */
class Product_lattice_dist : public virtual Product_lattice
{

protected:
	static double *temp_post_prob_holder;
	static MPI_Win win;
	inline bin_enc total_states_per_rank() const { return total_states() / world_size; }
	inline int state_to_offset(bin_enc state) const { return state % total_states_per_rank(); }
	inline int state_to_rank(bin_enc state) const { return state / total_states_per_rank(); }
	inline bin_enc offset_to_state(int offset) const { return offset + total_states_per_rank() * rank; }

public:
	Product_lattice_dist() {} // default constructor
	Product_lattice_dist(int n_atom, int n_variant, double *pi0);
	Product_lattice_dist(const Product_lattice &other, copy_op_t op) : Product_lattice(other, op){};
	double posterior_prob(bin_enc state) const override;
	void prior_probs(double *pi0) override;
	void update_probs(bin_enc experiment, bin_enc response, double **dilution) override;
	void update_probs_in_place(bin_enc experiment, bin_enc response, double **dilution) override;
	void update_metadata(double thres_up, double thres_lo) override;
	bool update_metadata_with_shrinking(double thres_up, double thres_lo) override;
	void shrinking(int curr_clas_atoms) override;
	double get_prob_mass(bin_enc state) const override { throw std::logic_error("Not Implemented."); }
	double get_atom_prob_mass(bin_enc atom) const override;
	bin_enc BBPA_serial(double prob) const override { throw std::logic_error("Serial BBPA algorithm is not supported in distributed model."); }
	bin_enc BBPA_omp(double prob) const override { throw std::logic_error("OMP BBPA algorithm is not supported in distributed model"); }
	bin_enc BBPA_mpi(double prob) const override;
	bin_enc BBPA_mpi_omp(double prob) const override;
#ifdef ENABLE_SIMD
	bin_enc BBPA_mpi_simd(double prob) const override;
#endif
	bin_enc BBPA(double prob) const override;
	virtual enum parallelism parallelism() const override { return MODEL_PARALLELISM; }
	virtual enum dilution dilution() const override { return NON_DILUTION; }
	static void MPI_Product_lattice_Initialize(int atoms, int variants);
	static void MPI_Product_lattice_Free();
};

class Product_lattice_dist_dilution : public Product_lattice_dist, public Product_lattice_dilution
{
public:
	Product_lattice_dist_dilution(int n_atom, int n_variant, double *pi0) : Product_lattice_dist(n_atom, n_variant, pi0) {}

	// Note the copy constructor directly calls the grandparent copy constructor, i.e., Product_lattice(ohter, copy, op)
	// Quoted from https://www.geeksforgeeks.org/multiple-inheritance-in-c/
	// "In the above program, constructor of ‘Person’ is called once. One important thing to note in the above output is,
	// the default constructor of ‘Person’ is called. When we use ‘virtual’ keyword, the default constructor of grandparent
	// class is called by default even if the parent classes explicitly call parameterized constructor. How to call the
	// parameterized constructor of the ‘Person’ class? The constructor has to be called in ‘TA’ class. For example, see
	// the following program. "
	Product_lattice_dist_dilution(Product_lattice_dist const &other, copy_op_t copy_op) : Product_lattice(other, copy_op) {}

	Product_lattice *create(int n_atom, int n_variant, double *pi0) const override { return new Product_lattice_dist_dilution(n_atom, n_variant, pi0); }

	Product_lattice *clone(copy_op_t op) const override { return new Product_lattice_dist_dilution(*this, op); };

	Product_lattice *convert_parallelism() override
	{
		Product_lattice *p = new Product_lattice_dilution(*this, SHALLOW_COPY_PROB_DIST);
		_post_probs = nullptr;
		delete this;
		return p;
	}

	enum dilution dilution() const override { return DILUTION; }

	std::string type() const override { return "Distributed-Dilution"; }
};

class Product_lattice_dist_non_dilution : public Product_lattice_dist, public Product_lattice_non_dilution
{
public:
	Product_lattice_dist_non_dilution(int n_atom, int n_variant, double *pi0) : Product_lattice_dist(n_atom, n_variant, pi0) {}

	// Note the copy constructor directly calls the grandparent copy constructor, i.e., Product_lattice(ohter, copy, op)
	// Quoted from https://www.geeksforgeeks.org/multiple-inheritance-in-c/
	// "In the above program, constructor of ‘Person’ is called once. One important thing to note in the above output is,
	// the default constructor of ‘Person’ is called. When we use ‘virtual’ keyword, the default constructor of grandparent
	// class is called by default even if the parent classes explicitly call parameterized constructor. How to call the
	// parameterized constructor of the ‘Person’ class? The constructor has to be called in ‘TA’ class. For example, see
	// the following program. "
	Product_lattice_dist_non_dilution(Product_lattice_dist const &other, copy_op_t op) : Product_lattice(other, op) {}

	Product_lattice *create(int n_atom, int n_variant, double *pi0) const override { return new Product_lattice_dist_non_dilution(n_atom, n_variant, pi0); }

	Product_lattice *clone(copy_op_t op) const override { return new Product_lattice_dist_non_dilution(*this, op); };

	Product_lattice *convert_parallelism() override
	{
		Product_lattice *p = new Product_lattice_non_dilution(*this, SHALLOW_COPY_PROB_DIST);
		_post_probs = nullptr;
		delete this;
		return p;
	}

	std::string type() const override { return "Distributed-NoDilution"; }
};

typedef struct BBPA_res
{
	double min;
	bin_enc candidate;

	BBPA_res(double val1 = 2.0, bin_enc val2 = -1)
	{
		min = val1;
		candidate = val2;
	}

	inline void reset() { min = 2.0, candidate = -1; }

	inline static void create_BBPA_res_type(MPI_Datatype *BBPA_res_type)
	{
		int lengths[2] = {1, 1};

		// Calculate displacements
		// In C, by default padding can be inserted between fields. MPI_Get_address will allow
		// to get the address of each struct field and calculate the corresponding displacement
		// relative to that struct base address. The displacements thus calculated will therefore
		// include padding if any.
		MPI_Aint displacements[2];
		struct BBPA_res dummy_BBPA_res;
		MPI_Aint base_address;
		MPI_Get_address(&dummy_BBPA_res, &base_address);
		MPI_Get_address(&dummy_BBPA_res.min, &displacements[0]);
		MPI_Get_address(&dummy_BBPA_res.candidate, &displacements[1]);
		displacements[0] = MPI_Aint_diff(displacements[0], base_address);
		displacements[1] = MPI_Aint_diff(displacements[1], base_address);

		MPI_Datatype types[2] = {MPI_DOUBLE, MPI_INT};
		MPI_Type_create_struct(2, lengths, displacements, types, BBPA_res_type);
	}

	inline static void BBPA_reduce(BBPA_res *in, BBPA_res *inout, int *len, MPI_Datatype *dptr)
	{
		if (in->min < inout->min)
		{
			inout->min = in->min;
			inout->candidate = in->candidate;
		}
	}

	static void BBPA_min(BBPA_res &a, BBPA_res &b)
	{
		if (a.min > b.min)
		{
			a.min = b.min;
			a.candidate = b.candidate;
		}
	}
} BBPA_res;

double **generate_dilution(int n, double alpha, double h);

#define BBPA_UNROLL_1(partition_id, ex, state, curr_subjs) \
	partition_id |= (1 & (((ex & (offset_to_state(state))) - ex) >> 31));

#define BBPA_UNROLL_2(partition_id, ex, state, curr_subjs) \
	BBPA_UNROLL_1(partition_id, ex, state, curr_subjs)     \
	partition_id |= (2 & (((ex & (offset_to_state(state) >> curr_subjs)) - ex) >> 31));

#define BBPA_UNROLL_3(partition_id, ex, state, curr_subjs) \
	BBPA_UNROLL_2(partition_id, ex, state, curr_subjs)     \
	partition_id |= (4 & (((ex & (offset_to_state(state) >> (2 * curr_subjs))) - ex) >> 31));

#define BBPA_UNROLL_4(partition_id, ex, state, curr_subjs) \
	BBPA_UNROLL_3(partition_id, ex, state, curr_subjs)     \
	partition_id |= (8 & (((ex & (offset_to_state(state) >> (3 * curr_subjs))) - ex) >> 31));

#define BBPA_UNROLL_5(partition_id, ex, state, curr_subjs) \
	BBPA_UNROLL_4(partition_id, ex, state, curr_subjs)     \
	partition_id |= (16 & (((ex & (offset_to_state(state) >> (4 * curr_subjs))) - ex) >> 31));

#define BBPA_UNROLL(times, partition_id, ex, state, curr_subjs) \
	BBPA_UNROLL_##times(partition_id, ex, state, curr_subjs)

/**
 * Helper functions for lattice shrinking
 */

// State index conversion from original layout to current (shrinked) layout
// ex: N = 3, k = 2, B is classified, A0 change from index 5 to index 3
bin_enc orig_curr_ind_conv(bin_enc orig_index_pos, bin_enc clas_subjs, int orig_subjs, int variants);

// Determine which atoms are eligible for shrinking,
// i.e., all associated diseases are classified
// sized under current layout
bin_enc curr_shrinkable_atoms(bin_enc curr_clas_atoms, int curr_subjs, int variants);

// Update classified subjects
bin_enc update_clas_subj(bin_enc clas_atoms, int orig_subjs, int variants);

// Check whether a subject is classified
bool subj_is_classified(bin_enc clas_atoms, int subj_pos, int orig_subjs, int variants);