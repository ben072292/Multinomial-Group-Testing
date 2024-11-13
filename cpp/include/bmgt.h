#ifndef BMGT_H
#define BMGT_H

#ifdef __cplusplus
extern "C" {
#endif
// Differentiate binary encoded states from regular index
typedef int bin_enc;

typedef enum lattice_copy_op
{
	NO_COPY_PROB_DIST,
	SHALLOW_COPY_PROB_DIST,
	DEEP_COPY_PROB_DIST
} lattice_copy_op_t;

typedef enum lattice_parallelism
{
	DIST_MODEL,
	REPL_MODEL = 2
} lattice_parallelism_t;

typedef enum dilution
{
	NON_DILUTION,
	DILUTION
} dilution_t;

typedef enum lattice_type // follow old conventions
{
	DIST_NON_DILUTION = DIST_MODEL + NON_DILUTION + 1,
	DIST_DILUTION = DIST_MODEL + DILUTION + 1,
	REPL_NON_DILUTION = REPL_MODEL + NON_DILUTION + 1,
	REPL_DILUTION = REPL_MODEL + DILUTION + 1
} lattice_type_t;

typedef enum tree_type
{
	DIST_GLOB_TREE_SYMM,
	DIST_GLOB_TREE_TRIM,
	DIST_GLB_TREE,
	DIST_TREE_INTRA,
	DIST_TREE,
	FUSION_TREE,
	GLOB_TREE,
	GLOB_PARTIAL_TREE_TRIM,
	GLOB_PARTIAL_TREE_SYMM,
	GLOB_PARTIAL_TREE_TRIM_MISSPEC,
	GLOB_TREE_INTRA
} tree_type_t;

/**
 * Product lattice operations
 */

// opaque handle to product lattice
typedef class Product_lattice *Product_lattice_t;
Product_lattice_t create_lattice(lattice_type_t type, int subjs, int variants, double *pi0);
Product_lattice_t clone_lattice(lattice_type_t type, lattice_copy_op_t op, Product_lattice_t lattice);
void MPI_Product_lattice_Initialize(lattice_type_t type, int subjs, int variants);
void MPI_Product_lattice_Finalize(lattice_type_t type);
void destroy_lattice(Product_lattice_t lattice);
double **generate_dilution(int n, double alpha, double h);
void update_lattice_probs(Product_lattice_t lattice, bin_enc experiment, bin_enc responses, double **dilution);
bin_enc BBPA(Product_lattice_t lattice);

/**
 * Tree-based Simulations
 */

// opaque handle to tree-based statistical analysis
typedef class Tree *Tree_t;
void run_dist_glob_tree_symm(int argc, char *argv[]);
void run_dist_glob_tree_trim(int argc, char *argv[]);
void run_dist_glob_tree(int argc, char *argv[]);
void run_dist_tree_intra(int argc, char *argv[]);
void run_dist_tree(int argc, char *argv[]);
void run_fusion_tree(int argc, char *argv[]);
void run_glob_tree_intra(int argc, char *argv[]);
void run_glob_tree(int argc, char *argv[]);
void run_glob_partial_tree_symm(int argc, char *argv[]);
void run_glob_partial_tree_trim(int argc, char *argv[]);
void run_glob_partial_tree_trim_misspec(int argc, char *argv[]);
void run_model_manipulation_benchmark(int argc, char *argv[]);
void run_BBPA_benchmark(int argc, char *argv[]);
void run_BBPA_benchmark_intra(int argc, char *argv[]);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif