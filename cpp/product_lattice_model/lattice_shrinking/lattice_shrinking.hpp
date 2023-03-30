#pragma once
#include "../../core.hpp"
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