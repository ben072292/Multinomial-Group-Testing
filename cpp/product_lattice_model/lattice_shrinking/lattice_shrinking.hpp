/**
 * Helper functions for lattice shrinking
*/

// State index conversion from original layout to current (shrinked) layout 
// ex: N = 3, k = 2, B is classified, A0 change from index 5 to index 3
inline int orig_curr_ind_conv(bin_enc orig_index_pos, bin_enc clas_subjs, int orig_subjs, int variants){
	int ret = orig_index_pos;
	for(int i = 0; i < orig_subjs; i++){
		for(int j = 0; j < variants; j++){
			if((clas_subjs & (1 << i)) && (orig_index_pos > (j * orig_subjs + i))){
				ret--;
			}
			else if((clas_subjs & (1 << i)) && (orig_index_pos == (j * orig_subjs + i))){ // handle the case where already classified subjects are considered
				return 0;
			}
		}
	}
	return (1 << ret);
}

// Determine which subjects are eligible for shrinking, 
// i.e., all associated diseases are classified
inline bin_enc curr_shrinkable_atoms(bin_enc curr_clas_atoms, int curr_subjs, int variants){
	bin_enc ret = 0;
	for(int i = 0; i < curr_subjs; i++){
		bool subj_classified = true;
		for(int j = 0; j < variants; j++){
			if(!(curr_clas_atoms & (1 << (j * curr_subjs + i)))){
				subj_classified = false;
				continue;
			}
		}
		if(subj_classified){
			for(int j = 0; j < variants; j++){
				ret |= (1 << (j * curr_subjs + i));
			}
		}
	}
	return ret;
}

// Update classified subjects
inline bin_enc update_clas_subj(bin_enc clas_atoms, int orig_subjs, int variants){
	bin_enc ret = 0;
	for(int i = 0; i < orig_subjs; i++){
		bool subj_classified = true;
		for(int j = 0; j < variants; j++){
			if(!(clas_atoms & (1 << (j * orig_subjs + i)))){
				subj_classified = false;
				continue;
			}
		}
		if(subj_classified){
			ret |= (1 << i);
		}
	}
	return ret;
}