/**
 * Helper functions for lattice shrinking
*/

// State index conversion from original layout to current (shrinked) layout 
// ex: N = 3, k = 2, B is classified, A0 change from index 5 to index 3
inline int orig_curr_ind_conv(int orig_index_pos, int clas_subjs, int orig_subjs, int variant){
	int ret = orig_index_pos;
	for(int i = 0; i < orig_subjs; i++){
		for(int j = 0; j < variant; j++){
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
inline int curr_shrinkable_atoms(int curr_clas_atoms, int curr_subj, int variant){
	int ret = 0;
	for(int i = 0; i < curr_subj; i++){
		bool subj_classified = true;
		for(int j = 0; j < variant; j++){
			if(!(curr_clas_atoms & (1 << (j * curr_subj + i)))){
				subj_classified = false;
				continue;
			}
		}
		if(subj_classified){
			for(int j = 0; j < variant; j++){
				ret |= (1 << (j * curr_subj + i));
			}
		}
	}
	return ret;
}

// Update classified subjects
inline int update_clas_subj(int clas_atoms, int orig_subj, int variant){
	int ret = 0;
	for(int i = 0; i < orig_subj; i++){
		bool subj_classified = true;
		for(int j = 0; j < variant; j++){
			if(!(clas_atoms & (1 << (j * orig_subj + i)))){
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