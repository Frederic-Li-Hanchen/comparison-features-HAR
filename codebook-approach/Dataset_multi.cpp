
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include "numpy.hpp"

#include "Dataset_multi.h"
#include "UtilityFunctions.h"

void load_codebook(string codebook_filename, vector< vector<double> > &codebook, int nb_codewords, int window_size){
	
	cout << ">> Open the codebook filename, " << codebook_filename << endl;
	vector<string> codebook_lines;
	read_lines(codebook_filename, codebook_lines);
	if(nb_codewords != (int)codebook_lines.size()){
		cout << "!!! ERROR: # of codewords (" << nb_codewords
			<< ") in seq_info.txt doesn't match # of lines in the codebook file (" << codebook_lines.size() << ")" << endl;
		exit(0);
	}
	
	for(int i = 0; i < (int)codebook_lines.size(); i++){
		
		vector<string> tmp = split(codebook_lines[i], " ");
		if((int)tmp.size() % window_size != 0){
			cout << "!!! ERROR: The dimensionality of " << i << "th codeword (" << tmp.size()
				<< ") must be dividable by the window size (" << window_size << ")" << endl;
			exit(0);
		}
		
		vector<double> codeword;
		for(int j = 0; j < (int)tmp.size(); j++)
			codeword.push_back( stod(tmp[j]) );
		codebook.push_back(codeword);
		
		if(i % 1000 == 0 || i == (int)codebook_lines.size() - 1){
			cout << "** " << i << "-th codeword:";
			for(int j = 0; j < (int)codebook[i].size(); j++)
				cout << " " << codebook[i][j];
			cout << endl;
		}
		
	}
	cout << ">> # of codewords = " << codebook.size() << ", # of dimensions = " << codebook[0].size() << endl;
	
}

// log(exp(m1) + exp(m2)) = log(exp(m1) + exp(m1+sub))
//                          log((1+exp(sub))*exp(m1))
//                          log(exp(m1)) + log(1+exp(sub))
//                          m1 + log(1+exp(sub))
double logSumExp(vector<double> &v){
	double m1, m2, sub;
	double Z = v[0];
	for(int i = 1; i < (int)v.size(); i++){
		if(Z >= v[i]){
			m1 = Z;
			m2 = v[i];
		}
		else{
			m1 = v[i];
			m2 = Z;
		}
		sub = m2 - m1;
		Z = m1 + log(1 + exp(sub));
	}
	return Z;
}

void extract_feature_soft_assignment_multi(string input_filename, string output_filename,
				vector< vector< vector<double> > > &codebooks, int window_size, int sliding_size, double sigma){
	
	cout << "*** Open the input numpy file, " << input_filename << " ***" << endl;
	vector<int> size;
	// raw_vals must be <# of examples> x <length of each example> x <# of sensors>
	vector<float> raw_vals;  //vector<double> raw_vals; // if the type of npy file elements is float64
	aoba::LoadArrayFromNumpy(input_filename, size, raw_vals);
	cout << ">> The loaded raw_vals has " << size.size() << " dimensions: "
		<< size[0] << " x " << size[1] << " x " << size[2] << endl;
	if(size[2] != (int)codebooks.size()){
		cout << "!!! ERROR: Example's dimensionality (" << size[2]
			<< ") does not match with the number of codebooks (" << codebooks.size() << ")" << endl;
		exit(0);
	}
	
	int nb_all_elems = 0;
	int nb_total_dims = 0;
	for(int i = 0; i < size[2]; i++){
		nb_all_elems += (size[0] * (int)(codebooks[i].size()));
		nb_total_dims += (int)codebooks[i].size();
	}
	cout << ">> # of all elements in the output numpy file = " << nb_all_elems
		<< " (total # of dimensions = " << nb_total_dims << ")" << endl;
	
	// Elements in this array are filled with "total_feat" for each example
	float* all_total_feats = (float*)malloc( nb_all_elems * sizeof(float) );
	
	int atf_offset = 0; // Indicates the offset position to store each example's "total_feat" in all_total_feats
	for(int ex_id = 0; ex_id < size[0]; ex_id++){
		
		if(ex_id % 100 == 0 || ex_id == size[0] - 1)
			cout << "*****\n** Extract features for " << ex_id << "th example (atf_offset:" << atf_offset << ")\n*****" << endl;
		// This is created by concatenating "feat" extracted for each dimension (i.e. early fusion)
		vector<double> total_feat;
		
		// dim_id changes from 0 to size[2]
		for(int dim_id = 0; dim_id < size[2]; dim_id++){
			
			bool debug_flag = false;
			if( (dim_id % 100 == 0 || dim_id == size[2] - 1) && (ex_id % 100 == 0 || ex_id == size[0] - 1) )
				debug_flag = true;
			// t changes from 0 to size[1] (63 = 64 - 1)
			//for(int t = 0; t < size[1]; t++){
			//	int pos = ex_id * size[1] * size[2] + t * size[2] + dim_id;
			//	if(ex_id == 0 && dim_id == 0)
			//		cout << "(" << ex_id << "," << dim_id << ") t = "
			//			<< t << " (" << pos << "): " << raw_vals[pos] << endl;
			//}
			
			int word_num = (int)codebooks[dim_id].size();
			vector<double> feat(word_num, 0);
			
			int window_id = 0;
			int window_start_pos = 0;
			int window_end_pos = window_start_pos + window_size - 1;
			while(window_end_pos < size[1]){
				
				int offset_pos = ex_id * size[1] * size[2] + window_start_pos * size[2] + dim_id;
				double offset = raw_vals[offset_pos];
				
				vector<double> window_feat;
				for(int t = window_start_pos; t <= window_end_pos; t++){
					int pos = ex_id * size[1] * size[2] + t * size[2] + dim_id;
//					window_feat.push_back( raw_vals[pos] - offset );
					window_feat.push_back( raw_vals[pos] );
				}
				
				int most_sim_word_id = -1;
				double min_dist = DBL_MAX;
				vector<double> tmp_for_norm(word_num, 0);
				for(int i = 0; i < word_num; i++){
					double dist = compute_eucldian_distance(window_feat, codebooks[dim_id][i]);
					tmp_for_norm[i] = (-0.5*dist) / (sigma*sigma);
					//cout << "** " << i << "-th codeword ** " << dist << ", " << tmp_for_norm[i] << endl;
					if(min_dist > dist){
						most_sim_word_id = i;
						min_dist = dist;
					}
				}
				//cout << "<< most_sim_word_id = " << most_sim_word_id << " (" << min_dist << ")" << endl;
				
				// Use logSumExp to avoid the situation where exp(tmp_for_norm[i]) will diverge.
				double log_sum = logSumExp(tmp_for_norm);
				//cout << ">> log_sum = " << log_sum << endl;
				
				double total_contribution = 0;
				for(int i = 0; i < word_num; i++){
					//cout << "++ " << i << "-th codeword ++ " << tmp_for_norm[i] << " ";
					tmp_for_norm[i] -= log_sum;
					//cout << tmp_for_norm[i] << " ";
					tmp_for_norm[i] = exp( tmp_for_norm[i] );
					//cout << tmp_for_norm[i] << endl;
					total_contribution += tmp_for_norm[i];
				}
				
				// This condition may be satisfied with a very very small probability, if there is no similar codeword to window_feat.
				// In this case, instead of ignoring window_feat, I prefer to incrementing the frequency of the anyway most similar to codeword.
				if(total_contribution < DBL_MIN){
					cerr << "!!! WARNING: total_contribution is rounded into ZERO (ex:" << ex_id
						<< ", dim:" << dim_id << ", w:" << window_id << ") -> Increment the frequency of "
						<< most_sim_word_id << "th word (" << min_dist << ")" << endl;
					feat[most_sim_word_id] += 1.0;
				}
				else{
					for(int i = 0; i < word_num; i++)
						feat[i] += tmp_for_norm[i];
				}

				if(debug_flag == true && window_id % 100 == 0){
					cout << "<" << dim_id << "th dim> " << window_id << "th window (" << window_start_pos
						<< " - " << window_end_pos << ") ::: offset:" << offset << " -> ";
					for(int i = 0; i < (int)window_feat.size(); i++)
						cout << window_feat[i] << " ";
					cout << " (" << window_feat.size() << ")\n==> The most similar word ID = "
						<< most_sim_word_id << " (" << min_dist << ") :::";
					for(int i = 0; i < (int)tmp_for_norm.size(); i++)
						if(tmp_for_norm[i] > 0.01)
							cout << " " << i << ":" << tmp_for_norm[i];
					cout << endl;
				}
				
				window_start_pos += sliding_size;
				window_end_pos += sliding_size;
				window_id++;
				
			} // end of 'while(window_end_pos < size[1])
			
			int prev_nb_dims = (int)total_feat.size(); // Only for debug
			if(window_id > 0){
				for(int i = 0; i < (int)feat.size(); i++)
					total_feat.push_back( feat[i] / (double)window_id );
			}
			else{
				// If the target sequence is too short and no window cannot be located,
				// add a feature where all dimensions are zero   
				for(int i = 0; i < (int)feat.size(); i++)
					total_feat.push_back( feat[i] );
			}
			
			if(debug_flag == true){
				cout << "<" << dim_id << "th dim> Soft assignment feature (# of windows = " << window_id << ")" << endl;
				int non_zero_dim_num = 0;
				for(int i = prev_nb_dims; i < (int)total_feat.size(); i++){
					if(total_feat[i] > 0.01){
						non_zero_dim_num++;
						cout << i << "(" << (i-prev_nb_dims) << "):"
							<< feat[i-prev_nb_dims] << "->" << total_feat[i] << " ";
					}
				}
				cout << " (" << non_zero_dim_num << ")" << endl;
			}
			//exit(0);
		} // end of 'for(int dim_id = 0; dim_id < size[2]; dim_id++)
		//exit(0);
		
		for(int i = 0; i < (int)total_feat.size(); i++)
			all_total_feats[i+atf_offset] = (float)total_feat[i];
		atf_offset += (int)total_feat.size();
		
	} // end of 'for(int ex_id = 0; ex_id < size[0]; ex_id++)
	//exit(0);
	
	cout << ">> Output features for all the examples into " << output_filename << endl;
	aoba::SaveArrayAsNumpy(output_filename, size[0], nb_total_dims, &all_total_feats[0]);
	free(all_total_feats);
	
}

void extract_feature_hard_assignment_multi(string input_filename, string output_filename,
									vector< vector< vector<double> > > &codebooks, int window_size, int sliding_size){
	
	cout << "*** Open the input numpy file, " << input_filename << " ***" << endl;
	vector<int> size;
	// raw_vals must be <# of examples> x <length of each example> x <# of sensors>
	vector<float> raw_vals;  //vector<double> raw_vals; // if the type of npy file elements is float64
	aoba::LoadArrayFromNumpy(input_filename, size, raw_vals);
	cout << ">> The loaded raw_vals has " << size.size() << " dimensions: "
		<< size[0] << " x " << size[1] << " x " << size[2] << endl;
	if(size[2] != (int)codebooks.size()){
		cout << "!!! ERROR: Example's dimensionality (" << size[2]
			<< ") does not match with the number of codebooks (" << codebooks.size() << ")" << endl;
		exit(0);
	}
	
	int nb_all_elems = 0;
	int nb_total_dims = 0;
	for(int i = 0; i < size[2]; i++){
		nb_all_elems += (size[0] * (int)(codebooks[i].size()));
		nb_total_dims += (int)codebooks[i].size();
	}
	cout << ">> # of all elements in the output numpy file = " << nb_all_elems
		<< " (total # of dimensions = " << nb_total_dims << ")" << endl;
	
	// Elements in this array are filled with "total_feat" for each example
	float* all_total_feats = (float*)malloc( nb_all_elems * sizeof(float) );
	
	int atf_offset = 0; // Indicates the offset position to store each example's "total_feat" in all_total_feats
	for(int ex_id = 0; ex_id < size[0]; ex_id++){
		
		if(ex_id % 100 == 0 || ex_id == size[0] - 1)
			cout << "*****\n** Extract features for " << ex_id << "th example (atf_offset:" << atf_offset << ")\n*****" << endl;
		// This is created by concatenating "feat" extracted for each dimension (i.e. early fusion)
		vector<double> total_feat;
		
		// dim_id changes from 0 to size[2]
		for(int dim_id = 0; dim_id < size[2]; dim_id++){
			
			bool debug_flag = false;
			if( (dim_id % 100 == 0 || dim_id == size[2] - 1) && (ex_id % 100 == 0 || ex_id == size[0] - 1) )
				debug_flag = true;
			// t changes from 0 to size[1] (63 = 64 - 1)
			//for(int t = 0; t < size[1]; t++){
			//	int pos = ex_id * size[1] * size[2] + t * size[2] + dim_id;
			//	if(ex_id == 0 && dim_id == 0)
			//		cout << "(" << ex_id << "," << dim_id << ") t = "
			//			<< t << " (" << pos << "): " << raw_vals[pos] << endl;
			//}
			
			int word_num = (int)codebooks[dim_id].size();
			vector<double> feat(word_num, 0);
			
			int window_id = 0;
			int window_start_pos = 0;
			int window_end_pos = window_start_pos + window_size - 1;
			while(window_end_pos < size[1]){
				
				int offset_pos = ex_id * size[1] * size[2] + window_start_pos * size[2] + dim_id;
				double offset = raw_vals[offset_pos];
				
				vector<double> window_feat;
				for(int t = window_start_pos; t <= window_end_pos; t++){
					int pos = ex_id * size[1] * size[2] + t * size[2] + dim_id;
					//window_feat.push_back( raw_vals[pos] - offset );
					window_feat.push_back( raw_vals[pos] );
				}
				
				int ass_word_id = -1;
				double min_dist = DBL_MAX;
				for(int i = 0; i < word_num; i++){
					double dist = compute_eucldian_distance(window_feat, codebooks[dim_id][i]);
					//if(debug_flag == true && window_id % 1000 == 0)
					//	cout << "-- " << i << "th codeword: " << dist << " (" << min_dist << ")" << endl;
					if(dist < min_dist){
						min_dist = dist;
						ass_word_id = i;
					}
				}
				feat[ass_word_id]++;
				
				if(debug_flag == true && window_id % 100 == 0){
					cout << "<" << dim_id << "th dim> " << window_id << "th window (" << window_start_pos
						<< " - " << window_end_pos << ") ::: offset:" << offset << " -> ";
					for(int i = 0; i < (int)window_feat.size(); i++)
						cout << window_feat[i] << " ";
					cout << " (" << window_feat.size() << ")\n==> Assigned word ID = "
						<< ass_word_id << " (" << min_dist << ")" << endl;
				}
				
				window_start_pos += sliding_size;
				window_end_pos += sliding_size;
				window_id++;
				
			} // end of 'while(window_end_pos < size[1])
			
			int prev_nb_dims = (int)total_feat.size(); // Only for debug
			if(window_id > 0){
				for(int i = 0; i < (int)feat.size(); i++)
					total_feat.push_back( feat[i] / (double)window_id );
			}
			else{
				// If the target sequence is too short and no window cannot be located,
				// add a feature where all dimensions are zero   
				for(int i = 0; i < (int)feat.size(); i++)
					total_feat.push_back( feat[i] );
			}
			
			if(debug_flag == true){
				cout << "<" << dim_id << "th dim> Hard assignment feature (# of windows = " << window_id << ")" << endl;
				int non_zero_dim_num = 0;
				for(int i = prev_nb_dims; i < (int)total_feat.size(); i++){
					if(total_feat[i] > 0){
						non_zero_dim_num++;
						cout << i << "(" << (i-prev_nb_dims) << "):"
							<< feat[i-prev_nb_dims] << "->" << total_feat[i] << " ";
					}
				}
				cout << " (" << non_zero_dim_num << ")" << endl;
			}
			
		} // end of 'for(int dim_id = 0; dim_id < size[2]; dim_id++)
		//exit(0);
		
		for(int i = 0; i < (int)total_feat.size(); i++)
			all_total_feats[i+atf_offset] = (float)total_feat[i];
		atf_offset += (int)total_feat.size();
		
	} // end of 'for(int ex_id = 0; ex_id < size[0]; ex_id++)
	//exit(0);
	
	cout << ">> Output features for all the examples into " << output_filename << endl;
	aoba::SaveArrayAsNumpy(output_filename, size[0], nb_total_dims, &all_total_feats[0]);
	free(all_total_feats);
	
}

