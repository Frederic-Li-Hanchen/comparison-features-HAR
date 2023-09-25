
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include "numpy.hpp"

#include "Dataset_single.h"
#include "Dataset_multi.h"
#include "UtilityFunctions.h"


void extract_feature_hard_assignment_single(string input_filename, string output_filename,
									vector< vector<double> > &codebook, int window_size, int sliding_size){
	
	cout << "*** Open the input numpy file, " << input_filename << " ***" << endl;
	vector<int> size;
	// raw_vals must be <# of sequences> x <length of each sequence> x <# of dimensions>
	vector<float> raw_vals;  //vector<double> raw_vals; // if the type of npy file elements is float64
	aoba::LoadArrayFromNumpy(input_filename, size, raw_vals);
	cout << ">> The loaded raw_vals has " << size.size() << " dimensions: "
		<< size[0] << " x " << size[1] << " x " << size[2] << endl;
	int nb_dims = (int)codebook[0].size() / window_size;
	if(size[2] != nb_dims){
		cout << "!!! ERROR: Example's dimensionality (" << size[2]
			<< ") does not match with the dimensionality of a codeword (" << nb_dims
			<< ", window_size:" << window_size << ", codeword_length:" << codebook[0].size() << ")" << endl;
		exit(0);
	}
	
	int word_num = (int)codebook.size();
	int nb_all_elems = size[0] * word_num;
	cout << ">> # of all elements in the output numpy file = " << nb_all_elems
		<< " (each sequence is coded by a " << word_num << " dimensional feature)" << endl;
	
	// Elements in this array are filled with "total_feat" for each example
	float* all_total_feats = (float*)malloc( nb_all_elems * sizeof(float) );
	
	int atf_offset = 0; // Indicates the offset position to store each example's "total_feat" in all_total_feats
	for(int ex_id = 0; ex_id < size[0]; ex_id++){
		
		bool debug_flag = false;
		if(ex_id % 100 == 0 || ex_id == size[0] - 1){
			cout << "*****\n** Extract features for " << ex_id << "th example (atf_offset:" << atf_offset << ")\n*****" << endl;
			debug_flag = true;
		}
		
		// Considering the smooth integration wiht already implemented functions (load_codebook
		// and compute_euclidean_distances), float values in raw_vals are casted into double.
		vector< vector<double> > seq;
		for(int i = 0; i < size[1]; i++){
			vector<double> vals;
			for(int j = 0; j < nb_dims; j++){
				int pos = ex_id * size[1] * nb_dims + i * nb_dims + j;
				vals.push_back( (double)raw_vals[pos] );
			}
			seq.push_back(vals);
		}
		//if(debug_flag){
		//	cout << "------- " << ex_id << "th sequence -------" << endl;
		//	for(int i = 0; i < (int)seq.size(); i++){
		//		cout << "-- " << i << ":";
		//		for(int j = 0; j < (int)seq[i].size(); j++)
		//			cout << " " << seq[i][j];
		//		cout << endl;
		//	}
		//}
		
		vector<float> feat(word_num, 0);
		int window_id = 0;
		int window_start_pos = 0;
		int window_end_pos = window_start_pos + window_size - 1;
		while(window_end_pos < size[1]){
			
			vector<double> offsets = seq[window_start_pos];
			vector<double> window_data;
			for(int i = window_start_pos; i <= window_end_pos; i++){
				for(int j = 0; j < nb_dims; j++)
					window_data.push_back( seq[i][j] );
					//window_data.push_back( seq[i][j] - offsets[j] );
			}
			
			int ass_word_id = -1;
			double min_dist = DBL_MAX;
			for(int i = 0; i < word_num; i++){
				double dist = compute_eucldian_distance(window_data, codebook[i]);
				//if(debug_flag == true && window_id % 100 == 0)
				//	cout << "-- " << i << "th codeword: " << dist << " (" << min_dist << ")" << endl;
				if(dist < min_dist){
					min_dist = dist;
					ass_word_id = i;
				}
			}
			feat[ass_word_id]++;
			
			if(debug_flag == true && window_id % 100 == 0){
				cout << "++ " << window_id << "th window (" << window_start_pos << " - " << window_end_pos
					<< ") ::: offsets (";
				for(int i = 0; i < (int)offsets.size(); i++)
					cout << offsets[i] << ", ";
				cout << ") -> ";
				for(int i = 0; i < (int)window_data.size(); i++)
					cout << window_data[i] << " ";
				cout << " (" << window_data.size() << ")\n==> Assigned word ID = "
					<< ass_word_id << " (" << min_dist << ")" << endl;
			}
			
			window_start_pos += sliding_size;
			window_end_pos += sliding_size;
			window_id++;
			
		} // end of 'while(window_end_pos < size[1])
		
		if(window_id > 0){
			for(int i = 0; i < (int)feat.size(); i++)
				feat[i] /= (float)window_id;
		}
		
		if(debug_flag == true){
			cout << ">> Hard assignment feature (# of windows = " << window_id << ")" << endl;
			int non_zero_dim_num = 0;
			for(int i = 0; i < (int)feat.size(); i++){
				if(feat[i] > 0){
					non_zero_dim_num++;
					cout << i << ":" << feat[i] << " ";
				}
			}
			cout << " (" << non_zero_dim_num << ")" << endl;
		}
		
		for(int i = 0; i < (int)feat.size(); i++)
			all_total_feats[i+atf_offset] = feat[i];
		atf_offset += (int)feat.size();
		
	} // end of 'for(int ex_id = 0; ex_id < size[0]; ex_id++)
	//exit(0);
	
	cout << ">> Output features for all the examples into " << output_filename << endl;
	aoba::SaveArrayAsNumpy(output_filename, size[0], word_num, &all_total_feats[0]);
	free(all_total_feats);
	
}

void extract_feature_soft_assignment_single(string input_filename, string output_filename,
				vector< vector<double> > &codebook, int window_size, int sliding_size, double sigma){
	
	cout << "*** Open the input numpy file, " << input_filename << " ***" << endl;
	vector<int> size;
	// raw_vals must be <# of sequences> x <length of each sequence> x <# of dimensions>
	vector<float> raw_vals;  //vector<double> raw_vals; // if the type of npy file elements is float64
	aoba::LoadArrayFromNumpy(input_filename, size, raw_vals);
	cout << ">> The loaded raw_vals has " << size.size() << " dimensions: "
		<< size[0] << " x " << size[1] << " x " << size[2] << endl;
	int nb_dims = (int)codebook[0].size() / window_size;
	if(size[2] != nb_dims){
		cout << "!!! ERROR: Example's dimensionality (" << size[2]
			<< ") does not match with the dimensionality of a codeword (" << nb_dims
			<< ", window_size:" << window_size << ", codeword_length:" << codebook[0].size() << ")" << endl;
		exit(0);
	}
	
	int word_num = (int)codebook.size();
	int nb_all_elems = size[0] * word_num;
	cout << ">> # of all elements in the output numpy file = " << nb_all_elems
		<< " (each sequence is coded by a " << word_num << " dimensional feature)" << endl;
	
	// Elements in this array are filled with "total_feat" for each example
	float* all_total_feats = (float*)malloc( nb_all_elems * sizeof(float) );
	
	int atf_offset = 0; // Indicates the offset position to store each example's "total_feat" in all_total_feats
	for(int ex_id = 0; ex_id < size[0]; ex_id++){
		
		bool debug_flag = false;
		if(ex_id % 100 == 0 || ex_id == size[0] - 1){
			cout << "*****\n** Extract features for " << ex_id << "th example (atf_offset:" << atf_offset << ")\n*****" << endl;
			debug_flag = true;
		}
		
		// Considering the smooth integration wiht already implemented functions (load_codebook
		// and compute_euclidean_distances), float values in raw_vals are casted into double.
		vector< vector<double> > seq;
		for(int i = 0; i < size[1]; i++){
			vector<double> vals;
			for(int j = 0; j < nb_dims; j++){
				int pos = ex_id * size[1] * nb_dims + i * nb_dims + j;
				vals.push_back( (double)raw_vals[pos] );
			}
			seq.push_back(vals);
		}
		//if(debug_flag){
		//	cout << "------- " << ex_id << "th sequence -------" << endl;
		//	for(int i = 0; i < (int)seq.size(); i++){
		//		cout << "-- " << i << ":";
		//		for(int j = 0; j < (int)seq[i].size(); j++)
		//			cout << " " << seq[i][j];
		//		cout << endl;
		//	}
		//}
		
		vector<float> feat(word_num, 0);
		int window_id = 0;
		int window_start_pos = 0;
		int window_end_pos = window_start_pos + window_size - 1;
		while(window_end_pos < size[1]){
			
			vector<double> offsets = seq[window_start_pos];
			vector<double> window_data;
			for(int i = window_start_pos; i <= window_end_pos; i++){
				for(int j = 0; j < nb_dims; j++)
					window_data.push_back( seq[i][j] );
					//window_data.push_back( seq[i][j] - offsets[j] );
			}
			
			int most_sim_word_id = -1;
			double min_dist = DBL_MAX;
			vector<double> tmp_for_norm(word_num, 0);
			for(int i = 0; i < word_num; i++){
				double dist = compute_eucldian_distance(window_data, codebook[i]);
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
				cerr << "!!! WARNING: total_contribution is rounded into ZERO (ex:"
					<< ex_id << ", w:" << window_id << ") -> Increment the frequency of "
					<< most_sim_word_id << "th word (" << min_dist << ")" << endl;
				feat[most_sim_word_id] += 1.0;
			}
			else{
				for(int i = 0; i < word_num; i++)
					feat[i] += tmp_for_norm[i];
			}
			
			if(debug_flag == true && window_id % 100 == 0){
				cout << "++ " << window_id << "th window (" << window_start_pos
					<< " - " << window_end_pos << ") ::: offsets (";
				for(int i = 0; i < (int)offsets.size(); i++)
					cout << offsets[i] << ", ";
				cout << ") -> ";
				for(int i = 0; i < (int)window_data.size(); i++)
					cout << window_data[i] << " ";
				cout << " (" << window_data.size() << ")\n==> The most similar word ID = "
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
		
		if(window_id > 0){
			for(int i = 0; i < (int)feat.size(); i++)
				feat[i] /= (float)window_id;
		}
		
		if(debug_flag == true){
			cout << ">> Soft assignment feature (# of windows = " << window_id << ", sigma:" << sigma << ")" << endl;
			int non_zero_dim_num = 0;
			for(int i = 0; i < (int)feat.size(); i++){
				if(feat[i] > 0.01){
					non_zero_dim_num++;
					cout << i << ":" << feat[i] << " ";
				}
			}
			cout << " (" << non_zero_dim_num << ")" << endl;
		}
		
		for(int i = 0; i < (int)feat.size(); i++)
			all_total_feats[i+atf_offset] = feat[i];
		atf_offset += (int)feat.size();
		
	} // end of 'for(int ex_id = 0; ex_id < size[0]; ex_id++)
	//exit(0);
	
	cout << ">> Output features for all the examples into " << output_filename << endl;
	aoba::SaveArrayAsNumpy(output_filename, size[0], word_num, &all_total_feats[0]);
	free(all_total_feats);
	
}

