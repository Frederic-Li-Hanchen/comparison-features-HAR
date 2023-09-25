
#include <string.h>
#include <iostream>
#include <string>
#include <vector>

#include "UtilityFunctions.h"
#include "Dataset_multi.h"
#include "Dataset_single.h"

void usage(){
	cout << "Usage: ./main -d <Directory of *_data.npy files> -w <Window size> -l <Sliding stride size>"
		<< " -c <Directory of codebook files> -n <# of codewords per codebook> -o <Directory of output feature files>" << endl;
	cout << " *** The followings are optional ***" << endl;
	cout << " -s <Smoothing parameter (default -1: A value <= 0 means hard assignment, otherwise soft assignment)>" << endl;
	cout << " -st <ID of the file (*_data.npy) from which feature encoding starts>" << endl;
	cout << "     (default -1 meaning that feature encoding will start from the first (0th) file)" << endl;
	cout << " -ed <ID of the file (*_data.npy) at which feature econding ends>" << endl;
	cout << "     (default -1 meaning that feature encoding will end at the last file>" << endl;
	exit(0);
}

int main(int argc, char* argv[]){
	
	string ex_dir = "";
	int window_size = -1;
	int sliding_size = -1;
	string codebook_dir = "";
	int codeword_num = -1;
	string output_dir = "";
	double sigma = -1;
	int st_id = -1;
	int ed_id = -1;
	for(int i = 1; i < argc; i++){
		if( strcmp(argv[i], "-d") == 0 )
			ex_dir = string( argv[++i] );
		else if( strcmp(argv[i], "-w") == 0 )
			window_size = atoi( argv[++i] );
		else if( strcmp(argv[i], "-l") == 0 )
			sliding_size = atoi( argv[++i] );
		else if( strcmp(argv[i], "-c") == 0 )
			codebook_dir = string( argv[++i] );
		else if( strcmp(argv[i], "-n") == 0 )
			codeword_num = atoi( argv[++i] );
		else if( strcmp(argv[i], "-o") == 0 )
			output_dir = string( argv[++i] ) + "/";
		else if( strcmp(argv[i], "-s") == 0 )
			sigma = atof( argv[++i] );
		else if( strcmp(argv[i], "-st") == 0 )
			st_id = atoi( argv[++i] );
		else if( strcmp(argv[i], "-ed") == 0 )
			ed_id = atoi( argv[++i] );
		else
			usage();
	}
	
	if( ex_dir.length() == 0 || window_size < 0 || sliding_size < 0
		|| codebook_dir.length() == 0 || codeword_num < 0 || output_dir.length() == 0 ){
		cout << "!!! ERROR: Wrong parameter specification, check the usage." << endl;
		usage(); 
	}
	
	cout << ">> Directory of *_data.npy files: " << ex_dir << endl;
	cout << ">> Window size: " << window_size << endl;
	cout << ">> Sliding stride size: " << sliding_size << endl;
	cout << ">> Directory of codebook files: " << codebook_dir << endl;
	cout << ">> # of codewords per codebook: " << codeword_num << endl;
	cout << ">> Directory of output feature files: " << output_dir << endl;
	cout << ">> Smoothing parameter: " << sigma << " ("
		<< ((sigma <= 0) ? "HARD assignment" : "SOFT assignment") << ")" << endl;
	
	vector<string> ex_filenames;
	list_files(ex_dir + "/*_data.npy", ex_filenames);
	cout << ">> # of example files = " << ex_filenames.size() << endl;
	for(int i = 0; i < (int)ex_filenames.size(); i++)
		cout << ">> " << i << "th example file: " << ex_filenames[i] << endl;
	if( st_id >= 0 && ed_id >= 0 && st_id > ed_id ){
		cout << "!!! ERROR: Wrong parameter specification for the starting file ID ("
		<< st_id << ") and the ending file ID (" << ed_id << ")" << endl;
		return 0;
	}
	if( st_id >= (int)ex_filenames.size() || ed_id >= (int)ex_filenames.size() ){
		cout << "!!! ERROR: Inapproripate starting or ending file ID (st:"
			<< st_id << ", ed:" << ed_id << ") for " << ex_filenames.size() << " files" << endl;
		return 0;
	}
	if(st_id < 0)
		st_id = 0;
	if(ed_id < 0)
		ed_id = (int)ex_filenames.size() -1;
	cout << ">> Start file ID: " << st_id << ", End file ID: " << ed_id << endl;

	if(check_exist(output_dir.c_str()) == false)
		make_directory(output_dir);

	vector<string> codebook_filenames;
	list_files(codebook_dir + "/codebook*.txt", codebook_filenames);
	cout << ">> # of codebook files = " << codebook_filenames.size() << endl;
	
	// There are multiple codebooks under codebook_dir, so "early fusion" is used to
	// extract a feature for each codebook, and concatenate such features for all codebooks
	// into a single high-dimensional feature (like the case of OPPORTUNITY dataset)
	if(codebook_filenames.size() > 1){
		
		// For each sensor, a codebook (vector< vector<double> >) is stored 
		vector< vector< vector<double> > > codebooks;
		for(int i = 0; i < (int)codebook_filenames.size(); i++){
			// !!! It is needed to process codebooks in the ascending order of their IDs
			string codebook_filename = codebook_dir + "codebook_s" + to_string(i) + ".txt";
			cout << "### " << i << "th codebook: " << codebook_filename << " ###" << endl;
			vector< vector<double> > codebook;
			load_codebook(codebook_filename, codebook, codeword_num, window_size);
			codebooks.push_back(codebook);
			//if(i == 1) exit(0);
		}
		
		for(int i = st_id; i <= ed_id; i++){ // (int)ex_filenames.size(); i++){
			string output_filename = output_dir + ex_filenames[i].substr( ex_filenames[i].rfind("/") + 1 ); 
			cout << "##########" << endl;
			cout << "## Encoding examples in " << i << "th file" << endl;
			cout << "## Input: " << ex_filenames[i] << "\n## Output: " << output_filename << endl;
			cout << "##########" << endl;
			if(sigma <= 0)
				extract_feature_hard_assignment_multi(ex_filenames[i], output_filename,
												codebooks, window_size, sliding_size);
			else
				extract_feature_soft_assignment_multi(ex_filenames[i], output_filename,
												codebooks, window_size, sliding_size, sigma);
			//exit(0);
		}
		
	}
	// The following case assumes that there is only one codebook under codebook_dir.
	// That is, early fusion is not perfomred, like the caes of UniMiB-SHAR dataset 
	else if(codebook_filenames.size() == 1){
		
		cout << "### Load the codebook, " << codebook_filenames[0] << " ###" << endl;
		vector< vector<double> > codebook;
		load_codebook(codebook_filenames[0], codebook, codeword_num, window_size);
		
		for(int i = st_id; i <= ed_id; i++){
			string output_filename = output_dir + ex_filenames[i].substr( ex_filenames[i].rfind("/") + 1 ); 
			cout << "##########" << endl;
			cout << "## Encoding examples in " << i << "th file" << endl;
			cout << "## Input: " << ex_filenames[i] << "\n## Output: " << output_filename << endl;
			cout << "##########" << endl;
			if(sigma <= 0)
				extract_feature_hard_assignment_single(ex_filenames[i], output_filename,
												codebook, window_size, sliding_size);
			else
				extract_feature_soft_assignment_single(ex_filenames[i], output_filename,
												codebook, window_size, sliding_size, sigma);
			//exit(0);
		}
		
	}
	else{
		cout << "!!! ERROR: No codebook file (codebook*.txt) is found under " << codebook_dir << endl;
		return 0;
	}
	
	return 0;
	
}
