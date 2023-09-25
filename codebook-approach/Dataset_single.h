
#ifndef DATASET_SINGLE
#define DATASET_SINGLE

#include <string>
#include <vector>

using namespace std;

void extract_feature_hard_assignment_single(string input_filename, string output_filename,
									vector< vector<double> > &codebook, int window_size, int sliding_size);
void extract_feature_soft_assignment_single(string input_filename, string output_filename,
				vector< vector<double> > &codebook, int window_size, int sliding_size, double sigma);


#endif

