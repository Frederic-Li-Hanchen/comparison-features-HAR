
#ifndef DATASET_MULTI
#define DATASET_MULTI

#include <string>
#include <vector>

using namespace std;

void load_codebook(string codebook_filename, vector< vector<double> > &codebook, int nb_codewords, int window_size);
double logSumExp(vector<double> &v);
void extract_feature_soft_assignment_multi(string input_filename, string output_filename,
				vector< vector< vector<double> > > &codebooks, int window_size, int sliding_size, double sigma);
void extract_feature_hard_assignment_multi(string input_filename, string output_filename,
				vector< vector< vector<double> > > &codebooks, int window_size, int sliding_size);

#endif

