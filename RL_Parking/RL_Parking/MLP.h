// Multi-layer Perceptron

#include<vector>

using namespace std;

class MLP
{
private:
	int in_size, out_size; // dimensions of input and output
	int hidden_num; // number of hidden

	vector<int> hidden_size; // size of each hidden layer
	vector<float> params; // parameters

	int policy_size; // total number of parameters

public:
	MLP(); // default MLP

	MLP(int insize, int outsize, vector<int> hidden_size); // customized MLP

	vector<float> run(vector<float> input_data); // run the model

};

