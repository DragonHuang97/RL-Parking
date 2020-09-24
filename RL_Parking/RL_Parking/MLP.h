/* Neural Networks */

#include<vector>

using namespace std;

// =================== multilayer perceptron class ===================
class MLP
{
private:
	int in_size, out_size; // dimensions of input and output
	int hidden_num; // number of hidden

	vector<int> hidden_size; // size of each hidden layer

public:
	vector<float> params; // parameters
	int policy_size; // total number of parameters

	MLP(); // default MLP

	MLP(int insize, int outsize, vector<int> hidden_size); // customized MLP

	vector<float> run(vector<float> input_data); // run the model

};


// ============== cross-entropy method optimizer class ====================

class CEM {
private:
	int n_iter; // total number of interations
	int batch_size;
	float elite_frac;
	float ini_std; // initial standard deviation

public:
	CEM(); // default CEM implementation
	
	CEM(int n_iter, int batch_size, float elite_frac, float ini_std); // customized CEM

	void train(MLP& network);
};

// ================================= reward ========================================
