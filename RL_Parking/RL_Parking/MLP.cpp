#include "MLP.h"
#include"Matrix.h"

MLP::MLP() {

	in_size = 1; // scalar input
	out_size = 1; // scalar output

	hidden_num = 1; // 1 hiddenlayer
	hidden_size = { 40 }; // 40 neurons in the hidden layer

	for (int i = 0; i < (in_size + 1) * hidden_size[0]; i++) params.push_back(0.0f); // first layer params
	for (int i = 0; i < (hidden_size[0] + 1) * out_size; i++) params.push_back(0.0f); // last layer params
	
	policy_size = params.size(); // total number of params
}

MLP::MLP(int i_size, int o_size, vector<int> h_size) {
	in_size = i_size;
	out_size = o_size;

	hidden_num = h_size.size();
	hidden_size = h_size;

	for (int i = 0; i < (in_size + 1) * hidden_size[0]; i++) params.push_back(0.0f); // layer 1 params

	for (int i = 0; i < hidden_num; i++) { // intermediate layers
		for (int j = 0; j < (hidden_size[i] + 1) * hidden_size[i + 1]; j++) params.push_back(0.0f);
	}

	for (int i = 0; i < (hidden_size[hidden_num - 1] + 1) * out_size; i++) params.push_back(0.0f); // layer 2 params

	policy_size = params.size();

}

vector<float> MLP::run(vector<float> in_data) {

	vector<float> result;

	vector<float> matrix;
	vector<float> biases;
	int m_start, m_end; // indices to keep track of

	// first layer (input -> hidden0)
	m_end = hidden_size[0] * in_size;

	matrix = vector<float>(params.begin(), params.begin() + m_end);
	biases = vector<float>(params.begin() + m_end, params.begin() + m_end + hidden_size[0]);

	result = Multiply(matrix, hidden_size[0], in_data.size(), in_data);
	Add(result, biases);

	// intermediate layers 
	for (int i = 0; i < hidden_num - 1; i++) { // (hidden_i -> hiddden_i+1)

		m_start = m_end + hidden_size[i];
		m_end = m_start + hidden_size[i + 1] * hidden_size[i];

		matrix = vector<float>(params.begin() + m_start, params.begin() + m_end);
		biases = vector<float>(params.begin() + m_end, params.begin() + m_end + hidden_size[i + 1]);

		result = Multiply(matrix, hidden_size[i + 1], hidden_size[i], result);
		Add(result, biases);

	}

	// last layer (last hidden -> output)
	m_start = m_end + hidden_size[hidden_num - 2];
	m_end = m_start + out_size * hidden_size[hidden_num - 1];

	matrix = vector<float>(params.begin(), params.begin() + m_end);
	biases = vector<float>(params.begin() + m_end, params.begin() + m_end + hidden_size[hidden_num - 1]);

	result = Multiply(matrix, out_size, hidden_size[hidden_num - 1], result);
	Add(result, biases);


	return result;
}