#include "Matrix.h"

vector<float> Multiply(vector<float> m, int row, int column, vector<float> v) {

	// input validity check
	if (row * column != m.size()) {
		printf("incorrect matrix dimensions");
		exit(1);
	}
	else if (column != v.size()) {
		printf("matrix dimensions incompatible with vector dimensions");
		exit(1);
	}

	// matrix multiplication
	vector<float> result;
	for (int i = 0; i < row; i++) {
		float tmp_result = 0;
		for (int j = 0; j < column; j++) tmp_result += (m[i * column + j] * v[j]);
		result.push_back(tmp_result);
	}

	return result;
}

// no return value, addition result directly applies to the first vector
void Add(vector<float>& v, vector<float> v1) {

	// input validity check
	if (v.size() != v1.size()) {
		printf("incompatible vector dimensions");
		exit(1);
	}

	// vector addition
	for (int i = 0; i < v.size(); i++) {
		v[i] += v1[i];
	}
}