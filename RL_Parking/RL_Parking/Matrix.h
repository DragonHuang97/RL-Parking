// Matrix related functions

#include <vector>
#include <ctime>

using namespace std;

// matrix * column vector
vector<float> Multiply(vector<float> m, int row, int column, vector<float> v); 

// vector + vector
void Add(vector<float>& v, vector<float> v1);

// Leaky Relu
void LeakyRelu(vector<float>& v);

// Normal Distribution
float NormalDistribution(float mean, float standard_deviation);