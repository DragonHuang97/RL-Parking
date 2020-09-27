// Matrix related functions

#include <vector>
#include <ctime>

using namespace std;

// matrix * column vector
vector<float> Multiply(vector<float> m, int row, int column, vector<float> v); 

// vector + vector
void Add(vector<float>& v, vector<float> v1);

// leaky Relu
void LeakyRelu(vector<float>& v);

// normal distribution random number generator
float NormalDistribution(float mean, float standard_deviation);



struct position {
	float x;
	float y;

	position();
	position(float px, float py);
};

struct state {
	position pos;
	float theta;

	state();

	state(float px, float py, float ptheta);
};

struct action {
	float v;
	float omega;

	action();

	action(float pv, float pomega);
};

float distance(position p1, position p2);