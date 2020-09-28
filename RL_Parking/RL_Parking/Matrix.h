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

struct vec3 {
	float x;
	float y;
	float z;

	vec3();
	vec3(float x, float y, float z);
};

struct position {
	float x;
	float y;

	position();
	position(float x, float y);
};

struct state {
	position pos;
	float theta;

	state();

	state(float x, float y, float theta);
};

struct car_state {
	position pos;
	float v;
	float omega;
	float theta;
	float delta;

	car_state();
	
	car_state(position position, float velocity, float omega, float theta, float delta);
};

struct action {
	float v;
	float omega;

	action();

	action(float v, float omega);
};

struct car_action {
	float accel;
	float steer;

	car_action();

	car_action(float acceleration, float steer);
};

float distance(position p1, position p2);