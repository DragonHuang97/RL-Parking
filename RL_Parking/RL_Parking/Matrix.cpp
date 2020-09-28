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
		for (int j = 0; j < column; j++) {
			tmp_result += (m[i * column + j] * v[j]);
		}
		result.push_back(tmp_result);
	}

	return result;
}

// addition of two vectors
// no return value, addition result directly applied to the first vector
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

// apply the leaky relu function
// no return value, result directly applies to the input vector
void LeakyRelu(vector<float>& v) {
	for (int i = 0; i < v.size(); i++) {
		v[i] = v[i] * (v[i] > 0) + 0.1 * v[i] * (v[i] < 0);
	}
}

// draw random number from normal distribution
float NormalDistribution(float mean, float standard_deviation) {
	float v1 = (static_cast <float> (rand()) + 1.0f) / (static_cast <float> (RAND_MAX) + 1.0f);
	float v2 = (static_cast <float> (rand()) + 1.0f) / (static_cast <float> (RAND_MAX) + 1.0f);
	float normal_random = cos(2 * 3.14 * v2) * sqrt(-2. * log(v1));
	return normal_random * standard_deviation + mean;
}



vec3::vec3() {
	x = 0.0f; y = 0.0f; z = 0.0f;
}

vec3::vec3(float px, float py, float pz) {
	x = px; y = py; z = pz;
}


position::position() {
	x = 0.0f; y = 0.0f;
}
position::position(float px, float py) {
	x = px;
	y = py;
}


state::state() {
	pos = position(0.0f, 0.0f);
	theta = 0.0f;
}
state::state(float px, float py, float ptheta) {
	pos = position(px, py);
	theta = ptheta;
}

car_state::car_state() {
	pos = position(0.0f, 0.0f);
	v = 0.0f;
	omega = 0.0f;
	theta = 0.0f;
	delta = 0.0f;
}


car_state::car_state(position pp, float pv, float po, float pt, float pd) {
	pos = pp;
	v = pv;
	omega = po;
	theta = pt;
	delta = pd;
}


action::action() {
	v = 0.0f;
	omega = 0.0f;
}

action::action(float pv, float pomega) {
	v = pv;
	omega = pomega;
}




car_action::car_action() {
	accel = 0.0f; 
	steer = 0.0f;
}

car_action::car_action(float acc, float st) {
	accel = acc; 
	steer = st;
}



float distance(position p1, position p2) {
	float dx = p1.x - p2.x;
	float dy = p1.y - p2.y;
	return sqrt(dx * dx + dy * dy);
}