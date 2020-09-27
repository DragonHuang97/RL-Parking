#include"MLP.h"
#include<iostream>

using namespace std;

float dt = 0.1f;
float runtime = 8.0f;

float car_start[3] = { -50.0f, 0.0f, 0.751f }; // x, y, theta
float car_goal[2] = { 50.0f, 0.0f };// x, y

float v_max = 80.0f;
float omega_max = 3.14f;

float car_w = 5;
float car_l = 10;





int main() {
	srand((int)time(0));

	MLP network(5, 2, { 60, 60 });// input (cx, cy, theta, gx, gy) output (v, omega)
	CEM optimizer;

	optimizer.train(network);
}



/*// simple learning test
int main() {

	srand((int)time(0));

	MLP network(1, 2, {60, 60});
	CEM optimizer;
	optimizer.train(network);
}
*/