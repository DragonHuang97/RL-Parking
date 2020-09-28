#include"MLP.h"
#include<iostream>

using namespace std;

float dt = 0.1f;
float runtime = 8.0f;

float car_start[3] = { -50.0f, 0.0f, 0.751f }; // x, y, theta
float car_goal[2] = { 25.0f, 25.0f };// x, y

float v_max = 80.0f;
float omega_max = 3.14f;

float car_w = 5;
float car_l = 10;





int main() {
	srand((int)time(0));

	MLP network(5, 2, { 60, 60 });// input (cx, cy, theta, gx, gy) output (v, omega)
	CEM optimizer(300, 50, 0.5f, 1.0f, 1.0f, 10);// n_iter, batch_size, elite_frac, ini_std, noise_fac, print_rate

	//run_task(network, state(car_start[0], car_start[1], car_start[2]), position(car_goal[0], car_goal[1]));

	optimizer.train(network);

	//run_task(network, state(car_start[0], car_start[1], car_start[2]), position(car_goal[0], car_goal[1]));
}


/*
// simple learning test
int main() {

	srand((int)time(0));

	MLP network(1, 1, {60, 60});
	CEM optimizer;
	//vector<float> input = { 100 };
	//cout << network.run(input)[0] << endl;
	optimizer.train(network);
	//cout << network.run(input)[0] << endl;
	for (int i = 0; i < 300; i++) {
		vector<float> input = { float(i)/50 };
		cout << sin(float(i) / 50) << endl;
		cout << network.run(input)[0] << endl << endl;
	}
}
*/