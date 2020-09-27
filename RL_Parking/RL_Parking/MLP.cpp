#include "MLP.h"
#include"random"
#include <queue>
#include <utility>
#include <iostream>

extern float dt;
extern float runtime;
extern float v_max;
extern float omega_max;

extern float car_start[3]; // x, y, theta
extern float car_goal[2];// x, y

// ========================= MLP ============================

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

	for (int i = 0; i < hidden_num - 1; i++) { // intermediate layers
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
	LeakyRelu(result);

	// intermediate layers 
	for (int i = 0; i < hidden_num - 1; i++) { // (hidden_i -> hiddden_i+1)

		m_start = m_end + hidden_size[i];
		m_end = m_start + hidden_size[i + 1] * hidden_size[i];

		matrix = vector<float>(params.begin() + m_start, params.begin() + m_end);
		biases = vector<float>(params.begin() + m_end, params.begin() + m_end + hidden_size[i + 1]);

		result = Multiply(matrix, hidden_size[i + 1], hidden_size[i], result);
		Add(result, biases);
		LeakyRelu(result);

	}

	// last layer (last hidden -> output)
	if (hidden_num > 1) m_start = m_end + hidden_size[hidden_num - 2];
	else m_start = m_end + hidden_size[0];
	m_end = m_start + out_size * hidden_size[hidden_num - 1];

	matrix = vector<float>(params.begin() + m_start, params.begin() + m_end);
	biases = vector<float>(params.begin() + m_end, params.begin() + m_end + out_size);

	result = Multiply(matrix, out_size, hidden_size[hidden_num - 1], result);
	Add(result, biases);

	return result;
}

void MLP::set_params(vector<float> newparams) {
	// validity check
	if (newparams.size() != policy_size) {
		printf("new parameters have incorrect dimensions");
		exit(1);
	}
	// set new parameters
	params = newparams;
}

// ========================= CEM ============================

CEM::CEM() {
	n_iter = 200;
	batch_size = 30;
	elite_frac = 0.3f;
	ini_std = 1.0f;
	noise_factor = 1.0f;
}

CEM::CEM(int n, int bs, float ef, float ini_s, float nf) {
	n_iter = n;
	batch_size = bs;
	elite_frac = ef;
	ini_std = ini_s;
	noise_factor = nf;
}

void CEM::train(MLP& network) {
	vector<float> p_mean(network.policy_size, 0.0f); // the mean of each parameter
	vector<float> p_std(network.policy_size, ini_std); // the standard deviation of each parameter

	int n_elite = round(batch_size * elite_frac);
	for (int i = 0; i < n_iter; i++) {

		priority_queue<pair<float, vector<float>>> ranking;
		float mean_reward = 0.0f;

		// train
		for (int j = 0; j < batch_size; j++) {

			// sample a set of parameters
			vector<float> sample_params;
			for (int p = 0; p < network.policy_size; p++) {
				float sample = NormalDistribution(p_mean[p], p_std[p]);
				sample_params.push_back(sample);
			}

			// evaluate the set of parameters
			network.set_params(sample_params);
			float rwd = this->reward(network);
			ranking.push(make_pair(rwd, sample_params));
			
		}
		
		// move the elite to a vector...
		vector < pair<float, vector<float>> > elite;
		for (int j = 0; j < n_elite; j++) { 
			mean_reward += ranking.top().first;
			elite.push_back(ranking.top());
			ranking.pop();
		}

		mean_reward /= n_elite;

		// recompute new mean
		for (int j = 0; j < network.policy_size; j++) {
			float new_mean = 0.0f;
			for (auto e:elite) {
				new_mean += e.second[j];
			}
			new_mean /= n_elite;
			p_mean[j] = new_mean;
		}

		// recompute new std.dev.
		for (int j = 0; j < network.policy_size; j++) {
			float new_std = 0.0f;
			for (auto e : elite) {
				new_std += pow((e.second[j] - p_mean[j]), 2);
			}
			new_std /= n_elite;
			new_std = sqrt(new_std);
			p_std[j] = new_std;
		}

		// log traing process
		if ((i + 1) % 20 == 0 || i == 0) printf("Iter %i: mean reward: %6.3f\n", (i + 1), mean_reward);
	}
}


// ======================== reward ============================
/**/
float CEM::reward(MLP& network) {
	//float rwd = 0.0f;
	return run_task(network, state(car_start[0], car_start[1], car_start[2]), position(car_goal[0], car_goal[1]));
}


/*
float CEM::reward(MLP& network) {
	float rwd = 0.0f;
	//some toy reward for testing 
	for (int i = 0; i < 300; i++) {
		vector<float> input = { (float)i/50 };
		vector<float> res = network.run(input);
		//float ref = exp(i / 50.0f);
		float ref = sin(i / 50.0f);
		//float ref = (i)/3.0 - 5;
		rwd -= abs(res[0] - ref);
		//ref = exp(i / 50.0f);
		//rwd -= abs(res[1] - ref);
	}
	return rwd/300.0;
}
*/


// Car update (return new state)
state update_state(action action, state cur_state) {
	state new_state;
	// clip
	if (action.v > v_max) action.v = v_max;
	else if (action.v < -v_max) action.v = -v_max;
	if (action.omega > omega_max) action.omega = omega_max;
	else if (action.omega < -omega_max) action.omega = -omega_max;
	// update position
	new_state.pos.x = cur_state.pos.x + action.v * cos(cur_state.theta) * dt;
	new_state.pos.y = cur_state.pos.y + action.v * sin(cur_state.theta) * dt;
	// update orientation
	new_state.theta = cur_state.theta + action.omega * dt;
	return new_state;
}

// Task performance evaluation
float run_task(MLP network, state init_state, position goal_pos) {
	vector<state> state_list;
	vector<action> action_list;

	// task simulation
	state cur_state = init_state;
	state_list.push_back(cur_state);

	float reward = 0.0f;
	for (int i = 0; i< int(runtime/dt); i++) {
		vector<float> input = { cur_state.pos.x, cur_state.pos.y, cur_state.theta, goal_pos.x, goal_pos.y };
		vector<float> res = network.run(input);
		action a(res[0], res[1]);
		action_list.push_back(a);
		cur_state = update_state(a, state_list.back()); // update current state
		state_list.push_back(cur_state);
	}

	// performance evaluation
	state cs, ns;
	action a;
	for (int i = 0; i < action_list.size(); i++) {
		cs = state_list[i];
		a = action_list[i];
		ns = state_list[i + 1];
		// distance penalty
		reward -= distance(cs.pos, ns.pos);
		// large velocity and rotation penalty
		reward -= abs(a.omega);
		reward -= abs(a.v);
	}
	// final position reward
	if (distance(ns.pos, goal_pos) < 20) reward += 1000;
	if (distance(ns.pos, goal_pos) < 10 && a.v < 5) reward += 10000;
	return reward;
}