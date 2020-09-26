#include"Matrix.h"
#include"MLP.h"
#include<iostream>

using namespace std;


// matrix multiplication test

int main() {
	MLP network(1, 1, {60, 60});
	CEM optimizer;
	optimizer.train(network);

	/*
	for (int i = 0; i < 30; i++) {
		cout << NormalDistribution(0.0f, 1.0f) << endl;
	}*/
}
