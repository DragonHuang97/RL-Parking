// C++ implementation of bicycle model

#include "MLP.h"


// model parameters of the car
float frontWheel;
float backWheel;
// number of sensors
int nSensors;
// damping
float damping;

extern float dt;

car_state Update(car_action action, car_state prev_state);

void Constrain(car_state& new_state);
