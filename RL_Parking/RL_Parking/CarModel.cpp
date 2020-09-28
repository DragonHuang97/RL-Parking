#include "CarModel.h"

// model parameters of the car
float frontWheel = 1.19f;
float backWheel = 1.12f;
// number of sensors
int nSensors = 0;
// damping
float damping = 1.0f;

car_state Update(car_action action, car_state prev_state)
{
    car_state new_state = prev_state;
    // symplectic euler integration
    float acceleration = action.accel;
    float delta = action.steer;
    new_state.v += dt * acceleration;
    // angle from the rear axle to the middle
    float beta = atan(tan(delta * 3.14f / 180.0f)
        * backWheel / (backWheel + frontWheel));
    new_state.omega = new_state.v * tan(delta * 3.14f / 180.0f)
        * cos(beta) / (backWheel + frontWheel) * 180.0f / 3.14f;
    Constrain(new_state);

    float vx = new_state.v * cos(new_state.theta);
    float vy = new_state.v * sin(new_state.theta);
    new_state.pos.x += vx * dt;
    new_state.pos.y += vy * dt;
    new_state.theta += new_state.omega * dt;

    new_state.delta = delta;
    // collect sensor data
    return new_state;
}

// keep the car on the ground and applied damping
void Constrain(car_state& new_state)
{
    //transform.position = new Vector3(transform.position.x, 0, transform.position.z);
    //transform.rotation = Quaternion.Euler(0, states[THETA], 0);
    new_state.v *= exp(-damping * dt);
}