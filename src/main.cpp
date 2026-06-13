// main.cpp (演示使用BDF2迭代器)
#include "vehicle.hpp"
#include <cstdio>

int main() {
  Vehicle car;
  BDF2Integrator integrator;
  ControlInput input;
  input.throttle = 0.3;
  input.brake = 0.0;
  input.steering = 0.0;
  input.clutch_pedal = 1.0;
  input.gear_request = 1;

  Time_Unit dt = 0.01;
  for (int step=0; step<5000; ++step) {
    car.set_control(input);
    integrator.step(car, dt);
    if (step % 500 == 0) {
      VehicleState state;
      car.get_state(state);
      printf("t=%.2f s, v=%.2f m/s, rpm=%.1f\n", step*dt, state.vehicle_speed, state.engine_rpm);
    }
  }
  return 0;
}