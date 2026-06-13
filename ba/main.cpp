#include "vehicle.hpp"
#include <cstdio>

int main() {
    Vehicle car;
    car.set_throttle(0.3);
    car.set_steering(0.1);
    const double dt = 0.02;
    car.set_steering(0.0);
    int g_time = 0;
    double spd,prev_spd = 0.0;
    bool once_flag = false;
    car.request_shift(1);
    for (int step = 0; step < 9999; ++step) {
        spd = car.get_state().vx;
        if (g_time!=10) {
          g_time++;
        }else {
        g_time = 0;
        }
        car.update(dt);
        auto s = car.get_state();
        if (g_time == 2) {
           printf("t=%.2f vx=%.5f engine=%.4f,drvt=%.2f,", step*dt, s.vx, s.engine_rpm,s.net_drive_torque);
           //car.request_shift(2);
           printf("w_rpm=%f \n",car.get_state().wheel_fr_rpm);
        }
        if (spd-prev_spd<0.0005 && step>2000 && once_flag==false) {
          car.request_shift(2);
          printf("shifted!\n");
          once_flag=true;
        }
        prev_spd = spd;
    }
    return 0;
}