// vehicle.hpp (新增核心类)
#pragma once
#include "engine.hpp"
#include "suspension.hpp"
#include "tire.hpp"
#include <array>
#include <cmath>

// 车辆状态结构体，用于数据交换
struct VehicleState {
  double engine_rpm;          // rad/s
  double vehicle_speed;       // m/s
  double yaw_rate;            // rad/s
  std::array<double, 4> wheel_angular_velocity; // rad/s
  std::array<double, 4> wheel_vertical_load;    // N
  double x, y, yaw;           // 位置与航向

  void print() const {
    
    // 调试用
  }
};

// 控制输入结构体
struct ControlInput {
  double throttle;            // 0~1
  double brake;               // 0~1
  double steering;            // rad, 前轮转向角
  double clutch_pedal;        // 0=分离,1=接合
  int gear_request;           // 请求档位，0表示空档，负数倒车
};

// 抽象迭代器接口
class Integrator {
public:
  virtual ~Integrator() = default;
  virtual void step(class Vehicle &vehicle, Time_Unit dt) = 0;
};

// 前置声明
class Vehicle {
public:
  Vehicle();
  void set_control(const ControlInput &input);
  void get_state(VehicleState &out) const;
  void set_state(const VehicleState &in);
  void exchangeState(VehicleState &state); // 对外交换数据（读+写）

  // 计算状态导数（用于积分器）
  void compute_derivatives(const VehicleState &state, VehicleState &deriv);

  // 内部更新，被迭代器调用
  void update_dynamics(Time_Unit dt);

private:
  // 动力总成组件
  EngineComponent engine;
  ClutchComponent clutch;
  TransmissionComponent transmission;
  DifferentialComponent diff;

  // 车轮组件（每个车轮独立悬挂+轮胎）
  struct Wheel {
    SuspensionComponent susp;
    PacejkaTire tire;
    double angular_velocity; // rad/s
    double radius;           // m
    double inertia;          // kg·m²
    double brake_torque;     // Nm, 来自制动系统
    double load;             // N, 垂直载荷
    Length_Unit ride_height; // 悬挂当前长度

    Wheel() : angular_velocity(0), radius(0.33), inertia(1.2), brake_torque(0),
              load(0), ride_height(0.35) {}
  };
  std::array<Wheel, 4> wheels; // 0:FL, 1:FR, 2:RL, 3:RR

  // 车身参数
  double mass;                 // kg
  double yaw_inertia;          // kg·m²
  double wheel_base;           // m
  double track_width;          // m
  double cg_height;            // m

  // 气动阻力
  double drag_coeff;
  double frontal_area;
  double air_density;

  // 当前状态
  VehicleState state;
  VehicleState state_prev;     // 用于BDF2的历史状态
  VehicleState deriv;          // 当前导数缓存
  ControlInput control;

  // 辅助计算函数
  void update_wheel_loads();
  void update_tire_forces(std::array<double,4>& Fx, std::array<double,4>& Fy, std::array<double,4>& Mz);
  void apply_drivetrain_torques(std::array<double,4>& drive_torques);
  void update_ride_heights();
};

// ---------- BDF2 积分器实现 ----------
class BDF2Integrator : public Integrator {
public:
  BDF2Integrator() : first_step(true) {}
  void step(Vehicle &vehicle, Time_Unit dt) override {
    VehicleState state0, deriv0;
    vehicle.get_state(state0);
    vehicle.compute_derivatives(state0, deriv0);

    if (first_step) {
      // 第一步使用改进欧拉（Heun）法启动
      VehicleState k1 = deriv0;
      VehicleState temp = state0;
      temp.engine_rpm += k1.engine_rpm * dt;
      temp.vehicle_speed += k1.vehicle_speed * dt;
      temp.yaw_rate += k1.yaw_rate * dt;
      for (int i=0;i<4;++i) temp.wheel_angular_velocity[i] += k1.wheel_angular_velocity[i] * dt;
      temp.x += k1.x * dt;
      temp.y += k1.y * dt;
      temp.yaw += k1.yaw * dt;

      vehicle.set_state(temp);
      VehicleState k2;
      vehicle.compute_derivatives(temp, k2);
      VehicleState new_state = state0;
      new_state.engine_rpm += (k1.engine_rpm + k2.engine_rpm) * 0.5 * dt;
      new_state.vehicle_speed += (k1.vehicle_speed + k2.vehicle_speed) * 0.5 * dt;
      new_state.yaw_rate += (k1.yaw_rate + k2.yaw_rate) * 0.5 * dt;
      for (int i=0;i<4;++i) new_state.wheel_angular_velocity[i] += (k1.wheel_angular_velocity[i] + k2.wheel_angular_velocity[i]) * 0.5 * dt;
      new_state.x += (k1.x + k2.x) * 0.5 * dt;
      new_state.y += (k1.y + k2.y) * 0.5 * dt;
      new_state.yaw += (k1.yaw + k2.yaw) * 0.5 * dt;
      vehicle.set_state(new_state);
      vehicle.get_state(state_prev);
      first_step = false;
    } else {
      // BDF2: x_{n+1} = 4/3 x_n - 1/3 x_{n-1} + 2/3 dt f(t_{n+1}, x_{n+1})
      // 采用一次固定点迭代
      VehicleState state_n, state_nm1;
      vehicle.get_state(state_n);
      state_nm1 = state_prev;

      VehicleState pred = state_n;
      // 预测步: 显式欧拉外推
      for (int i=0;i<4;++i) pred.wheel_angular_velocity[i] += deriv0.wheel_angular_velocity[i] * dt;
      pred.engine_rpm += deriv0.engine_rpm * dt;
      pred.vehicle_speed += deriv0.vehicle_speed * dt;
      pred.yaw_rate += deriv0.yaw_rate * dt;
      pred.x += deriv0.x * dt;
      pred.y += deriv0.y * dt;
      pred.yaw += deriv0.yaw * dt;

      vehicle.set_state(pred);
      VehicleState deriv_pred;
      vehicle.compute_derivatives(pred, deriv_pred);

      VehicleState new_state;
      new_state.engine_rpm = (4.0/3.0)*state_n.engine_rpm - (1.0/3.0)*state_nm1.engine_rpm + (2.0/3.0)*dt*deriv_pred.engine_rpm;
      new_state.vehicle_speed = (4.0/3.0)*state_n.vehicle_speed - (1.0/3.0)*state_nm1.vehicle_speed + (2.0/3.0)*dt*deriv_pred.vehicle_speed;
      new_state.yaw_rate = (4.0/3.0)*state_n.yaw_rate - (1.0/3.0)*state_nm1.yaw_rate + (2.0/3.0)*dt*deriv_pred.yaw_rate;
      for (int i=0;i<4;++i)
        new_state.wheel_angular_velocity[i] = (4.0/3.0)*state_n.wheel_angular_velocity[i] - (1.0/3.0)*state_nm1.wheel_angular_velocity[i] + (2.0/3.0)*dt*deriv_pred.wheel_angular_velocity[i];
      new_state.x = (4.0/3.0)*state_n.x - (1.0/3.0)*state_nm1.x + (2.0/3.0)*dt*deriv_pred.x;
      new_state.y = (4.0/3.0)*state_n.y - (1.0/3.0)*state_nm1.y + (2.0/3.0)*dt*deriv_pred.y;
      new_state.yaw = (4.0/3.0)*state_n.yaw - (1.0/3.0)*state_nm1.yaw + (2.0/3.0)*dt*deriv_pred.yaw;

      vehicle.set_state(new_state);
      vehicle.get_state(state_prev);
    }
    // 最后执行内部物理更新以保证组件间数据一致
    vehicle.update_dynamics(dt);
  }

private:
  bool first_step;
  VehicleState state_prev;
};