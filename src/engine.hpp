#pragma once
#include "unit.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>
class EngineComponent {
  Inertia_Unit crank_inertia; // 转动惯量 kg·m²
  Force_Unit max_torque;      // Nm
  RSpeed_Unit max_rpm;        // rad/s
  RSpeed_Unit idle_rpm;
  Force_Unit brake_torque; // 发动机制动 @ idle

  RSpeed_Unit current_rpm;
  Force_Unit input_torque;                          // 来自燃烧（油门）
  Force_Unit load_torque;                           // 来自离合器/附件
  const RAccelSpeed_Unit max_angular_accel = 150.0; // rad/s² (约 1430 RPM/s)
public:
  void update(Time_Unit dt) {
    Force_Unit net_torque = input_torque - load_torque;
    // 关键：限制最大角加速度

    RAccelSpeed_Unit angular_accel = net_torque / crank_inertia;
    angular_accel =
        std::clamp(angular_accel, -max_angular_accel, max_angular_accel);
    current_rpm += angular_accel * dt;
    // 转速上下界 + 断油逻辑
    if (current_rpm > max_rpm) {
      current_rpm = max_rpm;
      input_torque = 0.0; // 断油
    }
    if (current_rpm < 0.0)
      current_rpm = 0.0;
  }

  void set_load_torque(Force_Unit torque) { load_torque = torque; }
};

class ClutchComponent {
  Force_Unit friction_capacity;  // 最大静摩擦力矩 Nm
  RSpeed_Unit slip_threshold_av; // 转速差低于此值才锁止

  bool locked;
  Force_Unit transfer_torque; // 实际传递的扭矩

public:
  void update(RSpeed_Unit engine_av, RSpeed_Unit trans_input_av,
              Force_Unit engine_torque, Force_Unit trans_load_torque,
              Time_Unit dt) {
    RSpeed_Unit delta_av = engine_av - trans_input_av;

    // 打滑时传递扭矩受摩擦限制
    Force_Unit max_transfer = friction_capacity;
    if (!locked) {
      // 动态摩擦：可引入速度相关的摩擦系数下降
      max_transfer = friction_capacity *
                     (1.0 - 0.1 * std::min(1.0, std::abs(delta_av) / 100.0));
    }

    // 目标传递扭矩：尝试消除转速差
    Force_Unit desired_torque = engine_torque; // 理想情况是引擎扭矩完全传递
    transfer_torque = std::clamp(desired_torque, -max_transfer, max_transfer);

    // 锁止判断
    if (locked) {
      if (std::abs(desired_torque) >= max_transfer)
        locked = false;
    } else {
      if (std::abs(delta_av) < slip_threshold_av &&
          std::abs(desired_torque) < max_transfer)
        locked = true;
    }

    // 输出扭矩分别作用到引擎负载和变速箱输入轴
    // （由外部调用者应用）
  }
  virtual void force_disengaged();
};
class TransmissionComponent {
  std::vector<double> gear_ratios; // 前进档
  double final_drive;
  double rev_ratio;

  int current_gear;
  bool is_shifting;
  Time_Unit shift_timer;
  uint_fast16_t pending_gear;

public:
  void request_shift(int new_gear) {
    if (is_shifting)
      return;
    is_shifting = true;
    shift_timer = 0.2; // 换档时间 200ms
                       // 通知离合器断开连接（通过外部回调）
  }

  virtual double get_current_ratio();
  void update(Time_Unit dt, ClutchComponent &clutch, double engine_rpm,
              double wheelspeed) {
    if (is_shifting) {
      shift_timer -= dt;
      if (shift_timer <= 0.0) {
        // 换档完成，重新接合离合器
        current_gear = pending_gear;
        is_shifting = false;
        // 通知离合器可以尝试锁止
      }
      // 换档期间离合器打滑，引擎负载很小
      clutch.force_disengaged();
    }

    double ratio = get_current_ratio();
    double output_av = wheelspeed * ratio; // 变速箱输入轴转速
    // 提供给差速器的输出扭矩 = 离合器传递扭矩 * ratio
    // 同时计算反作用扭矩给离合器负载
  }
};
class DifferentialComponent {
  enum Type { OPEN, LIMITED_SLIP, LOCKED };

  Type type;
  double preload;    // LSD 预紧扭矩
  double bias_ratio; // 扭矩分配比

public:
  void distribute(Force_Unit input_torque, double left_speed,
                  double right_speed, Force_Unit &left_torque,
                  Force_Unit &right_torquem, Time_Unit dt) {
    if (type == LOCKED) {
      // 锁止差速器本质上是一个刚性连接，需要计算总扭矩然后按比例分配
      // 但为了避免数值爆炸，我们模拟一个高刚度扭转弹簧
      const double torsional_stiffness = 50000.0; // Nm/(rad/s)
      double delta_speed = left_speed - right_speed;
      double lock_torque = torsional_stiffness * delta_speed * dt; // 实际要积分
      // 然后分配到左右轮...
    }
    // LSD 类似离合器模型
  }
};