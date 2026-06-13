#pragma once
#include "unit.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
class EngineComponent {
  friend class Vehicle;
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
  EngineComponent()
      : crank_inertia(0.2), max_torque(300), max_rpm(600.0), idle_rpm(80.0),
        brake_torque(50), current_rpm(idle_rpm), input_torque(0),
        load_torque(0) {}

  void set_throttle(double pedal) { // pedal 0~1
    // 简化的扭矩-转速曲面：峰值扭矩在中间转速，怠速附近扭矩下降
    double rpm_ratio = current_rpm / max_rpm;
    double torque_factor = std::max(0.0, std::sin(rpm_ratio * M_PI));
    input_torque = pedal * max_torque * torque_factor;
  }

  void set_load_torque(Force_Unit torque) { load_torque = torque; }
  RSpeed_Unit get_rpm() const { return current_rpm; }

  void update(Time_Unit dt) {
    Force_Unit net_torque = input_torque - load_torque;
    RAccelSpeed_Unit angular_accel = net_torque / crank_inertia;
    angular_accel =
        std::clamp(angular_accel, -max_angular_accel, max_angular_accel);
    current_rpm += angular_accel * dt;

    if (current_rpm > max_rpm) {
      current_rpm = max_rpm;
      input_torque = 0.0; // 断油
    }
    if (current_rpm < idle_rpm && load_torque > 0) {
      // 怠速控制：补油避免熄火
      current_rpm = idle_rpm;
      if (net_torque < 0)
        input_torque += brake_torque * 0.1f * dt;
    }
    if (current_rpm < 0.0)
      current_rpm = 0.0;
  }
};

class ClutchComponent {
  friend class Vehicle;
  Force_Unit friction_capacity;   // 最大静摩擦力矩 Nm
  RSpeed_Unit slip_threshold_av;  // 转速差低于此值才锁止
  Force_Unit transfer_torque;     // 实际传递的扭矩
  double pedal_position;          // 0=分离, 1=接合

public:
  ClutchComponent()
      : friction_capacity(400), slip_threshold_av(2.0), transfer_torque(0),
        pedal_position(0) {}

  void set_pedal(double pos) { pedal_position = std::clamp(pos, 0.0, 1.0); }

  void update(Time_Unit dt, RSpeed_Unit engine_rpm,
              RSpeed_Unit input_shaft_rpm) {
    double rpm_diff = std::abs(engine_rpm - input_shaft_rpm);
    // 滑动摩擦扭矩随压紧力增大而增大，随转速差增大略微下降
    double max_transfer = friction_capacity * pedal_position;
    if (rpm_diff < slip_threshold_av && pedal_position > 0.99) {
      // 锁止状态：传递扭矩无滑差，但受限于最大静摩擦
      transfer_torque = max_transfer;
    } else {
      // 滑动摩擦模型
      double slip_factor = std::exp(-0.1 * rpm_diff);
      transfer_torque = max_transfer * slip_factor;
    }
    // 限制传递扭矩不超过输入扭矩+负载需求等，实际由外部平衡
  }

  Force_Unit get_transfer_torque() const { return transfer_torque; }
};
class TransmissionComponent {
  friend class Vehicle;
  std::vector<double> gear_ratios; // 前进档
  double final_drive;
  double rev_ratio;

  int current_gear;
  bool is_shifting;
  Time_Unit shift_timer;
  int_fast16_t pending_gear;

public:
   TransmissionComponent() : gear_ratios{3.5, 2.0, 1.3, 1.0, 0.8}, final_drive(3.2),
                            rev_ratio(3.0), current_gear(1), is_shifting(false),
                            shift_timer(0), pending_gear(1) {}

  void request_shift(int new_gear) {
    if (is_shifting) return;
    is_shifting = true;
    shift_timer = 0.2; // 200ms
    pending_gear = new_gear;
  }

  double get_current_ratio() const {
    if (current_gear >= 1 && current_gear <= static_cast<int>(gear_ratios.size()))
      return gear_ratios[current_gear - 1] * final_drive;
    return 0.0; // 空挡
  }

  double get_reverse_ratio() const { return rev_ratio * final_drive; }

  void update(Time_Unit dt, ClutchComponent &clutch, double engine_rpm,
              double wheel_angular_speed) {
    if (is_shifting) {
      shift_timer -= dt;
      if (shift_timer <= 0.0) {
        current_gear = pending_gear;
        is_shifting = false;
        // 换挡完成后可重新接合离合器（外部控制逻辑会重置离合器踏板）
      }
    }
    // 变速箱输入轴转速 = 输出轴转速(车轮转速换算到变速箱输出端) * 当前档位传动比?
    // 实际从车轮角速度反推: 变速箱输出轴转速 = wheel_angular_speed * final_drive?
    // 需要根据动力传递路径精确建模，此处简化：变速箱输入轴转速 = 车轮角速度 * get_current_ratio()
    // 输出扭矩给差速器在 Vehicle 层处理
  }
};
class DifferentialComponent {
  friend class Vehicle;
  enum Type { OPEN, LIMITED_SLIP, LOCKED };

  Type type;
  double preload;    // LSD 预紧扭矩
  double bias_ratio; // 扭矩分配比


public:
   DifferentialComponent() : type(OPEN), preload(100), bias_ratio(1.2) {}

  void distribute(Force_Unit input_torque, double left_speed, double right_speed,
                  Force_Unit &left_torque, Force_Unit &right_torque, Time_Unit dt) {
    if (type == OPEN) {
      left_torque = input_torque * 0.5;
      right_torque = input_torque * 0.5;
    } else if (type == LIMITED_SLIP) {
      double total = input_torque;
      double diff = (left_speed - right_speed) * preload;
      left_torque = (total * 0.5) - diff;
      right_torque = (total * 0.5) + diff;
      left_torque = std::clamp(left_torque, -total, total);
      right_torque = total - left_torque;
    } else { // LOCKED
      left_torque = input_torque * 0.5;
      right_torque = input_torque * 0.5;
    }
  }
};