#pragma once
#include "unit.hpp"
#include <algorithm>
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
/*


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

*/
 
};

class ClutchComponent {
  friend class Vehicle;
  Force_Unit friction_capacity;  // 最大静摩擦力矩 Nm
  RSpeed_Unit slip_threshold_av; // 转速差低于此值才锁止

  //bool locked;
  Force_Unit transfer_torque; // 实际传递的扭矩

public:
};
class TransmissionComponent {
  friend class Vehicle;
  std::vector<double> gear_ratios; // 前进档
  double final_drive;
  double rev_ratio;

  int current_gear;
  bool is_shifting;
  Time_Unit shift_timer;
  int_fast16_t pending_gear = 0;

public:
  void request_shift(int new_gear) {
    if (is_shifting)
      return;
    is_shifting = true;
    shift_timer = 0.2; // 换档时间 200ms
                       // 通知离合器断开连接（通过外部回调）
    pending_gear = new_gear;
  }

  // virtual double get_current_ratio() const;
  double get_current_ratio() const {
    if (current_gear >= 1 &&
        current_gear <= static_cast<int>(gear_ratios.size()))
      return gear_ratios[current_gear - 1];
    return 0.0; // 空挡
  }
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
      //clutch.force_disengaged();
    }

    double ratio = get_current_ratio();
    double output_av = wheelspeed * ratio; // 变速箱输入轴转速
    // 提供给差速器的输出扭矩 = 离合器传递扭矩 * ratio
    // 同时计算反作用扭矩给离合器负载
  }
};
class DifferentialComponent {
  friend class Vehicle;
  enum Type { OPEN, LIMITED_SLIP, LOCKED };

  Type type;
  double preload;    // LSD 预紧扭矩
  double bias_ratio; // 扭矩分配比

public:
  // 在 engine.hpp 的 DifferentialComponent 类中，修改 distribute 函数
  void distribute(Force_Unit input_torque, double left_speed,
                  double right_speed, Force_Unit &left_torque,
                  Force_Unit &right_torque, Time_Unit dt) {
    if (type == OPEN) {
      // 开放差速器：扭矩平均分配，左右轮自由差速
      left_torque = input_torque * 0.5;
      right_torque = input_torque * 0.5;
    } else if (type == LIMITED_SLIP) {
      // 限滑差速器：预紧扭矩 + 转速差敏感
      double total = input_torque;
      double diff = (left_speed - right_speed) * preload;
      left_torque = (total * 0.5) - diff;
      right_torque = (total * 0.5) + diff;
      // 保证分配后不超出总扭矩范围
      left_torque = std::clamp(left_torque, -total, total);
      right_torque = total - left_torque;
    } else if (type == LOCKED) {
      // 锁止差速器：视为刚性连接，简单平均分配（或按更复杂的扭转弹簧模型）
      left_torque = input_torque * 0.5;
      right_torque = input_torque * 0.5;
      
    }
  }
  // LSD 类似离合器模型

}
;