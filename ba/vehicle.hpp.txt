// vehicle.hpp
#pragma once
#include "engine.hpp"
#include "suspension.hpp"
#include "tire.hpp"
#include "unit.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

// -------------------------------------------------------------------
// 车辆状态结构体（只包含需要持久化的量）
// -------------------------------------------------------------------
struct VehicleState {
  // 运动学
  Speed_Unit vx = 0.0;        // 纵向速度 (m/s)
  Speed_Unit vy = 0.0;        // 侧向速度 (m/s)
  Speed_Unit ax = 0.0;        // 纵向加速度 (m/s²) - 用于载荷转移
  Speed_Unit ay = 0.0;        // 侧向加速度 (m/s²)
  RSpeed_Unit yaw_rate = 0.0; // 横摆角速度 (rad/s)
  double yaw_angle = 0.0;     // 横摆角 (rad)

  // 动力传动
  RSpeed_Unit engine_rpm = 0.0; // 发动机转速 (rad/s)
  RSpeed_Unit wheel_fl_rpm = 0.0;
  RSpeed_Unit wheel_fr_rpm = 0.0;
  RSpeed_Unit wheel_rl_rpm = 0.0;
  RSpeed_Unit wheel_rr_rpm = 0.0;

  // 变速箱状态
  int current_gear = 1;
  bool is_shifting = false;

  // 可选：半轴扭转角 (rad) 用于传动弹性
  double torsion_fl = 0.0;
  double torsion_fr = 0.0;
  double torsion_rl = 0.0;
  double torsion_rr = 0.0;

  // 调试/输出用（不参与积分）
  Force_Unit net_drive_torque = 0.0;
  std::array<Force_Unit, 4> fx = {0.0};
  std::array<Force_Unit, 4> fy = {0.0};
  std::array<Force_D_Length_Unit, 4> mz = {0.0};
  std::array<Length_Unit, 4> suspension_compression = {0.0};
};

// -------------------------------------------------------------------
// 车辆主类
// -------------------------------------------------------------------
class Vehicle {
private:
  double last_clutch_delta_dont_use = 0.0;

public:
  Vehicle() {
    // ---------- 发动机参数 ----------
    engine_.max_torque = 300.0;  // Nm
    engine_.max_rpm = 600.0;     // rad/s (~5700 rpm)
    engine_.idle_rpm = 75.0;     // rad/s (~716 rpm)
    engine_.brake_torque = 30.0; // Nm
    engine_.crank_inertia = 0.4; // kg·m²

    // ---------- 离合器参数 ----------
    clutch_.friction_capacity = 400.0; // Nm
    clutch_.slip_threshold_av = 5.0;   // rad/s
    // clutch_.locked = true;

    // ---------- 变速箱 ----------
    transmission_.gear_ratios = {3.5, 2.0, 1.3, 1.0, 0.8};
    transmission_.final_drive = 3.2;
    transmission_.rev_ratio = -3.2;
    transmission_.current_gear = 1;
    transmission_.is_shifting = false;

    // ---------- 差速器 ----------
    differential_.type = DifferentialComponent::OPEN;
    differential_.preload = 0.0; // LSD预紧扭矩
    differential_.bias_ratio = 1.0;

    // ---------- 悬架 ----------
    for (auto &susp : suspensions_) {
      susp.rest_length = 0.3;
      susp.stiffness = 30000.0;
      susp.bump_damping = 2500.0;
      susp.rebound_damping = 3500.0;
      susp.anti_roll_stiffness = 5000.0;
    }

    // ---------- 轮胎 ----------
    PacejkaParameters tire_params;
    tire_params.Bx = 12.0;
    tire_params.Cx = 1.6;
    tire_params.Dx = 1.0;
    tire_params.By = 10.0;
    tire_params.Cy = 1.3;
    tire_params.Dy = 1.0;
    tire_params.Bz = 8.0;
    tire_params.Cz = 2.0;
    tire_params.Dz = 0.08;
    for (auto &tire : tires_)
      tire = PacejkaTire(tire_params);
    // 侧偏刚度随载荷变化系数（经验值）
    tire_stiffness_load_factor_ = 0.2;

    // ---------- 整车参数 ----------
    mass_ = 1500.0;             // kg
    yaw_inertia_ = 2500.0;      // kg·m²
    wheel_radius_ = 0.32;       // m
    wheel_inertia_ = 1.2;       // kg·m² per wheel
    cda_ = 0.65;                // 空气阻力系数 × 迎风面积 (m²)
    air_density_ = 1.225;       // kg/m³
    brake_max_torque_ = 2000.0; // Nm (四轮总计)
    distribution_fr_ = 0.6;     // 前轴制动力比例

    // 几何参数
    wheelbase_front_ = 1.2;     // 前轴到质心 (m)
    wheelbase_rear_ = 1.4;      // 后轴到质心 (m)
    track_ = 1.6;               // 轮距 (m)
    roll_center_height_ = 0.45; // 侧倾中心高度 (m)
    cg_height_ = 0.55;          // 质心高度 (m)

    // 驱动形式：前后轴分配比例 (0=后驱, 1=前驱)
    torque_split_front_ = 1.0;    // 1.0 = 前驱, 0.0 = 后驱, 0.5 = 全驱50:50
    driven_axles_ = {true, true}; // 前轴驱动, 后轴驱动

    // 半轴扭转刚度 (Nm/rad)
    halfshaft_stiffness_ = 20000.0;

    // 路面附着
    road_mu_ = 0.9;

    // 发动机角加速度限制
    max_engine_accel_ = 150.0; // rad/s²

    // ---------- 初始状态 ----------
    state_ = VehicleState();
    state_.engine_rpm = engine_.idle_rpm;
    state_.current_gear = 1;
  }

  // 驾驶员输入
  void set_throttle(double pedal) { throttle_ = std::clamp(pedal, 0.0, 1.0); }
  void set_brake(double pedal) { brake_ = std::clamp(pedal, 0.0, 1.0); }
  void set_steering(double angle) {
    steering_angle_ = std::clamp(angle, -0.8, 0.8);
  }
  void request_shift(int new_gear) { transmission_.request_shift(new_gear); }

  // 获取当前状态（只读拷贝）
  VehicleState get_state() const { return state_; }

  // 固定步长更新（纯函数式，无顺序依赖）
  void update(Time_Unit dt) {
    dt = std::min(dt, 0.05);
    if (dt <= 0.0)
      return;

    // 1. 保存旧状态快照（所有计算只读此副本）
    const VehicleState old = state_;

    // 2. 基于旧状态计算所有力和扭矩（纯计算，不修改成员变量）
    // ------------------------------------------------------------
    // 2.1 轮胎垂直载荷（包含纵向和侧向载荷转移）
    std::array<Force_Unit, 4> normal_loads;
    compute_wheel_loads(old, normal_loads);

    // 2.2 轮胎滑移率和侧偏角
    std::array<double, 4> slip_ratio, slip_angle;
    compute_tire_slips(old, slip_ratio, slip_angle);

    // 2.3 轮胎力（考虑载荷对侧偏刚度的影响）
    std::array<PacejkaTire::TireForces, 4> tire_forces;
    for (int i = 0; i < 4; ++i) {
      // 调整轮胎参数中的侧向刚度因子（随载荷变化）
      PacejkaParameters params = tires_[i].get_params(); // 需要添加getter
      double load_factor = normal_loads[i] / (mass_ * 9.81 / 4.0); // 名义载荷
      params.By *= (1.0 + tire_stiffness_load_factor_ * (load_factor - 1.0));
      params.Dy *= (1.0 + tire_stiffness_load_factor_ * (load_factor - 1.0));
      PacejkaTire temp_tire(params);
      temp_tire.set_normal_load(normal_loads[i]);
      temp_tire.set_road_mu(road_mu_);
      temp_tire.set_slip(slip_ratio[i], slip_angle[i]);
      tire_forces[i] = temp_tire.compute();
      // 存储到状态（供外部输出）
      state_.fx[i] = tire_forces[i].Fx;
      state_.fy[i] = tire_forces[i].Fy;
      state_.mz[i] = tire_forces[i].Mz;
    }

    // 2.4 动力传动系统扭矩
    Force_Unit engine_torque = compute_engine_torque(throttle_, old.engine_rpm);
    RSpeed_Unit trans_input_av = compute_transmission_input_av(old);
    Force_Unit clutch_torque =
        compute_clutch_torque(old.engine_rpm, trans_input_av, engine_torque,dt);
    Force_Unit trans_out_torque = clutch_torque *
                                  transmission_.get_current_ratio() *
                                  transmission_.final_drive;

    // 2.5 前后轴扭矩分配
    Force_Unit front_axle_torque = trans_out_torque * torque_split_front_;
    Force_Unit rear_axle_torque =
        trans_out_torque * (1.0 - torque_split_front_);

    // 2.6 差速器分配（左右轮）
    Force_Unit left_torque_front = 0.0, right_torque_front = 0.0;
    Force_Unit left_torque_rear = 0.0, right_torque_rear = 0.0;
    if (driven_axles_[0]) {
      differential_.distribute(front_axle_torque, old.wheel_fl_rpm,
                               old.wheel_fr_rpm, left_torque_front,
                               right_torque_front, dt);
    }
    if (driven_axles_[1]) {
      differential_.distribute(rear_axle_torque, old.wheel_rl_rpm,
                               old.wheel_rr_rpm, left_torque_rear,
                               right_torque_rear, dt);
    }

    // 2.7 半轴扭转弹性（可选）
    if (halfshaft_stiffness_ > 0.0) {
      apply_torsional_spring(old, left_torque_front, right_torque_front,
                             left_torque_rear, right_torque_rear, dt);
    }

    // 2.8 刹车扭矩
    std::array<Force_Unit, 4> brake_torques;
    compute_brake_torques(brake_, brake_torques);

    // 2.9 空气阻力
    Force_Unit air_drag = 0.5 * air_density_ * cda_ * old.vx * std::abs(old.vx);

    // 2.10 车辆合力和横摆力矩
    Force_Unit Fx_total = 0.0, Fy_total = 0.0;
    double Mz_total = 0.0;
    for (int i = 0; i < 4; ++i) {
      double steer = (i < 2) ? steering_angle_ : 0.0;
      double fx_veh = tire_forces[i].Fx * std::cos(steer) -
                      tire_forces[i].Fy * std::sin(steer);
      double fy_veh = tire_forces[i].Fx * std::sin(steer) +
                      tire_forces[i].Fy * std::cos(steer);
      Fx_total += fx_veh;
      Fy_total += fy_veh;
      double lx = (i < 2) ? wheelbase_front_ : -wheelbase_rear_;
      double ly = (i == 0 || i == 2) ? -track_ / 2.0 : track_ / 2.0;
      Mz_total += lx * fy_veh + ly * fx_veh + tire_forces[i].Mz;
    }
    Fx_total -= air_drag;

    // 3. 积分得到新状态（一次性计算所有）
    // ------------------------------------------------------------
    VehicleState new_state;

    // 平动加速度（保存到状态中供下次载荷转移使用）
    new_state.ax = Fx_total / mass_;
    new_state.ay = Fy_total / mass_;
    new_state.vx = old.vx + new_state.ax * dt;
    new_state.vy = old.vy + new_state.ay * dt;
    new_state.yaw_rate = old.yaw_rate + (Mz_total / yaw_inertia_) * dt;
    new_state.yaw_angle = old.yaw_angle + new_state.yaw_rate * dt;
    if (new_state.vx < -0.5)
      new_state.vx = -0.5;

    // 车轮角速度
    new_state.wheel_fl_rpm =
        integrate_wheel(old.wheel_fl_rpm, left_torque_front, brake_torques[0],
                        tire_forces[0].Fx, dt);
    new_state.wheel_fr_rpm =
        integrate_wheel(old.wheel_fr_rpm, right_torque_front, brake_torques[1],
                        tire_forces[1].Fx, dt);
    new_state.wheel_rl_rpm =
        integrate_wheel(old.wheel_rl_rpm, left_torque_rear, brake_torques[2],
                        tire_forces[2].Fx, dt);
    new_state.wheel_rr_rpm =
        integrate_wheel(old.wheel_rr_rpm, right_torque_rear, brake_torques[3],
                        tire_forces[3].Fx, dt);

    // 发动机转速
    Force_Unit net_engine_torque = engine_torque - clutch_torque;
    double engine_accel = net_engine_torque / engine_.crank_inertia;
    engine_accel =
        std::clamp(engine_accel, -max_engine_accel_, max_engine_accel_);
    new_state.engine_rpm = old.engine_rpm + engine_accel * dt;
    // 硬钳位——但关键是 clutch_torque 能把发动机拖下来！

    if (new_state.engine_rpm < 0.0)
      new_state.engine_rpm = 0.0;

    // 变速箱状态（需要根据新发动机转速和车速更新换档逻辑）
    double wheel_speed =
        (new_state.wheel_fl_rpm + new_state.wheel_fr_rpm) * 0.5 * wheel_radius_;
    transmission_.update(dt, clutch_, new_state.engine_rpm, wheel_speed);
    new_state.current_gear = transmission_.current_gear;
    new_state.is_shifting = transmission_.is_shifting;

    // 记录调试量
    new_state.net_drive_torque = left_torque_front + right_torque_front +
                                 left_torque_rear + right_torque_rear;
    for (int i = 0; i < 4; ++i)
      new_state.suspension_compression[i] =
          normal_loads[i] / suspensions_[i].stiffness;

    // 更新半轴扭转角（用于下一帧弹性计算）
    update_torsion_angles(old, new_state, left_torque_front, right_torque_front,
                          left_torque_rear, right_torque_rear, dt);

    // 4. 原子替换状态
    state_ = new_state;
  }

private:
  // ---------- 子组件 ----------
  EngineComponent engine_;
  ClutchComponent clutch_;
  TransmissionComponent transmission_;
  DifferentialComponent differential_;
  std::array<SuspensionComponent, 4> suspensions_;
  std::array<PacejkaTire, 4> tires_;

  // ---------- 车辆参数 ----------
  double mass_, yaw_inertia_, wheel_radius_, wheel_inertia_;
  double cda_, air_density_, brake_max_torque_, distribution_fr_;
  double wheelbase_front_, wheelbase_rear_, track_;
  double roll_center_height_, cg_height_;
  double torque_split_front_;        // 前轴扭矩分配比例
  std::array<bool, 2> driven_axles_; // 前轴、后轴是否驱动
  double halfshaft_stiffness_;       // 半轴扭转刚度 (Nm/rad)
  double road_mu_;
  double max_engine_accel_;
  double tire_stiffness_load_factor_; // 侧偏刚度随载荷变化系数

  // ---------- 驾驶员输入 ----------
  double throttle_ = 0.0;
  double brake_ = 0.0;
  double steering_angle_ = 0.0;

  // ---------- 当前状态 ----------
  VehicleState state_;

  // ========== 核心计算函数（只读旧状态）==========
  void compute_wheel_loads(const VehicleState &s,
                           std::array<Force_Unit, 4> &loads) const {
    // 静态轴荷分配 (前60%/后40%)
    double static_front = mass_ * 9.81 * 0.6;
    double static_rear = mass_ * 9.81 * 0.4;

    // 纵向载荷转移 (基于状态中的 ax)
    double transfer_long = (s.ax / 9.81) * mass_ * 9.81 *
                           (cg_height_ / (wheelbase_front_ + wheelbase_rear_));
    // 侧向载荷转移 (基于 ay)
    double transfer_lat_front =
        (s.ay / 9.81) * mass_ * 9.81 * (roll_center_height_ / track_) * 0.6;
    double transfer_lat_rear =
        (s.ay / 9.81) * mass_ * 9.81 * (roll_center_height_ / track_) * 0.4;

    loads[0] =
        (static_front / 2.0) - transfer_long / 2.0 - transfer_lat_front / 2.0;
    loads[1] =
        (static_front / 2.0) - transfer_long / 2.0 + transfer_lat_front / 2.0;
    loads[2] =
        (static_rear / 2.0) + transfer_long / 2.0 - transfer_lat_rear / 2.0;
    loads[3] =
        (static_rear / 2.0) + transfer_long / 2.0 + transfer_lat_rear / 2.0;

    for (auto &l : loads)
      l = std::max(100.0, l);
  }

  void compute_tire_slips(const VehicleState &s, std::array<double, 4> &sr,
                          std::array<double, 4> &sa) const {
    // 使用真实速度，但避免除零
    double vx = std::abs(s.vx);
    const double epsilon = 0.1; // 从 0.001 增大，减少静止时的假 slip

    for (int i = 0; i < 4; ++i) {
      double wheel_ang = 0.0;
      switch (i) {
      case 0:
        wheel_ang = s.wheel_fl_rpm;
        break;
      case 1:
        wheel_ang = s.wheel_fr_rpm;
        break;
      case 2:
        wheel_ang = s.wheel_rl_rpm;
        break;
      case 3:
        wheel_ang = s.wheel_rr_rpm;
        break;
      }
      double v_wheel = wheel_ang * wheel_radius_;

      // SAE 标准 slip 定义，避免静止时符号翻转
      if (vx < epsilon && std::abs(v_wheel) < epsilon) {
        sr[i] = 0.0; // 都静止，无 slip
      } else if (vx < epsilon) {
        // 车静止但轮在转：纯驱动 slip
        sr[i] = 1.0;
      } else {
        sr[i] = (v_wheel - vx) / std::max(vx, std::abs(v_wheel));
      }
      sr[i] = std::clamp(sr[i], -1.0, 1.0);

      double steer = (i < 2) ? steering_angle_ : 0.0;
      double alpha = steer - std::atan2(s.vy, std::max(vx, epsilon));
      sa[i] = std::clamp(alpha, -0.8, 0.8);
    }
  }

  // vehicle.hpp: compute_engine_torque
  // 删除整个过渡带，只保留硬钳位
  Force_Unit compute_engine_torque(double throttle, RSpeed_Unit rpm) const {
    // 怠速控制
    double base_torque = engine_.max_torque * throttle;
    if (throttle < 0.05 && rpm < engine_.idle_rpm) {
      double error = engine_.idle_rpm - rpm;
      base_torque += std::clamp(error * 3.0, 0.0, engine_.max_torque * 0.2);
    }
    // Limiter：超转时切断燃料，发动机扭矩 = 0
    if (rpm >= engine_.max_rpm) {
        base_torque = 0.0;  // 燃料切断
    }
    return base_torque; // 始终返回油门对应的扭矩
  }

  /*
  RSpeed_Unit compute_transmission_input_av(const VehicleState &s) const {
      double avg_wheel_rad = (s.wheel_fl_rpm + s.wheel_fr_rpm) * 0.5;
      double ratio = transmission_.get_current_ratio();
      if (std::abs(ratio) < 1e-6)
        return 0.0;
      // 车轮 → 差速器输入（× final_drive）→ 变速箱输出
      // 变速箱输出 × gear_ratio = 变速箱输入（即离合器从动盘转速）
      return avg_wheel_rad * ratio * transmission_.final_drive;
    }


  */
  RSpeed_Unit compute_transmission_input_av(const VehicleState &s) const {
    // 用驱动轮的平均转速，而不是固定前轮
    double avg_wheel = 0.0;
    int count = 0;
    if (driven_axles_[0]) {
      avg_wheel += (s.wheel_fl_rpm + s.wheel_fr_rpm) * 0.5;
      count++;
    }
    if (driven_axles_[1]) {
      avg_wheel += (s.wheel_rl_rpm + s.wheel_rr_rpm) * 0.5;
      count++;
    }
    if (count > 0)
      avg_wheel /= count;

    double ratio = transmission_.get_current_ratio();
    if (std::abs(ratio) < 1e-6)
      return 0.0;
    return avg_wheel * ratio * transmission_.final_drive;
  }

  Force_Unit compute_clutch_torque(RSpeed_Unit engine_av, RSpeed_Unit trans_av,
                                   Force_Unit engine_torque, Time_Unit dt) {
    double delta = engine_av - trans_av;

    // 摩擦容量
    double max_tq = clutch_.friction_capacity;
    double slip_factor = 1.0 - 0.15 * std::min(1.0, std::abs(delta) / 100.0);
    max_tq *= slip_factor;

    // === 刚度项 ===
    double k = 2.0; // 可以降到 1.0-1.5
    double stiffness_term = delta * k;

    // === 阻尼项 ===
    // delta_dot = (delta - prev_delta) / dt
    double delta_dot = (delta - last_clutch_delta_dont_use) / dt;
    double c = 0.3; // 阻尼系数，单位 Nm/(rad/s²)，需调参
    double damping_term = c * delta_dot;

    double desired = stiffness_term - damping_term;
    desired = std::clamp(desired, -max_tq, max_tq);

    last_clutch_delta_dont_use = delta; // 更新状态
    return desired;
  }
  void compute_brake_torques(double brake_pedal,
                             std::array<Force_Unit, 4> &bt) const {
    double total = brake_max_torque_ * brake_pedal;
    double front = total * distribution_fr_;
    double rear = total * (1.0 - distribution_fr_);
    bt[0] = -front / 2.0;
    bt[1] = -front / 2.0;
    bt[2] = -rear / 2.0;
    bt[3] = -rear / 2.0;
  }

  RSpeed_Unit integrate_wheel(RSpeed_Unit old_rpm, Force_Unit drive_tq,
                              Force_Unit brake_tq, Force_Unit Fx,
                              Time_Unit dt) const {
    Force_Unit tire_resist_tq = -Fx * wheel_radius_;
    Force_Unit net_tq = drive_tq + brake_tq + tire_resist_tq;
    double alpha = net_tq / wheel_inertia_;
    double new_rpm = old_rpm + alpha * dt;
    if (new_rpm < -5.0)
      new_rpm = -5.0;
    return new_rpm;
  }

  // 半轴扭转弹性模型：将扭矩存储在 torsion 状态中，并修正实际传递到车轮的扭矩
  void apply_torsional_spring(const VehicleState &old, Force_Unit &left_tq_f,
                              Force_Unit &right_tq_f, Force_Unit &left_tq_r,
                              Force_Unit &right_tq_r, Time_Unit dt) {

    // left_tq_f = halfshaft_stiffness_ * dt * engine_.current_rpm /60;
    //  简化的弹簧模型：实际传递扭矩 = 刚度 * 扭转角
    //  扭转角增量 = (输入扭矩 - 输出扭矩) / 刚度 / dt?
    //  更精确的做法需要半轴两端转速差积分 这里采用一阶滞后近似：实际扭矩 = (k *
    //  dt) / (1 + k * dt) * 输入扭矩
    //  但为了稳定性，直接使用当前扭矩乘以一个传递系数
    //  更简单：不做积分，直接使用弹簧力，但需要扭转角状态变量
    //  由于我们已经将扭转角存储在 state 中，这里可以使用下面的更新函数
    //  此函数只做占位，实际更新在 update_torsion_angles 中完成
    (void)old;
    (void)dt; // 避免未使用警告
              // 本例中我们不做额外修正，因为非必须
  }

  void update_torsion_angles(const VehicleState &old, VehicleState &new_state,
                             Force_Unit left_tq_f, Force_Unit right_tq_f,
                             Force_Unit left_tq_r, Force_Unit right_tq_r,
                             Time_Unit dt) {
    if (halfshaft_stiffness_ <= 0.0)
      return;
    // 半轴两侧转速差积分得到扭转角变化
    // 简化：假设半轴输入端转速等于差速器输出转速，输出端等于车轮转速
    // 实际需要更精确模型，这里仅做示例，保留原值
    new_state.torsion_fl = old.torsion_fl;
    new_state.torsion_fr = old.torsion_fr;
    new_state.torsion_rl = old.torsion_rl;
    new_state.torsion_rr = old.torsion_rr;
  }
};

// 注意：DifferentialComponent::distribute 需要在 engine.hpp
// 中实现，以下给出参考实现 请将以下代码添加到 engine.hpp 的
// DifferentialComponent 类中
/*
void distribute(Force_Unit input_torque, double left_speed, double right_speed,
                Force_Unit &left_torque, Force_Unit &right_torque, Time_Unit dt)
{ if (type == OPEN) { left_torque = input_torque * 0.5; right_torque =
input_torque * 0.5; } else if (type == LIMITED_SLIP) { double total =
input_torque; double diff = (left_speed - right_speed) * preload; left_torque =
(total * 0.5) - diff; right_torque = (total * 0.5) + diff; left_torque =
std::clamp(left_torque, -total, total); right_torque = total - left_torque; }
else if (type == LOCKED) {
        // 锁止时强制转速相等，简单平均分配（弹簧模型更复杂，此处简化）
        left_torque = input_torque * 0.5;
        right_torque = input_torque * 0.5;
    }
}
*/