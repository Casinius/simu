// VehicleDynamics.h
#ifndef VEHICLE_DYNAMICS_H
#define VEHICLE_DYNAMICS_H

#include "solve.h"
#include "solve_config.h"
#include <cmath>
#include <array>
#include <string>
#include <functional>

// ============================================================================
// 16-DOF 车辆动力学模型（增加发动机转速）
// 状态向量索引定义:
//   [0]  V_x   - 车辆坐标系纵向速度 (m/s)
//   [1]  V_y   - 车辆坐标系横向速度 (m/s)
//   [2]  V_z   - 车辆坐标系垂向速度 (m/s)
//   [3]  p     - 侧倾角速度 roll rate (rad/s)
//   [4]  q     - 俯仰角速度 pitch rate (rad/s)
//   [5]  r     - 横摆角速度 yaw rate (rad/s)
//   [6]  z_fR  - 右前非簧载质量垂向位移 (m)
//   [7]  z_fL  - 左前非簧载质量垂向位移 (m)
//   [8]  z_rR  - 右后非簧载质量垂向位移 (m)
//   [9]  z_rL  - 左后非簧载质量垂向位移 (m)
//   [10] w_fR  - 右前轮转速 (rad/s)
//   [11] w_fL  - 左前轮转速 (rad/s)
//   [12] w_rR  - 右后轮转速 (rad/s)
//   [13] w_rL  - 左后轮转速 (rad/s)
//   [14] delta - 前轮转向角 (rad)
//   [15] omega_e - 发动机转速 (rad/s)
// ============================================================================

namespace Vehicle {

// Brush 轮胎模型参数
template<class Scalar>
struct BrushTireParams {
    Scalar Cx;      // 纵向刚度 (N)
    Scalar Cy;      // 侧向刚度 (N/rad)
    Scalar mu0;     // 峰值摩擦系数
    Scalar epsilon; // 速度影响因子
};

// 发动机 MVM 模型参数
template<class Scalar>
struct EngineMVMParams {
    Scalar I_e;          // 发动机转动惯量 (kg*m^2)
    Scalar T_max;        // 最大扭矩 (N*m)
    Scalar omega_opt;    // 峰值扭矩转速 (rad/s)
    Scalar omega_max;    // 最大转速 (rad/s)
    Scalar tau_throttle; // 节气门响应时间常数 (s)
    Scalar f0, f1;       // 摩擦扭矩系数 (N*m, N*m/(rad/s))
    Scalar volumetric_efficiency; // 容积效率
};

// 变速箱参数
template<class Scalar>
struct TransmissionParams {
    int n_gears;                    // 档位数
    std::vector<Scalar> gear_ratios; // 各档传动比
    Scalar final_drive;             // 主减速比
    Scalar efficiency;              // 传动效率
    Scalar I_trans;                 // 变速箱转动惯量 (kg*m^2)
    int current_gear;               // 当前档位 (1-based)
};

// 差速锁参数（扭矩滑差模型）
template<class Scalar>
struct DifferentialParams {
    Scalar k_slip;      // 滑差系数 (N*m*s/rad)
    Scalar lock_limit;  // 锁止扭矩限制 (N*m)
    bool is_locked;     // 是否锁止
};

// 车辆参数结构体
template<class Scalar>
struct VehicleParams {
    // --- 质量与惯性 ---
    Scalar m_s;      // 簧载质量 (kg)
    Scalar m_usf;    // 前轴非簧载质量 (单侧, kg)
    Scalar m_usr;    // 后轴非簧载质量 (单侧, kg)
    Scalar I_xx;     // 簧载质量绕X轴转动惯量 (kg*m^2)
    Scalar I_yy;     // 簧载质量绕Y轴转动惯量 (kg*m^2)
    Scalar I_zz;     // 簧载质量绕Z轴转动惯量 (kg*m^2)
    Scalar I_wheel;  // 车轮绕自身旋转轴转动惯量 (kg*m^2)

    // --- 几何尺寸 ---
    Scalar l_f;      // 质心到前轴距离 (m)
    Scalar l_r;      // 质心到后轴距离 (m)
    Scalar t_f;      // 前轮距 (m)
    Scalar t_r;      // 后轮距 (m)
    Scalar h_s;      // 质心到侧倾轴高度 (m)
    Scalar h_CG;     // 质心到地面高度 (m)
    Scalar R_w;      // 车轮有效半径 (m)

    // --- 悬架参数 ---
    Scalar K_sf;     // 前悬架弹簧刚度 (N/m)
    Scalar K_sr;     // 后悬架弹簧刚度 (N/m)
    Scalar C_sf;     // 前悬架阻尼系数 (N*s/m)
    Scalar C_sr;     // 后悬架阻尼系数 (N*s/m)

    // --- 轮胎参数 (Brush 模型) ---
    BrushTireParams<Scalar> brush_front;  // 前轮轮胎参数
    BrushTireParams<Scalar> brush_rear;   // 后轮轮胎参数
    Scalar K_t;      // 轮胎垂向刚度 (N/m)
    Scalar C_t;      // 轮胎垂向阻尼 (N*s/m)

    // --- 空气动力学 ---
    Scalar rho;      // 空气密度 (kg/m^3)
    Scalar Cd;       // 风阻系数
    Scalar Af;       // 迎风面积 (m^2)

    // --- 转向系统 ---
    Scalar tau_delta; // 转向时间常数 (s)

    // --- 动力传动系统 ---
    EngineMVMParams<Scalar> engine;
    TransmissionParams<Scalar> transmission;
    DifferentialParams<Scalar> differential_front;  // 前桥差速锁
    DifferentialParams<Scalar> differential_rear;   // 后桥差速锁（如适用）
    bool is_fwd;  // 是否前驱（false则为后驱）

    // --- 制动参数 ---
    Scalar brake_bias;  // 制动力分配系数（前轴）
    
    // --- 重力 ---
    static constexpr Scalar g = 9.81;
};

// 路面输入
struct RoadInput {
    std::function<Scalar(Scalar x, Scalar y)> z_road;
    std::function<Scalar(Scalar x, Scalar y)> mu;
};

// 驾驶员输入
template<class Scalar>
struct DriverInput {
    Scalar throttle;       // 节气门开度 (0-1)
    Scalar brake_pedal;    // 制动踏板开度 (0-1)
    Scalar delta_cmd;      // 方向盘指令 (rad)
    int gear_cmd;          // 请求档位 (0表示空档, >0为档位)
};

// ============================================================================
// 车辆动力学核心类
// ============================================================================
template <typename Scalar>
class VehicleDynamics {
public:
    VehicleDynamics(const VehicleParams<Scalar>& params);
    
    void setDriverInput(const DriverInput<Scalar>& input) { driver_input_ = input; }
    void setRoadInput(const RoadInput& road) { road_ = road; }
    
    RHSFunc<Scalar> getRHS() const;
    void update(State<Scalar>& state, Scalar t, Scalar dt) const;
    
    // 状态提取辅助函数
    static Scalar Vx(const State<Scalar>& y)  { return y(0); }
    static Scalar Vy(const State<Scalar>& y)  { return y(1); }
    static Scalar Vz(const State<Scalar>& y)  { return y(2); }
    static Scalar rollRate(const State<Scalar>& y) { return y(3); }
    static Scalar pitchRate(const State<Scalar>& y){ return y(4); }
    static Scalar yawRate(const State<Scalar>& y)  { return y(5); }
    static Scalar engineSpeed(const State<Scalar>& y) { return y(15); }
    
    void computeDerived(const State<Scalar>& y, 
                        std::array<Scalar,4>& Fz,
                        std::array<Scalar,4>& Fx,
                        std::array<Scalar,4>& Fy,
                        std::array<Scalar,4>& slip_ratio,
                        std::array<Scalar,4>& slip_angle) const;

private:
    VehicleParams<Scalar> p_;
    DriverInput<Scalar> driver_input_;
    RoadInput road_;
    
    // Brush 轮胎模型
    void brushTireForces(Scalar kappa, Scalar alpha, Scalar Fz, Scalar mu,
                         Scalar Cx, Scalar Cy, Scalar& Fx, Scalar& Fy) const;
    
    // MVM 发动机模型
    Scalar mvmEngineTorque(Scalar throttle, Scalar omega_e) const;
    
    // 变速箱模型
    Scalar transmissionOutputTorque(Scalar T_engine, Scalar omega_e, Scalar omega_wheels_avg) const;
    
    // 差速锁扭矩分配
    void differentialTorqueSplit(Scalar T_in, Scalar omega_L, Scalar omega_R,
                                 const DifferentialParams<Scalar>& diff_params,
                                 Scalar& T_L, Scalar& T_R) const;
    
    // 传动系负载计算
    Scalar computeEngineLoadTorque(const std::array<Scalar,4>& T_drive,
                                   const std::array<Scalar,4>& omega_wheel) const;
    
    void computeTireForces(const State<Scalar>& y, int wheel_idx,
                           Scalar& Fx, Scalar& Fy, Scalar& Fz,
                           Scalar& slip_ratio, Scalar& slip_angle) const;
    
    State<Scalar> computeSprungForce(const State<Scalar>& y,
                                      const std::array<Scalar,4>& Fxt,
                                      const std::array<Scalar,4>& Fyt,
                                      const std::array<Scalar,4>& Fzt) const;
    
    static constexpr int FL = 0, FR = 1, RL = 2, RR = 3;
};

// ============================================================================
// 实现部分
// ============================================================================

template <typename Scalar>
VehicleDynamics<Scalar>::VehicleDynamics(const VehicleParams<Scalar>& params)
    : p_(params) {
    road_.z_road = [](Scalar, Scalar) { return Scalar(0.0); };
    road_.mu = [](Scalar, Scalar) { return Scalar(1.0); };
    driver_input_.throttle = 0;
    driver_input_.brake_pedal = 0;
    driver_input_.delta_cmd = 0;
    driver_input_.gear_cmd = 1;
}

template <typename Scalar>
Scalar VehicleDynamics<Scalar>::mvmEngineTorque(Scalar throttle, Scalar omega_e) const {
    const auto& e = p_.engine;
    
    // 限制发动机转速范围
    Scalar omega = std::max(Scalar(100), std::min(omega_e, e.omega_max));
    
    // 指示扭矩（基于节气门和转速的简化模型）
    Scalar throttle_eff = std::max(Scalar(0), std::min(Scalar(1), throttle));
    Scalar omega_norm = (omega - e.omega_opt) / (e.omega_max - e.omega_opt);
    Scalar torque_factor = throttle_eff * (1 - omega_norm * omega_norm);
    Scalar T_ind = e.T_max * torque_factor * e.volumetric_efficiency;
    
    // 摩擦扭矩
    Scalar T_fric = e.f0 + e.f1 * omega;
    
    // 净扭矩
    return std::max(Scalar(0), T_ind - T_fric);
}

template <typename Scalar>
Scalar VehicleDynamics<Scalar>::transmissionOutputTorque(Scalar T_engine, Scalar omega_e, 
                                                          Scalar omega_wheels_avg) const {
    const auto& t = p_.transmission;
    
    if (t.current_gear <= 0 || t.current_gear > t.n_gears) {
        return 0;  // 空档
    }
    
    Scalar gear_ratio = t.gear_ratios[t.current_gear - 1];
    Scalar total_ratio = gear_ratio * t.final_drive;
    
    // 输出扭矩 = 发动机扭矩 * 总传动比 * 效率
    Scalar T_out = T_engine * total_ratio * t.efficiency;
    
    return T_out;
}

template <typename Scalar>
void VehicleDynamics<Scalar>::differentialTorqueSplit(Scalar T_in, Scalar omega_L, Scalar omega_R,
                                                      const DifferentialParams<Scalar>& diff_params,
                                                      Scalar& T_L, Scalar& T_R) const {
    if (diff_params.is_locked) {
        // 完全锁止：平均分配，允许转速差
        T_L = T_in / 2;
        T_R = T_in / 2;
    } else {
        // 扭矩滑差模型
        Scalar delta_omega = omega_R - omega_L;
        Scalar slip_torque = diff_params.k_slip * delta_omega;
        
        // 限制滑差扭矩
        slip_torque = std::max(-diff_params.lock_limit, 
                              std::min(diff_params.lock_limit, slip_torque));
        
        T_L = std::max(Scalar(0), T_in / 2 - slip_torque);
        T_R = std::max(Scalar(0), T_in / 2 + slip_torque);
        
        // 确保总和等于输入扭矩
        Scalar total = T_L + T_R;
        if (total > 0) {
            T_L = T_L / total * T_in;
            T_R = T_R / total * T_in;
        }
    }
}

template <typename Scalar>
Scalar VehicleDynamics<Scalar>::computeEngineLoadTorque(const std::array<Scalar,4>& T_drive,
                                                        const std::array<Scalar,4>& omega_wheel) const {
    const auto& t = p_.transmission;
    
    if (t.current_gear <= 0) return 0;
    
    Scalar gear_ratio = t.gear_ratios[t.current_gear - 1];
    Scalar total_ratio = gear_ratio * t.final_drive;
    
    // 计算驱动轮总驱动扭矩（考虑驱动形式）
    Scalar total_drive_torque = 0;
    if (p_.is_fwd) {
        total_drive_torque = T_drive[FL] + T_drive[FR];
    } else {
        total_drive_torque = T_drive[RL] + T_drive[RR];
    }
    
    // 折算到发动机的负载扭矩
    Scalar T_load = total_drive_torque / (total_ratio * t.efficiency);
    
    return T_load;
}

template <typename Scalar>
void VehicleDynamics<Scalar>::brushTireForces(Scalar kappa, Scalar alpha, Scalar Fz, Scalar mu,
                                              Scalar Cx, Scalar Cy, Scalar& Fx, Scalar& Fy) const {
    const Scalar epsilon = 1e-6;
    
    // 最大摩擦力
    Scalar F_max = mu * Fz;
    
    // 纯纵向和纯侧向的理论最大滑移/侧偏
    Scalar kappa_max = F_max / (Cx + epsilon);
    Scalar tan_alpha_max = F_max / (Cy + epsilon);
    
    // 限制滑移率和侧偏角
    Scalar kappa_eff = std::max(-kappa_max, std::min(kappa_max, kappa));
    Scalar tan_alpha = std::tan(alpha);
    Scalar tan_alpha_eff = std::max(-tan_alpha_max, std::min(tan_alpha_max, tan_alpha));
    
    // 线性区域的力
    Scalar Fx_lin = Cx * kappa_eff;
    Scalar Fy_lin = Cy * tan_alpha_eff;
    
    // 总滑移
    Scalar s_total = std::sqrt(kappa_eff * kappa_eff + tan_alpha_eff * tan_alpha_eff);
    
    if (s_total < epsilon) {
        Fx = 0;
        Fy = 0;
        return;
    }
    
    // 摩擦椭圆限制
    Scalar F_combined_lin = std::sqrt(Fx_lin * Fx_lin + Fy_lin * Fy_lin);
    
    if (F_combined_lin <= F_max) {
        // 线性区域
        Fx = Fx_lin;
        Fy = Fy_lin;
    } else {
        // 饱和区域
        Scalar ratio = F_max / (F_combined_lin + epsilon);
        Fx = Fx_lin * ratio;
        Fy = Fy_lin * ratio;
    }
    
    // 考虑速度对摩擦系数的影响
    // (这里简化处理)
}

template <typename Scalar>
RHSFunc<Scalar> VehicleDynamics<Scalar>::getRHS() const {
    const VehicleDynamics* self = this;
    
    return [self](Scalar t, const State<Scalar>& y) -> State<Scalar> {
        const auto& p = self->p_;
        const auto& d = self->driver_input_;
        
        State<Scalar> dy = State<Scalar>::Zero(16);
        
        // 提取状态
        Scalar Vx = y(0), Vy = y(1), Vz = y(2);
        Scalar p_rate = y(3), q_rate = y(4), r_rate = y(5);
        Scalar omega_e = y(15);
        
        // 车辆速度大小
        Scalar V_horiz = std::sqrt(Vx*Vx + Vy*Vy);
        if (V_horiz < 0.1) V_horiz = 0.1;
        
        // 空气阻力
        Scalar F_aero_x = -0.5 * p.rho * p.Cd * p.Af * Vx * std::abs(Vx);
        Scalar F_aero_y = -0.5 * p.rho * p.Cd * p.Af * Vy * std::abs(Vy);
        
        // 计算四轮轮胎力
        std::array<Scalar,4> Fxt, Fyt, Fzt;
        std::array<Scalar,4> kappa, alpha;
        for (int i = 0; i < 4; ++i) {
            self->computeTireForces(y, i, Fxt[i], Fyt[i], Fzt[i], kappa[i], alpha[i]);
        }
        
        // 计算驱动扭矩（通过传动系模型）
        std::array<Scalar,4> T_drive = {0, 0, 0, 0};
        
        // 发动机净扭矩
        Scalar T_engine_net = self->mvmEngineTorque(d.throttle, omega_e);
        
        // 制动力矩（基于制动踏板和制动力分配）
        std::array<Scalar,4> T_brake = {0, 0, 0, 0};
        Scalar brake_torque_total = d.brake_pedal * 3000;  // 最大制动力矩3000 Nm
        if (p.is_fwd) {
            T_brake[FL] = T_brake[FR] = brake_torque_total * p.brake_bias / 2;
            T_brake[RL] = T_brake[RR] = brake_torque_total * (1 - p.brake_bias) / 2;
        } else {
            T_brake[FL] = T_brake[FR] = brake_torque_total * (1 - p.brake_bias) / 2;
            T_brake[RL] = T_brake[RR] = brake_torque_total * p.brake_bias / 2;
        }
        
        // 变速箱输出扭矩
        Scalar avg_wheel_speed = 0;
        if (p.is_fwd) {
            avg_wheel_speed = (y(10) + y(11)) / 2;
        } else {
            avg_wheel_speed = (y(12) + y(13)) / 2;
        }
        Scalar T_trans_out = self->transmissionOutputTorque(T_engine_net, omega_e, avg_wheel_speed);
        
        // 差速锁扭矩分配
        if (p.is_fwd) {
            self->differentialTorqueSplit(T_trans_out, y(11), y(10), p.differential_front,
                                          T_drive[FL], T_drive[FR]);
            // 后轮无驱动扭矩
            T_drive[RL] = T_drive[RR] = 0;
        } else {
            self->differentialTorqueSplit(T_trans_out, y(13), y(12), p.differential_rear,
                                          T_drive[RL], T_drive[RR]);
            T_drive[FL] = T_drive[FR] = 0;
        }
        
        // 坐标转换：轮胎力从轮胎坐标系到车辆坐标系
        Scalar delta = y(14);
        std::array<Scalar,4> Fx_v, Fy_v;
        
        for (int i = 0; i < 2; ++i) {
            Fx_v[i] = Fxt[i] * std::cos(delta) - Fyt[i] * std::sin(delta);
            Fy_v[i] = Fxt[i] * std::sin(delta) + Fyt[i] * std::cos(delta);
        }
        for (int i = 2; i < 4; ++i) {
            Fx_v[i] = Fxt[i];
            Fy_v[i] = Fyt[i];
        }
        
        // 总外力
        Scalar Fx_total = F_aero_x;
        Scalar Fy_total = F_aero_y;
        Scalar Fz_total = -p.m_s * p.g;
        
        for (int i = 0; i < 4; ++i) {
            Fx_total += Fx_v[i];
            Fy_total += Fy_v[i];
            Fz_total += Fzt[i];
        }
        
        // 簧载质量平移方程
        Scalar m_total = p.m_s + 2*p.m_usf + 2*p.m_usr;
        dy(0) = Fx_total / m_total + Vy * r_rate - Vz * q_rate;
        dy(1) = Fy_total / m_total + Vz * p_rate - Vx * r_rate;
        dy(2) = Fz_total / m_total + Vx * q_rate - Vy * p_rate;
        
        // 簧载质量旋转方程
        Scalar Mx = 0, My = 0, Mz = 0;
        
        // 轮胎力产生的力矩
        // FL
        Mx += Fzt[0] * (p.t_f/2) - Fy_v[0] * p.h_CG;
        My += -Fzt[0] * p.l_f + Fx_v[0] * p.h_CG;
        Mz += Fx_v[0] * (p.t_f/2) - Fy_v[0] * p.l_f;
        
        // FR
        Mx += -Fzt[1] * (p.t_f/2) - Fy_v[1] * p.h_CG;
        My += -Fzt[1] * p.l_f + Fx_v[1] * p.h_CG;
        Mz += -Fx_v[1] * (p.t_f/2) - Fy_v[1] * p.l_f;
        
        // RL
        Mx += Fzt[2] * (p.t_r/2) - Fy_v[2] * p.h_CG;
        My += Fzt[2] * p.l_r + Fx_v[2] * p.h_CG;
        Mz += Fx_v[2] * (p.t_r/2) - Fy_v[2] * (-p.l_r);
        
        // RR
        Mx += -Fzt[3] * (p.t_r/2) - Fy_v[3] * p.h_CG;
        My += Fzt[3] * p.l_r + Fx_v[3] * p.h_CG;
        Mz += -Fx_v[3] * (p.t_r/2) - Fy_v[3] * (-p.l_r);
        
        dy(3) = (Mx + (p.I_yy - p.I_zz) * q_rate * r_rate) / p.I_xx;
        dy(4) = (My + (p.I_zz - p.I_xx) * r_rate * p_rate) / p.I_yy;
        dy(5) = (Mz + (p.I_xx - p.I_yy) * p_rate * q_rate) / p.I_zz;
        
        // 非簧载质量垂向运动
        Scalar z_s = 0, phi = 0, theta = 0;
        Scalar z_s_static = -(p.m_s * p.g) / (4 * (p.K_sf + p.K_sr));
        
        for (int i = 0; i < 4; ++i) {
            Scalar z_us = y(6 + i);
            Scalar K_susp = (i < 2) ? p.K_sf : p.K_sr;
            Scalar C_susp = (i < 2) ? p.C_sf : p.C_sr;
            
            Scalar x_wheel = (i < 2) ? p.l_f : -p.l_r;
            Scalar y_wheel = (i % 2 == 0) ? 
                           ((i < 2) ? p.t_f/2 : p.t_r/2) :
                           ((i < 2) ? -p.t_f/2 : -p.t_r/2);
            Scalar z_r = self->road_.z_road(x_wheel, y_wheel);
            
            Scalar F_tire_z = p.K_t * (z_r - z_us);
            
            Scalar z_s_wheel = z_s + ((i < 2) ? p.l_f : -p.l_r) * theta 
                             + ((i % 2 == 0) ? 1 : -1) * ((i < 2) ? p.t_f/2 : p.t_r/2) * phi;
            Scalar F_susp_z = K_susp * (z_us - z_s_wheel) + C_susp * (0 - Vz);
            
            Scalar m_us = (i < 2) ? p.m_usf : p.m_usr;
            dy(6 + i) = (F_tire_z - F_susp_z) / m_us - p.g;
        }
        
        // 车轮旋转动力学
        for (int i = 0; i < 4; ++i) {
            Scalar T_net = T_drive[i] - T_brake[i] - Fxt[i] * p.R_w;
            dy(10 + i) = T_net / p.I_wheel;
        }
        
        // 转向动力学
        dy(14) = (d.delta_cmd - y(14)) / p.tau_delta;
        
        // 发动机动力学
        Scalar T_load = self->computeEngineLoadTorque(T_drive, {y(10), y(11), y(12), y(13)});
        Scalar I_total = p.engine.I_e + p.transmission.I_trans;
        dy(15) = (T_engine_net - T_load) / I_total;
        
        return dy;
    };
}

template <typename Scalar>
void VehicleDynamics<Scalar>::computeTireForces(
    const State<Scalar>& y, int wheel_idx,
    Scalar& Fx, Scalar& Fy, Scalar& Fz,
    Scalar& slip_ratio, Scalar& slip_angle) const {
    
    const auto& p = p_;
    Scalar Vx = y(0), Vy = y(1);
    Scalar r_rate = y(5);
    Scalar delta = y(14);
    Scalar omega = y(10 + wheel_idx);
    Scalar z_us = y(6 + wheel_idx);
    
    // 车轮中心位置
    Scalar x_w = (wheel_idx < 2) ? p.l_f : -p.l_r;
    Scalar y_w = (wheel_idx % 2 == 0) ? 
                   ((wheel_idx < 2) ? p.t_f/2 : p.t_r/2) :
                   ((wheel_idx < 2) ? -p.t_f/2 : -p.t_r/2);
    
    // 车轮中心速度
    Scalar Vx_w = Vx - r_rate * y_w;
    Scalar Vy_w = Vy + r_rate * x_w;
    
    // 轮胎坐标系速度
    Scalar Vx_t, Vy_t;
    if (wheel_idx < 2) {
        Vx_t = Vx_w * std::cos(delta) + Vy_w * std::sin(delta);
        Vy_t = -Vx_w * std::sin(delta) + Vy_w * std::cos(delta);
    } else {
        Vx_t = Vx_w;
        Vy_t = Vy_w;
    }
    
    // 滑移率计算
    Scalar V_c = omega * p.R_w;
    if (std::abs(Vx_t) < 0.1) {
        slip_ratio = 0;
    } else if (Vx_t > V_c) {
        slip_ratio = (V_c - Vx_t) / Vx_t;  // 制动
    } else {
        slip_ratio = (V_c - Vx_t) / V_c;   // 驱动
    }
    slip_ratio = std::max(Scalar(-1.0), std::min(Scalar(1.0), slip_ratio));
    
    // 滑移角
    if (std::abs(Vx_t) < 0.1) {
        slip_angle = 0;
    } else {
        slip_angle = std::atan2(Vy_t, std::abs(Vx_t));
    }
    
    // 垂向载荷（简化计算）
    Scalar Fz_static = (p.m_s * p.g / 4) + 
                       ((wheel_idx < 2) ? p.m_usf : p.m_usr) * p.g;
    
    // 动态载荷转移（简化）
    Scalar ax = 0;  // 需要从状态估计
    Scalar ay = 0;
    Scalar Fz_xfer_long = (wheel_idx < 2) ? -p.m_s * ax * p.h_CG / (2 * (p.l_f + p.l_r)) :
                                             p.m_s * ax * p.h_CG / (2 * (p.l_f + p.l_r));
    Scalar Fz_xfer_lat = ((wheel_idx % 2 == 0) ? 1 : -1) * 
                         p.m_s * ay * p.h_CG / (2 * ((wheel_idx < 2) ? p.t_f : p.t_r));
    
    Fz = Fz_static + Fz_xfer_long + Fz_xfer_lat;
    Fz = std::max(Scalar(100.0), Fz);
    
    // 路面附着系数
    Scalar mu = this->road_.mu(x_w, y_w);
    
    // 选择轮胎参数
    const auto& brush_params = (wheel_idx < 2) ? p.brush_front : p.brush_rear;
    
    // Brush 模型计算
    brushTireForces(slip_ratio, slip_angle, Fz, mu,
                    brush_params.Cx, brush_params.Cy, Fx, Fy);
}

template <typename Scalar>
void VehicleDynamics<Scalar>::update(State<Scalar>& state, Scalar t, Scalar dt) const {
    auto rhs = getRHS();
    State<Scalar> k1 = rhs(t, state);
    State<Scalar> k2 = rhs(t + dt/2, state + dt/2 * k1);
    State<Scalar> k3 = rhs(t + dt/2, state + dt/2 * k2);
    State<Scalar> k4 = rhs(t + dt, state + dt * k3);
    state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4);
}

template <typename Scalar>
void VehicleDynamics<Scalar>::computeDerived(
    const State<Scalar>& y,
    std::array<Scalar,4>& Fz,
    std::array<Scalar,4>& Fx,
    std::array<Scalar,4>& Fy,
    std::array<Scalar,4>& slip_ratio,
    std::array<Scalar,4>& slip_angle) const {
    for (int i = 0; i < 4; ++i) {
        computeTireForces(y, i, Fx[i], Fy[i], Fz[i], slip_ratio[i], slip_angle[i]);
    }
}

// ============================================================================
// 预定义参数集（示例：中型轿车）
// ============================================================================
template<class Scalar>
inline VehicleParams<Scalar> makeDefaultVehicleParams() {
    VehicleParams<Scalar> p;
    
    // 质量与惯性
    p.m_s = 1200.0;
    p.m_usf = 45.0;
    p.m_usr = 40.0;
    p.I_xx = 400.0;
    p.I_yy = 1200.0;
    p.I_zz = 1400.0;
    p.I_wheel = 1.2;
    
    // 几何
    p.l_f = 1.25;
    p.l_r = 1.35;
    p.t_f = 1.55;
    p.t_r = 1.55;
    p.h_s = 0.35;
    p.h_CG = 0.55;
    p.R_w = 0.32;
    
    // 悬架
    p.K_sf = 28000.0;
    p.K_sr = 24000.0;
    p.C_sf = 2500.0;
    p.C_sr = 2200.0;
    
    // Brush 轮胎参数
    p.brush_front.Cx = 80000.0;   // 纵向刚度
    p.brush_front.Cy = 60000.0;   // 侧向刚度 (N/rad)
    p.brush_front.mu0 = 1.0;      // 峰值摩擦系数
    p.brush_front.epsilon = 0.01;
    
    p.brush_rear.Cx = 75000.0;
    p.brush_rear.Cy = 58000.0;
    p.brush_rear.mu0 = 1.0;
    p.brush_rear.epsilon = 0.01;
    
    p.K_t = 200000.0;
    p.C_t = 200.0;
    
    // 空气动力学
    p.rho = 1.225;
    p.Cd = 0.30;
    p.Af = 2.2;
    
    // 转向
    p.tau_delta = 0.1;
    
    // 发动机 MVM 参数
    p.engine.I_e = 0.2;
    p.engine.T_max = 200.0;
    p.engine.omega_opt = 4000 * M_PI / 30;  // 4000 rpm
    p.engine.omega_max = 6000 * M_PI / 30;   // 6000 rpm
    p.engine.tau_throttle = 0.05;
    p.engine.f0 = 20.0;
    p.engine.f1 = 0.01;
    p.engine.volumetric_efficiency = 0.85;
    
    // 变速箱参数
    p.transmission.n_gears = 5;
    p.transmission.gear_ratios = {3.5, 2.0, 1.3, 1.0, 0.8};
    p.transmission.final_drive = 3.7;
    p.transmission.efficiency = 0.92;
    p.transmission.I_trans = 0.15;
    p.transmission.current_gear = 1;
    
    // 差速锁参数
    p.differential_front.k_slip = 50.0;
    p.differential_front.lock_limit = 200.0;
    p.differential_front.is_locked = false;
    
    p.differential_rear.k_slip = 50.0;
    p.differential_rear.lock_limit = 200.0;
    p.differential_rear.is_locked = false;
    
    // 驱动形式
    p.is_fwd = true;
    
    // 制动
    p.brake_bias = 0.6;
    
    return p;
}

} // namespace Vehicle

#endif // VEHICLE_DYNAMICS_H