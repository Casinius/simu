#pragma once
#include "unit.hpp"
#include <algorithm>
#include <cmath>

// -------------------------------------------------------------------
// Pacejka 89 / 2002 魔术公式核心函数
// 输入 slip (滑移率或滑移角, 无量纲), 以及 B,C,D,E 系数
// 输出 归一化力 (0~1)
// -------------------------------------------------------------------
inline double pacejka_f0(double slip, double B, double C, double D, double E) {
  // 避免 atan(巨大值) 溢出，限制 slip 范围
  const double max_slip = 50.0; // 50 弧度已经足够覆盖任何物理情况
  double x = std::clamp(slip, -max_slip, max_slip);
  double Bx = B * x;
  double arg = C * std::atan(Bx - E * (Bx - std::atan(Bx)));
  return D * std::sin(arg);
}

// -------------------------------------------------------------------
// 完整的轮胎参数
// -------------------------------------------------------------------
struct PacejkaParameters {
  // 纵向力系数 (Fx)
  double Bx = 10.0; // 刚度因子
  double Cx = 1.65; // 形状因子
  double Dx = 1.0;  // 峰值因子 (乘以垂直载荷后得最大力)
  double Ex = 0.0;  // 曲率因子
  double Sx = 0.0;  // 滑移率偏移 (不常用)

  // 侧向力系数 (Fy)
  double By = 10.0;
  double Cy = 1.35;
  double Dy = 1.0;
  double Ey = 0.0;
  double Sy = 0.0; // 滑移角偏移

  // 回正力矩系数 (Mz)
  double Bz = 10.0;
  double Cz = 2.0;
  double Dz = 0.1; // 通常较小
  double Ez = -20.0;
  double Sz = 0.0;

  // 允许的最大滑移率/滑移角 (用于平滑)
  double max_slip_ratio = 1.0;        // 纵向滑移率上限 (1 表示完全滑转)
  double max_slip_angle = M_PI / 3.0; // 侧偏角上限 60°
};

// -------------------------------------------------------------------
// 轮胎模型类
// -------------------------------------------------------------------
class PacejkaTire {
public:
  PacejkaTire() = default;
  explicit PacejkaTire(const PacejkaParameters &params) : m_params(params) {}

  // 设置路面附着系数 (一般 0.5~1.2)
  void set_road_mu(double mu) { m_road_mu = mu; }

  // 设置当前垂直载荷 (N)
  void set_normal_load(double Fz) { m_normal_load = std::max(0.0, Fz); }

  // 输入当前轮胎运动状态
  void set_slip(double slip_ratio, double slip_angle) {
    // 滑移率：正值表示驱动/滑转，负值表示制动/滑移
    m_raw_slip_ratio = slip_ratio;
    m_raw_slip_angle = slip_angle;
  }

  // 计算并获取三个力/力矩 (Fx, Fy, Mz)
  struct TireForces {
    Force_Unit Fx;          // 纵向力 (N)
    Force_Unit Fy;          // 侧向力 (N)
    Force_D_Length_Unit Mz; // 回正力矩 (Nm)
  };

  TireForces compute() {
    // 1. 钳位输入，保证 Magic Formula 输入稳定
    double sr = std::clamp(m_raw_slip_ratio, -m_params.max_slip_ratio,
                           m_params.max_slip_ratio);
    double sa = std::clamp(m_raw_slip_angle, -m_params.max_slip_angle,
                           m_params.max_slip_angle);

    // 2. 考虑路面附着系数对峰值力的缩放 (通常缩放 D 因子)
    double mu_factor = m_road_mu;
    double Dx_scaled = m_params.Dx * mu_factor * m_normal_load;
    double Dy_scaled = m_params.Dy * mu_factor * m_normal_load;

    // 3. 计算纯滑移/纯侧偏时的力 (无量纲部分)
    double fx0 = pacejka_f0(sr - m_params.Sx, m_params.Bx, m_params.Cx, 1.0,
                            m_params.Ex);
    double fy0 = pacejka_f0(sa - m_params.Sy, m_params.By, m_params.Cy, 1.0,
                            m_params.Ey);
    double mz0 = pacejka_f0(sa - m_params.Sz, m_params.Bz, m_params.Cz, 1.0,
                            m_params.Ez);

    // 4. 组合滑移 (Combined Slip) 修正 —— 使用相对简单的乘积模型，保证平滑过渡
    double total_slip = std::sqrt(sr * sr + std::tan(sa) * std::tan(sa));
    // 防止除零
    if (total_slip < 1e-6) {
      return {fx0 * Dx_scaled, fy0 * Dy_scaled,
              mz0 * m_params.Dz * m_normal_load};
    }

    // 纯工况下的力（带正确符号）
    double Fx_pure = fx0 * Dx_scaled;
    double Fy_pure = fy0 * Dy_scaled;

    // 不用指数衰减 太容易打滑
    // const double k = 3.0; // 衰减系数，可调
    // double G = std::exp(-k * total_slip * total_slip);

    double G = std::max(0.1, 1.0 - 0.5 * total_slip); // 或更标准的模型

    // 先衰减
    double Fx = Fx_pure * G;
    double Fy = Fy_pure * G;

    // 再摩擦椭圆限制（保持符号）
    double max_force = mu_factor * m_normal_load;
    double F_comb = std::sqrt(Fx * Fx + Fy * Fy);
    if (F_comb > max_force && F_comb > 0.0) {
      double scale = max_force / F_comb;
      Fx *= scale;
      Fy *= scale;
    }

    // 回正力矩: 受侧向力与侧偏角影响，也受组合滑移影响
    double Mz = mz0 * m_params.Dz * m_normal_load * G; // 简化模型

    return {Fx, Fy, Mz};
  }

  // 直接获取当前已计算的值（由外部更新状态后调用 compute 获取）
  // 也可将 compute 拆分为 update 和 get 函数
  auto get_params() const { return m_params; }
  void set_params(PacejkaParameters &&params) { m_params = params; }

private:
  PacejkaParameters m_params;
  double m_road_mu = 1.0;
  double m_normal_load = 0.0;
  double m_raw_slip_ratio = 0.0;
  double m_raw_slip_angle = 0.0;
};