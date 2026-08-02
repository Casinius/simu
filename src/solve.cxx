// ODESolver.cpp
#include "solve.h"
// #include "solve_config.h"
#include <algorithm>
#include <cmath>
#include <iostream>

#include <ratio>
#pragma once
// ======================================================================
// RK45 实现
// ======================================================================
// Butcher 表系数 (Fehlberg 4(5))

namespace rk45_const {
const auto a21 = 1.0 / 4.0;
const auto a31 = 3.0 / 32.0, a32 = 9.0 / 32.0;
const auto a41 = 1932.0 / 2197.0, a42 = -7200.0 / 2197.0, a43 = 7296.0 / 2197.0;
const auto a51 = 439.0 / 216.0, a52 = -8.0, a53 = 3680.0 / 513.0,
           a54 = -845.0 / 4104.0;
const auto a61 = -8.0 / 27.0, a62 = 2.0, a63 = -3544.0 / 2565.0,
           a64 = 1859.0 / 4104.0, a65 = -11.0 / 40.0;

const auto c2 = 1.0 / 4.0, c3 = 3.0 / 8.0, c4 = 12.0 / 13.0, c5 = 1.0,
           c6 = 1.0 / 2.0;

const auto b4_1 = 25.0 / 216.0, b4_3 = 1408.0 / 2565.0, b4_4 = 2197.0 / 4104.0,
           b4_5 = -1.0 / 5.0;
const auto b5_1 = 16.0 / 135.0, b5_3 = 6656.0 / 12825.0,
           b5_4 = 28561.0 / 56430.0, b5_5 = -9.0 / 50.0, b5_6 = 2.0 / 55.0;
} // namespace rk45_const
template <class Scalar>
RK45Solver<Scalar>::RK45Solver(Scalar atol, Scalar rtol, Scalar h_min,
                               Scalar h_max, Scalar safety, Scalar fac_min,
                               Scalar fac_max)
    : atol_(atol), rtol_(rtol), h_min_(h_min), h_max_(h_max), safety_(safety),
      fac_min_(fac_min), fac_max_(fac_max) {}

template <class Scalar>
void RK45Solver<Scalar>::solve(const RHSFunc<Scalar> &f, Scalar t0, Scalar t1,
                               const State<Scalar> &y0, Scalar h0,
                               std::vector<Scalar> &times,
                               std::vector<State<Scalar>> &states) {

  times.clear();
  states.clear();

  // std::cout << "RK45: h_min_ = " << h_min_ << ", h_max_ = " << h_max_ <<
  // std::endl;
  Scalar t = t0;
  State<Scalar> y = y0;
  Scalar h = std::min(h0, h_max_);
  times.push_back(t);
  states.push_back(y);

  while (t < t1) {
    // 保证最后一步不越过终点
    if (t + h > t1)
      h = t1 - t;
    if (h < h_min_) {
      std::cerr << "RK45: step size below minimum! t = " << t << ", h = " << h
                << std::endl;
      break;
    }

    auto [y_next, error, h_next] = step(f, t, y, h);

    // 误差评估（混合相对/绝对误差）
    Scalar scale = atol_ + rtol_ * y.cwiseAbs().maxCoeff();
    Scalar err_norm = error / scale;

    if (err_norm <= 1.0) { // 接受该步
      t += h;
      y = std::move(y_next);
      times.push_back(t);
      states.push_back(y);
      // 调整下一步长
      if (t >= t1)
        break;
      Scalar fac = safety_ * std::pow(err_norm, -0.2);
      fac = std::clamp(fac, fac_min_, fac_max_);
      h = std::min(h * fac, h_max_);
    } else { // 拒绝，缩小步长重试
      Scalar fac = safety_ * std::pow(err_norm, -0.25);
      fac = std::clamp(fac, fac_min_, 0.9);
      h = std::max(h * fac, h_min_);
    }
  }
}

// 在 solve.cxx 中替换 RK45Solver::step 中的 error 计算部分
template <class Scalar>
typename RK45Solver<Scalar>::StepResult
RK45Solver<Scalar>::step(const RHSFunc<Scalar> &f, Scalar t,
                         const State<Scalar> &y, Scalar h) {
  using namespace rk45_const;
  // ... 计算 k1..k6, y4, y5 相同 ...
  State<Scalar> k1 = f(t, y);
  State<Scalar> k2 = f(t + c2 * h, y + h * (a21 * k1));
  State<Scalar> k3 = f(t + c3 * h, y + h * (a31 * k1 + a32 * k2));
  State<Scalar> k4 = f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3));
  State<Scalar> k5 =
      f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4));
  State<Scalar> k6 = f(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 +
                                            a64 * k4 + a65 * k5));

  State<Scalar> y4 =
      y + h * (b4_1 * k1 + b4_3 * k3 + b4_4 * k4 + b4_5 * k5); // 4阶近似
  State<Scalar> y5 = y + h * (b5_1 * k1 + b5_3 * k3 + b5_4 * k4 + b5_5 * k5 +
                              b5_6 * k6); // 5阶近似
  State<Scalar> error_vec = y5 - y4;      // 分量误差
  // 计算混合误差范数（分量相对/绝对）
  Scalar err_norm = 0.0;
  for (int i = 0; i < y.size(); ++i) {
    Scalar scale = atol_ + rtol_ * std::abs(y(i));
    Scalar e = std::abs(error_vec(i)) / scale;
    err_norm = std::max(err_norm, e); // 无穷范数，也可使用均方根
  }
  return {std::move(y5), err_norm, h};
}

// ======================================================================
// 标准固定步长BDF2 实现
// ======================================================================
template <class Scalar>
BDF2Solver<Scalar>::BDF2Solver(Scalar newton_tol, int max_iter)
    : newton_tol_(newton_tol), max_iter_(max_iter) {}

// 在 solve.cxx 中替换 BDF2Solver::solve 的第一步启动部分
template <class Scalar>
void BDF2Solver<Scalar>::solve(const RHSFunc<Scalar> &f, Scalar t0, Scalar t1,
                               const State<Scalar> &y0, Scalar h0,
                               std::vector<Scalar> &times,
                               std::vector<State<Scalar>> &states) {
  times.clear();
  states.clear();

  // 第一步：使用隐式欧拉生成 y1（固定步长 h）
  State<Scalar> y1 = y0;
  // 步长限制
  Scalar h = std::min(h0, t1 - t0);
  if (h <= 0)
    return;

  Scalar t = t0;
  State<Scalar> y = y0;
  times.push_back(t);
  states.push_back(y);

  const int n = y0.size();
  const Scalar newton_tol = newton_tol_;
  const int max_iter = max_iter_;
  for (int iter = 0; iter < max_iter; ++iter) {
    State<Scalar> res = y1 - (y0 + h * f(t + h, y1));
    if (res.norm() < newton_tol)
      break;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> J(n, n);
    Scalar eps = 1e-8;
    for (int j = 0; j < n; ++j) {
      State<Scalar> yp = y1, ym = y1;
      yp(j) += eps;
      ym(j) -= eps;
      State<Scalar> rp = yp - (y0 + h * f(t + h, yp));
      State<Scalar> rm = ym - (y0 + h * f(t + h, ym));
      J.col(j) = (rp - rm) / (2 * eps);
    }
    Eigen::PartialPivLU<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        lu(J);
    y1 += lu.solve(-res);
  }
  Scalar t1_bdf = t + h;
  times.push_back(t1_bdf);
  states.push_back(y1);

  // 后续 BDF2 步进
  State<Scalar> y_n = y, y_n1 = y1;
  Scalar t_n = t, t_n1 = t1_bdf;

  while (t_n1 < t1 - 1e-12) {
    Scalar t_n2 = t_n1 + h;
    if (t_n2 > t1) {
      // 最后一段，改用隐式欧拉完成（或减小步长）
      h = t1 - t_n1;
      t_n2 = t1;
      // 隐式欧拉单步
      State<Scalar> y_n2 = y_n1;
      for (int iter = 0; iter < max_iter; ++iter) {
        State<Scalar> res = y_n2 - (y_n1 + h * f(t_n2, y_n2));
        if (res.norm() < newton_tol)
          break;
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> J(n, n);
        Scalar eps = 1e-8;
        for (int j = 0; j < n; ++j) {
          State<Scalar> yp = y_n2, ym = y_n2;
          yp(j) += eps;
          ym(j) -= eps;
          State<Scalar> rp = yp - (y_n1 + h * f(t_n2, yp));
          State<Scalar> rm = ym - (y_n1 + h * f(t_n2, ym));
          J.col(j) = (rp - rm) / (2 * eps);
        }
        Eigen::PartialPivLU<
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
            lu(J);
        y_n2 += lu.solve(-res);
      }
      times.push_back(t_n2);
      states.push_back(y_n2);
      break;
    } else {
      // 标准 BDF2 步
      State<Scalar> y_n2 = dyn_eps_bdf2_step(f, t_n, y_n, t_n1, y_n1, h);
      if (y_n2.size() == 0)
        break;
      times.push_back(t_n2);
      states.push_back(y_n2);
      y_n = y_n1;
      y_n1 = y_n2;
      t_n = t_n1;
      t_n1 = t_n2;
    }
  }
}
// not reuse , correct impl

template <class Scalar>
State<Scalar>
BDF2Solver<Scalar>::bdf2_step(const RHSFunc<Scalar> &f, Scalar t_n,
                              const State<Scalar> &y_n, Scalar t_n1,
                              const State<Scalar> &y_n1, Scalar h) {
  Scalar t_n2 = t_n1 + h;
  int n = y_n.size();
  State<Scalar> y = y_n1; // 初始猜测

  // 牛顿迭代求解隐式方程:
  // F(y) = y - [ 4/3 y_{n+1} - 1/3 y_n + 2/3 h f(t_{n+2}, y) ] = 0
  for (int iter = 0; iter < max_iter_; ++iter) {
    State<Scalar> F =
        y - (4.0 / 3.0 * y_n1 - 1.0 / 3.0 * y_n + (2.0 / 3.0) * h * f(t_n2, y));

    if (F.norm() < newton_tol_) {
      return y; // 收敛
    }

    // 数值 Jacobian: J(i,j) = dF_i / dy_j
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n);
    Scalar eps = 1e-8;
    for (int j = 0; j < n; ++j) {
      State<Scalar> y_plus = y;
      y_plus(j) += eps;
      State<Scalar> y_minus = y;
      y_minus(j) -= eps;

      State<Scalar> F_plus = y_plus - (4.0 / 3.0 * y_n1 - 1.0 / 3.0 * y_n +
                                       (2.0 / 3.0) * h * f(t_n2, y_plus));
      State<Scalar> F_minus = y_minus - (4.0 / 3.0 * y_n1 - 1.0 / 3.0 * y_n +
                                         (2.0 / 3.0) * h * f(t_n2, y_minus));
      J.col(j) = (F_plus - F_minus) / (2.0 * eps);
    }

    // 求解线性系统 J * delta = -F
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(J);
    State<Scalar> delta = lu.solve(-F);
    y += delta;
  }

  std::cerr << "BDF2: Newton iteration did not converge at t = " << t_n2
            << std::endl;
  return State<Scalar>(); // 返回空向量表示失败
}

// 带复用和残差收敛慢hint的 ，但是在少量数据下由于要复制性能不如自动向量化重新计算;
template <class Scalar>
State<Scalar>
BDF2Solver<Scalar>::dyn_eps_bdf2_step(const RHSFunc<Scalar> &f, Scalar t_n,
                                       const State<Scalar> &y_n, Scalar t_n1,
                                       const State<Scalar> &y_n1, Scalar h) {
    Scalar t_n2 = t_n1 + h;
    int n = y_n.size();
    State<Scalar> y = y_n1;                 // 初始猜测
    State<Scalar> F, delta;
    Eigen::PartialPivLU<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> lu;
    bool jac_uptodate = false;

    for (int iter = 0; iter < max_iter_; ++iter) {
        F = y - (4.0/3.0 * y_n1 - 1.0/3.0 * y_n + (2.0/3.0) * h * f(t_n2, y));
        Scalar res_norm = F.norm();
        if (res_norm < newton_tol_) {
            return y;
        }

        // 仅在第一次迭代或雅可比需要更新时计算
        if (!jac_uptodate) {
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> J(n, n);
            for (int j = 0; j < n; ++j) {
                Scalar eps = this->__eps_policy(y, j, res_norm);
                State<Scalar> y_plus = y, y_minus = y;
                y_plus(j) += eps; y_minus(j) -= eps;
                State<Scalar> F_plus = y_plus - (4.0/3.0 * y_n1 - 1.0/3.0 * y_n +
                                                 (2.0/3.0) * h * f(t_n2, y_plus));
                State<Scalar> F_minus = y_minus - (4.0/3.0 * y_n1 - 1.0/3.0 * y_n +
                                                   (2.0/3.0) * h * f(t_n2, y_minus));
                J.col(j) = (F_plus - F_minus) / (2.0 * eps);
            }
            lu.compute(J);
            jac_uptodate = true;
        }

        delta = lu.solve(-F);
        if (delta.array().isNaN().any()) {
            std::cerr << "BDF2: singular Jacobian at t=" << t_n2 << std::endl;
            return State<Scalar>();
        }
        y += delta;

        // 可选：检测收敛速度，若残差下降过慢则标记雅可比需更新
        if (iter > 0 && res_norm > 0.5 * prev_res_norm) {
            jac_uptodate = false;   // 下次迭代重新计算雅可比
        }
        prev_res_norm = res_norm;
    }
    std::cerr << "BDF2: Newton did not converge at t=" << t_n2 << std::endl;
    return State<Scalar>();
}

// ======================================================================
// IRK2 实现 (二阶隐式 Runge‑Kutta)
// ======================================================================
template <typename Scalar>
IRK2Solver<Scalar>::IRK2Solver(Method method, Scalar newton_tol, int max_iter)
    : method_(method), newton_tol_(newton_tol), max_iter_(max_iter) {}

template <typename Scalar>
void IRK2Solver<Scalar>::solve(const RHSFunc<Scalar> &f, Scalar t0, Scalar t1,
                               const State<Scalar> &y0, Scalar h0,
                               std::vector<Scalar> &times,
                               std::vector<State<Scalar>> &states) {
  times.clear();
  states.clear();
  Scalar t = t0;
  State<Scalar> y = y0;
  Scalar h = h0;
  times.push_back(t);
  states.push_back(y);

  while (t < t1 - 1e-12) {
    if (t + h > t1)
      h = t1 - t;

    auto result = step(f, t, y, h);
    if (result.y_next.size() == 0) {
      // 失败：尝试减半步长
      h *= 0.5;
      if (h < 1e-15) {
        std::cerr << "IRK2Solver: step size too small, aborting at t=" << t
                  << std::endl;
        break;
      }
      std::cerr << "IRK2Solver: step failed, retrying with h=" << h
                << std::endl;
      continue;
    }

    t += h;
    y = std::move(result.y_next);
    times.push_back(t);
    states.push_back(y);
  }
}

template <typename Scalar>
typename IRK2Solver<Scalar>::StepResult
IRK2Solver<Scalar>::step(const RHSFunc<Scalar> &f, Scalar t,
                         const State<Scalar> &y, Scalar h) {
  if (y.size() == 0 || !f) {
    return {State<Scalar>(), 0, h};
  }

  const int n = y.size();
  const Scalar c = 0.5, a = 0.5;

  /*

  // 使用零阶预测器（初值取当前状态），避免显式欧拉产生过大估计
  State<Scalar> y_next = y;
  State<Scalar> delta(n), residual(n);
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> J(n, n);

const Scalar eps = Scalar(1e-8);   // 固定绝对扰动，适应跨尺度


  */

  State<Scalar> y_next = y + h * f(t, y); // 显式 Euler 预测（比零阶好）
  State<Scalar> delta(n), residual(n);
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> J(n, n);

  for (int iter = 0; iter < max_iter_; ++iter) {
    Scalar t_mid = t + 0.5 * h;
    State<Scalar> y_mid = 0.5 * (y + y_next);
    residual = y_next - (y + h * f(t_mid, y_mid));

    if (residual.norm() < newton_tol_) {
      return {std::move(y_next), 0, h};
    }

    // === 关键修改：用策略计算 eps ===
    for (int j = 0; j < n; ++j) {
      Scalar eps = this->__eps_policy(y_next, j, residual.norm());

      State<Scalar> y_plus = y_next, y_minus = y_next;
      y_plus(j) += eps;
      y_minus(j) -= eps;

      State<Scalar> y_mid_plus = 0.5 * (y + y_plus);
      State<Scalar> y_mid_minus = 0.5 * (y + y_minus);
      State<Scalar> res_plus = y_plus - (y + h * f(t_mid, y_mid_plus));
      State<Scalar> res_minus = y_minus - (y + h * f(t_mid, y_mid_minus));
      J.col(j) = (res_plus - res_minus) / (2 * eps);
    }
    Eigen::PartialPivLU<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        lu(J);
    delta = lu.solve(-residual);
    if (delta.array().isNaN().any()) {
      std::cerr << "IRK2: singular Jacobian at t=" << t << ", iter=" << iter
                << std::endl;
      return {State<Scalar>(), 0, h};
    }
    y_next += delta;
  }

  /*

   for (int iter = 0; iter < max_iter_; ++iter) {
      Scalar t_mid = t + c * h;
      State<Scalar> y_mid = (1 - a) * y + a * y_next;
      residual = y_next - (y + h * f(t_mid, y_mid));

      if (residual.norm() < newton_tol_) {
        return {std::move(y_next), 0, h};
      }

      // 数值雅可比：固定绝对扰动
      for (int j = 0; j < n; ++j) {
        State<Scalar> y_plus = y_next, y_minus = y_next;
        y_plus(j) += eps;
        y_minus(j) -= eps;

        State<Scalar> y_mid_plus = (1 - a) * y + a * y_plus;
        State<Scalar> y_mid_minus = (1 - a) * y + a * y_minus;
        State<Scalar> res_plus = y_plus - (y + h * f(t_mid, y_mid_plus));
        State<Scalar> res_minus = y_minus - (y + h * f(t_mid, y_mid_minus));
        J.col(j) = (res_plus - res_minus) / (2 * eps);
      }

      Eigen::PartialPivLU<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          lu(J);
      delta = lu.solve(-residual);
      if (delta.array().isNaN().any()) {
        std::cerr << "IRK2: singular Jacobian at t=" << t << ", iter=" << iter
                  << std::endl;
        return {State<Scalar>(), 0, h};
      }
      y_next += delta;
    }


  */

  std::cerr << "IRK2: Startup iteration did not converge at t=" << t
            << std::endl;
  return {State<Scalar>(), 0, h};
}



// ======================================================================
// 半隐式 Euler（隐式 Euler）实现
// ======================================================================
template<class Scalar>
SemiImplicitEulerSolver<Scalar>::SemiImplicitEulerSolver(Scalar newton_tol, int max_iter)
    : newton_tol_(newton_tol), max_iter_(max_iter) {}

template<class Scalar>
void SemiImplicitEulerSolver<Scalar>::solve(const RHSFunc<Scalar>& f,
                                            Scalar t0, Scalar t1,
                                            const State<Scalar>& y0, Scalar h0,
                                            std::vector<Scalar>& times,
                                            std::vector<State<Scalar>>& states) {
    times.clear();
    states.clear();

    Scalar t = t0;
    State<Scalar> y = y0;
    Scalar h = h0;          // 固定步长（简单起见，也可以接受自适应，但这里仅演示固定步长）
    times.push_back(t);
    states.push_back(y);

    while (t < t1) {
        if (t + h > t1) h = t1 - t;
        State<Scalar> y_next = implicit_euler_step(f, t, h, y);
        t += h;
        y = std::move(y_next);
        times.push_back(t);
        states.push_back(y);
    }
}

template<class Scalar>
State<Scalar> SemiImplicitEulerSolver<Scalar>::implicit_euler_step(const RHSFunc<Scalar>& f,
                                                                    Scalar t, Scalar h,
                                                                    const State<Scalar>& y_curr) {
    int n = y_curr.size();
    State<Scalar> y = y_curr;   // 初始猜测

    // 隐式方程: y = y_curr + h * f(t+h, y)
    for (int iter = 0; iter < max_iter_; ++iter) {
        State<Scalar> F = y - y_curr - h * f(t + h, y);
        if (F.norm() < newton_tol_) {
            return y;
        }

        // 数值 Jacobian
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> J(n, n);
        Scalar eps = 1e-8;
        for (int j = 0; j < n; ++j) {
            State<Scalar> y_plus = y, y_minus = y;
            y_plus(j) += eps;
            y_minus(j) -= eps;
            State<Scalar> F_plus = y_plus - y_curr - h * f(t + h, y_plus);
            State<Scalar> F_minus = y_minus - y_curr - h * f(t + h, y_minus);
            J.col(j) = (F_plus - F_minus) / (2.0 * eps);
        }

        Eigen::PartialPivLU<decltype(J)> lu(J);
        State<Scalar> delta = lu.solve(-F);
        y += delta;
    }

    std::cerr << "SemiImplicitEuler: Newton iteration did not converge at t = " << t + h << std::endl;
    return State<Scalar>();   // 失败返回空向量
}
