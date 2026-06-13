// ODESolver.cpp
#include "solve.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// ======================================================================
// RK45 实现
// ======================================================================
// Butcher 表系数 (Fehlberg 4(5))
namespace {
    const double a21 = 1.0/4.0;
    const double a31 = 3.0/32.0,   a32 = 9.0/32.0;
    const double a41 = 1932.0/2197.0, a42 = -7200.0/2197.0, a43 = 7296.0/2197.0;
    const double a51 = 439.0/216.0, a52 = -8.0, a53 = 3680.0/513.0, a54 = -845.0/4104.0;
    const double a61 = -8.0/27.0, a62 = 2.0, a63 = -3544.0/2565.0, a64 = 1859.0/4104.0, a65 = -11.0/40.0;

    const double c2 = 1.0/4.0, c3 = 3.0/8.0, c4 = 12.0/13.0, c5 = 1.0, c6 = 1.0/2.0;

    const double b4_1 = 25.0/216.0,   b4_3 = 1408.0/2565.0, b4_4 = 2197.0/4104.0, b4_5 = -1.0/5.0;
    const double b5_1 = 16.0/135.0,   b5_3 = 6656.0/12825.0, b5_4 = 28561.0/56430.0, b5_5 = -9.0/50.0, b5_6 = 2.0/55.0;
}

RK45Solver::RK45Solver(double atol, double rtol,
                       double h_min, double h_max,
                       double safety, double fac_min, double fac_max)
    : atol_(atol), rtol_(rtol), h_min_(h_min), h_max_(h_max),
      safety_(safety), fac_min_(fac_min), fac_max_(fac_max) {}

void RK45Solver::solve(const RHSFunc& f,
                       double t0, double t1,
                       const State& y0, double h0,
                       std::vector<double>& times,
                       std::vector<State>& states) {

    times.clear();
    states.clear();

    //std::cout << "RK45: h_min_ = " << h_min_ << ", h_max_ = " << h_max_ << std::endl;
    double t = t0;
    State y = y0;
    double h = std::min(h0, h_max_);
    times.push_back(t);
    states.push_back(y);

    while (t < t1) {
        // 保证最后一步不越过终点
        if (t + h > t1) h = t1 - t;
        if (h < h_min_) {
            std::cerr << "RK45: step size below minimum! t = " << t << ", h = " << h << std::endl;
            break;
        }

        auto [y_next, error, h_next] = step(f, t, y, h);

        // 误差评估（混合相对/绝对误差）
        double scale = atol_ + rtol_ * y.cwiseAbs().maxCoeff();
        double err_norm = error / scale;

        if (err_norm <= 1.0) {  // 接受该步
            t += h;
            y = std::move(y_next);
            times.push_back(t);
            states.push_back(y);
            // 调整下一步长
            if (t >= t1) break;
            double fac = safety_ * std::pow(err_norm, -0.2);
            fac = std::clamp(fac, fac_min_, fac_max_);
            h = std::min(h * fac, h_max_);
        } else {  // 拒绝，缩小步长重试
            double fac = safety_ * std::pow(err_norm, -0.25);
            fac = std::clamp(fac, fac_min_, 0.9);
            h = std::max(h * fac, h_min_);
        }
    }
}

RK45Solver::StepResult RK45Solver::step(const RHSFunc& f, double t, const State& y, double h) {
    State k1 = f(t, y);
    State k2 = f(t + c2 * h, y + h * (a21 * k1));
    State k3 = f(t + c3 * h, y + h * (a31 * k1 + a32 * k2));
    State k4 = f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3));
    State k5 = f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4));
    State k6 = f(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5));

    State y4 = y + h * (b4_1 * k1 + b4_3 * k3 + b4_4 * k4 + b4_5 * k5);   // 4阶近似
    State y5 = y + h * (b5_1 * k1 + b5_3 * k3 + b5_4 * k4 + b5_5 * k5 + b5_6 * k6); // 5阶近似

    double error = (y5 - y4).norm();   // 误差估计的范数
    return {std::move(y5), error, h};
}

// ======================================================================
// BDF2 实现
// ======================================================================
BDF2Solver::BDF2Solver(double newton_tol, int max_iter)
    : newton_tol_(newton_tol), max_iter_(max_iter) {}

void BDF2Solver::solve(const RHSFunc& f,
                       double t0, double t1,
                       const State& y0, double h0,
                       std::vector<double>& times,
                       std::vector<State>& states) {
    times.clear();
    states.clear();

    double h = h0;                 // BDF2 使用固定步长
    double t = t0;
    State y = y0;
    times.push_back(t);
    states.push_back(y);

    // 第一步: 使用 RK45 获取第二个点 y1 (也可以用任何单步显式方法)
    RK45Solver rk45(1e-8, 1e-6, h*(1e-4), h);
    std::vector<double> temp_t;
    std::vector<State> temp_y;
    rk45.solve(f, t, t + h, y, h, temp_t, temp_y);
    if (temp_y.size() < 2) {
        std::cerr << "BDF2: failed to generate initial point using RK45.\n";
        return;
    }
    double t1_bdf = t + h;
    State y1 = temp_y.back();
    times.push_back(t1_bdf);
    states.push_back(y1);

    t = t1_bdf;
    State y_n = y;      // y_{n}
    State y_n1 = y1;    // y_{n+1}
    double t_n = t0;
    double t_n1 = t;

    // 后续所有步进均使用 BDF2
    while (t_n1 < t1) {
        double t_n2 = t_n1 + h;
        if (t_n2 > t1) {
            // 最后一段不完整步长做特殊处理：缩短步长并重算（简化，也可用 RK45 完成）
            h = t1 - t_n1;
            t_n2 = t1;
        }
        State y_n2 = bdf2_step(f, t_n, y_n, t_n1, y_n1, h);
        if (y_n2.size() == 0) break;  // 牛顿迭代失败

        times.push_back(t_n2);
        states.push_back(y_n2);

        // 更新滑动窗口
        y_n = y_n1;
        y_n1 = y_n2;
        t_n = t_n1;
        t_n1 = t_n2;
    }
}

BDF2Solver::State BDF2Solver::bdf2_step(const RHSFunc& f, double t_n, const State& y_n,
                            double t_n1, const State& y_n1, double h) {
    double t_n2 = t_n1 + h;
    int n = y_n.size();
    State y = y_n1;   // 初始猜测

    // 牛顿迭代求解隐式方程:
    // F(y) = y - [ 4/3 y_{n+1} - 1/3 y_n + 2/3 h f(t_{n+2}, y) ] = 0
    for (int iter = 0; iter < max_iter_; ++iter) {
        State F = y - (4.0/3.0 * y_n1 - 1.0/3.0 * y_n + (2.0/3.0) * h * f(t_n2, y));

        if (F.norm() < newton_tol_) {
            return y;   // 收敛
        }

        // 数值 Jacobian: J(i,j) = dF_i / dy_j
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n);
        double eps = 1e-8;
        for (int j = 0; j < n; ++j) {
            State y_plus = y;
            y_plus(j) += eps;
            State y_minus = y;
            y_minus(j) -= eps;

            State F_plus = y_plus - (4.0/3.0 * y_n1 - 1.0/3.0 * y_n + (2.0/3.0) * h * f(t_n2, y_plus));
            State F_minus = y_minus - (4.0/3.0 * y_n1 - 1.0/3.0 * y_n + (2.0/3.0) * h * f(t_n2, y_minus));
            J.col(j) = (F_plus - F_minus) / (2.0 * eps);
        }

        // 求解线性系统 J * delta = -F
        Eigen::PartialPivLU<Eigen::MatrixXd> lu(J);
        State delta = lu.solve(-F);
        y += delta;
    }

    std::cerr << "BDF2: Newton iteration did not converge at t = " << t_n2 << std::endl;
    return State();   // 返回空向量表示失败
}