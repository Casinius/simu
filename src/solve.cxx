// ODESolver.cpp
#include "solve.h"
//#include "solve_config.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <ratio>
#pragma once
// ======================================================================
// RK45 实现
// ======================================================================
// Butcher 表系数 (Fehlberg 4(5))

namespace rk45_const {
    const auto a21 = 1.0/4.0;
    const auto a31 = 3.0/32.0,   a32 = 9.0/32.0;
    const auto a41 = 1932.0/2197.0, a42 = -7200.0/2197.0, a43 = 7296.0/2197.0;
    const auto a51 = 439.0/216.0, a52 = -8.0, a53 = 3680.0/513.0, a54 = -845.0/4104.0;
    const auto a61 = -8.0/27.0, a62 = 2.0, a63 = -3544.0/2565.0, a64 = 1859.0/4104.0, a65 = -11.0/40.0;

    const auto c2 = 1.0/4.0, c3 = 3.0/8.0, c4 = 12.0/13.0, c5 = 1.0, c6 = 1.0/2.0;

    const auto b4_1 = 25.0/216.0,   b4_3 = 1408.0/2565.0, b4_4 = 2197.0/4104.0, b4_5 = -1.0/5.0;
    const auto b5_1 = 16.0/135.0,   b5_3 = 6656.0/12825.0, b5_4 = 28561.0/56430.0, b5_5 = -9.0/50.0, b5_6 = 2.0/55.0;
}
template<class Scalar>
RK45Solver<Scalar>::RK45Solver(Scalar atol, Scalar rtol,
                       Scalar h_min, Scalar h_max,
                       Scalar safety, Scalar fac_min, Scalar fac_max)
    : atol_(atol), rtol_(rtol), h_min_(h_min), h_max_(h_max),
      safety_(safety), fac_min_(fac_min), fac_max_(fac_max) {}

template<class Scalar>
void RK45Solver<Scalar>::solve(const RHSFunc<Scalar>& f,
                       Scalar t0, Scalar t1,
                       const State<Scalar>& y0, Scalar h0,
                       std::vector<Scalar>& times,
                       std::vector<State<Scalar>>& states) {

    times.clear();
    states.clear();

    //std::cout << "RK45: h_min_ = " << h_min_ << ", h_max_ = " << h_max_ << std::endl;
    Scalar t = t0;
    State<Scalar> y = y0;
    Scalar h = std::min(h0, h_max_);
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
        Scalar scale = atol_ + rtol_ * y.cwiseAbs().maxCoeff();
        Scalar err_norm = error / scale;

        if (err_norm <= 1.0) {  // 接受该步
            t += h;
            y = std::move(y_next);
            times.push_back(t);
            states.push_back(y);
            // 调整下一步长
            if (t >= t1) break;
            Scalar fac = safety_ * std::pow(err_norm, -0.2);
            fac = std::clamp(fac, fac_min_, fac_max_);
            h = std::min(h * fac, h_max_);
        } else {  // 拒绝，缩小步长重试
            Scalar fac = safety_ * std::pow(err_norm, -0.25);
            fac = std::clamp(fac, fac_min_, 0.9);
            h = std::max(h * fac, h_min_);
        }
    }
}
template<class Scalar>
typename RK45Solver<Scalar>::StepResult RK45Solver<Scalar>::step(const RHSFunc<Scalar>& f, Scalar t, const State<Scalar>& y, Scalar h) {
    using namespace rk45_const;
    State<Scalar> k1 = f(t, y);
    State<Scalar> k2 = f(t + c2 * h, y + h * (a21 * k1));
    State<Scalar> k3 = f(t + c3 * h, y + h * (a31 * k1 + a32 * k2));
    State<Scalar> k4 = f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3));
    State<Scalar> k5 = f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4));
    State<Scalar> k6 = f(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5));

    State<Scalar> y4 = y + h * (b4_1 * k1 + b4_3 * k3 + b4_4 * k4 + b4_5 * k5);   // 4阶近似
    State<Scalar> y5 = y + h * (b5_1 * k1 + b5_3 * k3 + b5_4 * k4 + b5_5 * k5 + b5_6 * k6); // 5阶近似

    Scalar error = (y5 - y4).norm();   // 误差估计的范数
    return {std::move(y5), error, h};
}


namespace tist45_const {
    // c 系数
    const auto c2 = 0.161;
    const auto c3 = 0.327;
    const auto c4 = 0.9;
    const auto c5 = 0.980025;
    const auto c6 = 1.0;
    const auto c7 = 1.0;   // 与 c6 相同，FSAL 特性

    // a 矩阵 (下三角)
    const auto a21 = 0.161;
    const auto a31 = 0.0;
    const auto a32 = 0.327;
    const auto a41 = 0.923076923076923;
    const auto a42 = -0.323076923076923;
    const auto a43 = 0.3;
    const auto a51 = 0.152266367435797;
    const auto a52 = -0.123256385228215;
    const auto a53 = 0.409722222222222;
    const auto a54 = 0.541292357570418;
    const auto a61 = 0.067783106173768;
    const auto a62 = 0.24777275020553;
    const auto a63 = 0.137777777777778;
    const auto a64 = 0.0210178144254;
    const auto a65 = 0.525648610417524;
    const auto a71 = 0.048255588106345;
    const auto a72 = 0.174543926757248;
    const auto a73 = 0.057777777777778;
    const auto a74 = -0.04601475954772;
    const auto a75 = 0.162914043198731;
    const auto a76 = 0.602523379707618;

    // 5阶权重 (用于推进)
    const auto b5_1 = 0.096460766818065;
    const auto b5_2 = 0.185622075808907;
    const auto b5_3 = 0.144444444444444;
    const auto b5_4 = 0.025714285714286;
    const auto b5_5 = 0.077950646243423;
    const auto b5_6 = 0.338088259600815;
    const auto b5_7 = 0.131719532372261;

    // 4阶权重 (用于误差估计)
    const auto b4_1 = 0.121884775838502;
    const auto b4_2 = 0.187537998392598;
    const auto b4_3 = 0.155555555555556;
    const auto b4_4 = -0.007246960641757;
    const auto b4_5 = 0.156691159147222;
    const auto b4_6 = 0.253422102595331;
    const auto b4_7 = 0.133155369112548;
}

template<class Scalar>
TSIT45Resolver<Scalar>::TSIT45Resolver(Scalar atol, Scalar rtol,
                       Scalar h_min, Scalar h_max,
                       Scalar safety, Scalar fac_min, Scalar fac_max)
    : atol_(atol), rtol_(rtol), h_min_(h_min), h_max_(h_max),
      safety_(safety), fac_min_(fac_min), fac_max_(fac_max) {}

template<class Scalar>
void TSIT45Resolver<Scalar>::solve(const RHSFunc<Scalar>& f,
                       Scalar t0, Scalar t1,
                       const State<Scalar>& y0, Scalar h0,
                       std::vector<Scalar>& times,
                       std::vector<State<Scalar>>& states) {

    times.clear();
    states.clear();

    //std::cout << "RK45: h_min_ = " << h_min_ << ", h_max_ = " << h_max_ << std::endl;
    Scalar t = t0;
    State<Scalar> y = y0;
    Scalar h = std::min(h0, h_max_);
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
        Scalar scale = atol_ + rtol_ * y.cwiseAbs().maxCoeff();
        Scalar err_norm = error / scale;

        if (err_norm <= 1.0) {  // 接受该步
            t += h;
            y = std::move(y_next);
            times.push_back(t);
            states.push_back(y);
            // 调整下一步长
            if (t >= t1) break;
            Scalar fac = safety_ * std::pow(err_norm, -0.2);
            fac = std::clamp(fac, fac_min_, fac_max_);
            h = std::min(h * fac, h_max_);
        } else {  // 拒绝，缩小步长重试
            Scalar fac = safety_ * std::pow(err_norm, -0.25);
            fac = std::clamp(fac, fac_min_, 0.9);
            h = std::max(h * fac, h_min_);
        }
    }
}


template<class Scalar>
typename TSIT45Resolver<Scalar>::StepResult TSIT45Resolver<Scalar>::step(const RHSFunc<Scalar>& f, Scalar t, const State<Scalar>& y, Scalar h){
    using namespace tist45_const;
    State<Scalar> k1 = f(t, y);
    State<Scalar> k2 = f(t + c2 * h, y + h * (a21 * k1));
    State<Scalar> k3 = f(t + c3 * h, y + h * (a31 * k1 + a32 * k2));
    State<Scalar> k4 = f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3));
    State<Scalar> k5 = f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4));
    State<Scalar> k6 = f(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5));

    State<Scalar> y4 = y + h * (b4_1 * k1 + b4_3 * k3 + b4_4 * k4 + b4_5 * k5);   // 4阶近似
    State<Scalar> y5 = y + h * (b5_1 * k1 + b5_3 * k3 + b5_4 * k4 + b5_5 * k5 + b5_6 * k6); // 5阶近似

    Scalar error = (y5 - y4).norm();   // 误差估计的范数
    return {std::move(y5), error, h};
}










// ======================================================================
// BDF2 实现
// ======================================================================
template<class Scalar>
BDF2Solver<Scalar>::BDF2Solver(Scalar newton_tol, int max_iter)
    : newton_tol_(newton_tol), max_iter_(max_iter) {}
template<class Scalar>
void BDF2Solver<Scalar>::solve(const RHSFunc<Scalar>& f,
                       Scalar t0, Scalar t1,
                       const State<Scalar>& y0, Scalar h0,
                       std::vector<Scalar>& times,
                       std::vector<State<Scalar>>& states) {
    times.clear();
    states.clear();

    Scalar h = h0;                 // BDF2 使用固定步长
    Scalar t = t0;
    State<Scalar> y = y0;
    times.push_back(t);
    states.push_back(y);

    // 第一步: 使用 RK45 获取第二个点 y1 (也可以用任何单步显式方法)
    RK45Solver<Scalar> rk45(1e-8, 1e-6, h*(1e-4), h);
    std::vector<Scalar> temp_t;
    std::vector<State<Scalar>> temp_y;
    rk45.solve(f, t, t + h, y, h, temp_t, temp_y);
    if (temp_y.size() < 2) {
        std::cerr << "BDF2: failed to generate initial point using RK45.\n";
        return;
    }
    Scalar t1_bdf = t + h;
    State<Scalar> y1 = temp_y.back();
    times.push_back(t1_bdf);
    states.push_back(y1);

    t = t1_bdf;
    State<Scalar> y_n = y;      // y_{n}
    State<Scalar> y_n1 = y1;    // y_{n+1}
    Scalar t_n = t0;
    Scalar t_n1 = t;

    // 后续所有步进均使用 BDF2
    while (t_n1 < t1) {
        Scalar t_n2 = t_n1 + h;
        if (t_n2 > t1) {
            // 最后一段不完整步长做特殊处理：缩短步长并重算（简化，也可用 RK45 完成）
            h = t1 - t_n1;
            t_n2 = t1;
        }
        State<Scalar> y_n2 = bdf2_step(f, t_n, y_n, t_n1, y_n1, h);
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
template<class Scalar>
State<Scalar> BDF2Solver<Scalar>::bdf2_step(const RHSFunc<Scalar>& f, Scalar t_n, const State<Scalar>& y_n,
                    Scalar t_n1, const State<Scalar>& y_n1, Scalar h) {
    Scalar t_n2 = t_n1 + h;
    int n = y_n.size();
    State<Scalar> y = y_n1;   // 初始猜测

    // 牛顿迭代求解隐式方程:
    // F(y) = y - [ 4/3 y_{n+1} - 1/3 y_n + 2/3 h f(t_{n+2}, y) ] = 0
    for (int iter = 0; iter < max_iter_; ++iter) {
        State<Scalar> F = y - (4.0/3.0 * y_n1 - 1.0/3.0 * y_n + (2.0/3.0) * h * f(t_n2, y));

        if (F.norm() < newton_tol_) {
            return y;   // 收敛
        }

        // 数值 Jacobian: J(i,j) = dF_i / dy_j
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n);
        Scalar eps = 1e-8;
        for (int j = 0; j < n; ++j) {
            State<Scalar> y_plus = y;
            y_plus(j) += eps;
            State<Scalar> y_minus = y;
            y_minus(j) -= eps;

            State<Scalar> F_plus = y_plus - (4.0/3.0 * y_n1 - 1.0/3.0 * y_n + (2.0/3.0) * h * f(t_n2, y_plus));
            State<Scalar> F_minus = y_minus - (4.0/3.0 * y_n1 - 1.0/3.0 * y_n + (2.0/3.0) * h * f(t_n2, y_minus));
            J.col(j) = (F_plus - F_minus) / (2.0 * eps);
        }

        // 求解线性系统 J * delta = -F
        Eigen::PartialPivLU<Eigen::MatrixXd> lu(J);
        State<Scalar> delta = lu.solve(-F);
        y += delta;
    }

    std::cerr << "BDF2: Newton iteration did not converge at t = " << t_n2 << std::endl;
    return State<Scalar>();   // 返回空向量表示失败
}
