// main_vehicle_example.cpp
// 示例：使用 VehicleDynamics 与 ODESolver 进行车辆仿真

#include "precond_krylov.h"
#include "solve_config.h"
#include <cstddef>
#include <iostream>
#include <fstream>
#include <numeric>
#include <print>
#include <ranges>
#include <vector>
#include <cmath>
#include <tao/json.hpp>
Eigen::VectorX<Scalar> van_der_pol_rhs(Scalar t, const Eigen::VectorX<Scalar>& y) {
    const Scalar mu = 1000.0;   // 刚性参数
    Eigen::VectorX<Scalar> dydt(2);
    dydt(0) = y(1);
    dydt(1) = mu * (1 - y(0)*y(0)) * y(1) - y(0);
    return dydt;
}
State<Scalar> robertson(Scalar t, const State<Scalar>& y) {
    // y0, y1, y2
    Scalar k1 = 0.04;
    Scalar k2 = 3e7;
    Scalar k3 = 1e4;
    State<Scalar> dydt(3);
    dydt(0) = -k1*y(0) + k3*y(1)*y(2);
    dydt(1) = k1*y(0) - k2*y(1)*y(1) - k3*y(1)*y(2);
    dydt(2) = k2*y(1)*y(1);
    return dydt;
}
Eigen::VectorX<Scalar> double_pendulum_rhs(Scalar t, const Eigen::VectorX<Scalar>& y) {
    // y = [theta1, omega1, theta2, omega2]
    Scalar g = 9.81;
    Scalar m1 = 1.0, m2 = 1.0;
    Scalar L1 = 1.0, L2 = 1.0;

    Scalar dtheta1 = y(1);
    Scalar dtheta2 = y(3);

    Scalar delta = y(2) - y(0);
    Scalar den = (m1 + m2) * L1 - m2 * L1 * cos(delta) * cos(delta);
    Scalar domega1 = ( -m2 * L1 * y(1)*y(1) * sin(delta) * cos(delta)
                       - (m1 + m2) * g * sin(y(0))
                       - m2 * L2 * y(3)*y(3) * sin(delta) ) / den;
    Scalar domega2 = ( L1 * ( (m1 + m2) * g * sin(y(0)) * cos(delta)
                              + m2 * L1 * y(1)*y(1) * sin(delta)
                              + m2 * L2 * y(3)*y(3) * sin(delta) * cos(delta) )
                       - L1 * ( -m2 * L1 * y(1)*y(1) * sin(delta) * cos(delta)
                                - (m1 + m2) * g * sin(y(0))
                                - m2 * L2 * y(3)*y(3) * sin(delta) ) ) / (L1 * den);

    Eigen::VectorX<Scalar> dydt(4);
    dydt << dtheta1, domega1, dtheta2, domega2;
    return dydt;
}
// ======================================================================
// Krylov 框架自检 + ODE 求解器注入示例
// ======================================================================
static int krylov_selfcheck() {
  using pk::Vector;
  auto fail = [](const char *what) {
    std::cerr << "krylov selfcheck: FAIL (" << what << ")" << std::endl;
    std::exit(1);
  };

  // ---- 1. 合成 SPD 矩阵：PCG + 预条件对比 ----
  const int n = 200;
  Eigen::MatrixXd B = Eigen::MatrixXd::Random(n, n);
  // 对角跨 3 个数量级：保证 jacobi/ssor 相对 plain 有明确收益
  Eigen::VectorXd s(n);
  for (int i = 0; i < n; ++i)
    s(i) = std::pow(10.0, -3.0 * i / (n - 1));
  Eigen::MatrixXd Asp =
      s.asDiagonal() * (B.transpose() * B) * s.asDiagonal() +
      Eigen::MatrixXd::Identity(n, n);
  Vector<double> b = Vector<double>::Random(n);
  auto Aop = [&](const Vector<double> &v) { return Vector<double>(Asp * v); };

  Vector<double> x = Vector<double>::Zero(n);
  auto r_plain = pk::pcg<double>(Aop, b, x);
  if (!r_plain.converged || r_plain.relres > 1e-8)
    fail("pcg plain SPD");

  Vector<double> xj = Vector<double>::Zero(n);
  auto r_jac = pk::pcg<double>(Aop, b, xj, pk::jacobi<double>(Asp));
  if (!r_jac.converged || r_jac.relres > 1e-8)
    fail("pcg jacobi SPD");

  Vector<double> xs = Vector<double>::Zero(n);
  auto r_ssor = pk::pcg<double>(Aop, b, xs, pk::ssor<double>(Asp));
  if (!r_ssor.converged || r_ssor.relres > 1e-8)
    fail("pcg ssor SPD");

  std::cout << "pcg SPD iters: plain=" << r_plain.iterations
            << " jacobi=" << r_jac.iterations
            << " ssor=" << r_ssor.iterations << std::endl;
  if (!(r_jac.iterations < r_plain.iterations &&
        r_ssor.iterations < r_plain.iterations))
    fail("preconditioners do not reduce iterations on SPD");

  // ---- 2. 病态对角（跨 6 个数量级）：jacobi 必须显著优于无预条件 ----
  Eigen::VectorXd d(n);
  for (int i = 0; i < n; ++i)
    d(i) = std::pow(10.0, -6.0 * i / (n - 1));
  Eigen::MatrixXd Ddiag = d.asDiagonal();
  auto Dop = [&](const Vector<double> &v) { return Vector<double>(Ddiag * v); };
  pk::Params<double> wide;
  wide.max_iter = 100000;
  Vector<double> xi = Vector<double>::Zero(n);
  auto r_ill_plain = pk::pcg<double>(Dop, b, xi, nullptr, wide);
  Vector<double> xij = Vector<double>::Zero(n);
  auto r_ill_jac = pk::pcg<double>(Dop, b, xij, pk::jacobi<double>(d), wide);
  std::cout << "pcg ill-conditioned iters: plain=" << r_ill_plain.iterations
            << " jacobi=" << r_ill_jac.iterations << std::endl;
  if (!r_ill_jac.converged ||
      r_ill_jac.iterations >= r_ill_plain.iterations)
    fail("jacobi ineffective on ill-conditioned diagonal");

  // ---- 3. 非对称对角占优：bicgstab + gmres ----
  Eigen::MatrixXd C = Eigen::MatrixXd::Random(n, n) * 0.2 +
                      2.0 * Eigen::MatrixXd::Identity(n, n);
  auto Cop = [&](const Vector<double> &v) { return Vector<double>(C * v); };
  Vector<double> xb = Vector<double>::Zero(n);
  auto r_bicg = pk::bicgstab<double>(Cop, b, xb);
  if (!r_bicg.converged || r_bicg.relres > 1e-8)
    fail("bicgstab asymmetric");
  Vector<double> xg = Vector<double>::Zero(n);
  auto r_gm = pk::gmres<double>(Cop, b, xg);
  if (!r_gm.converged || r_gm.relres > 1e-8)
    fail("gmres asymmetric");
  std::cout << "asymmetric: bicgstab iters=" << r_bicg.iterations
            << " gmres iters=" << r_gm.iterations << std::endl;

  std::cout << "krylov selfcheck: PASS" << std::endl;
  return 0;
}

// 用注入的 PCG+jacobi 跑 van der Pol，比对 LU 与 Krylov 两配置的末端状态
static void ode_injection_demo() {
  State<Scalar> y0(2);
  y0 << 2.0, 0.0;
  Scalar t0 = 0.0, t1 = 2.0, h0 = 0.001;

  // 配置 A：默认 PartialPivLU
  SemiImplicitEulerSolver<Scalar> lu_solver;
  std::vector<Scalar> t_lu;
  std::vector<State<Scalar>> y_lu;
  lu_solver.solve(van_der_pol_rhs, t0, t1, y0, h0, t_lu, y_lu);

  // 配置 B：注入 PCG + jacobi 预条件（用户文档示例写法）
  SemiImplicitEulerSolver<Scalar> krylov_solver;
  krylov_solver.__linear_solve =
      [&](auto A, const auto &J, const State<Scalar> &rhs) {
        auto M = pk::jacobi<Scalar>(State<Scalar>(J.diagonal()));
        pk::Params<Scalar> p;
        p.rtol = 1e-8;
        p.max_iter = 300;
        auto x = rhs;
        x.setZero();
        auto r = pk::pcg(A, rhs, x, M, p);
        return r.converged ? x : State<Scalar>(); // 失败 → 空向量
      };
  std::vector<Scalar> t_kr;
  std::vector<State<Scalar>> y_kr;
  krylov_solver.solve(van_der_pol_rhs, t0, t1, y0, h0, t_kr, y_kr);

  if (y_lu.size() != y_kr.size() || y_lu.empty()) {
    std::cerr << "ode injection: FAIL (step count mismatch)" << std::endl;
    std::exit(1);
  }
  Scalar maxdiff = 0;
  for (size_t i = 0; i < y_lu.size(); ++i) {
    const int nn = std::min(y_lu[i].size(), y_kr[i].size());
    for (int j = 0; j < nn; ++j)
      maxdiff = std::max(maxdiff, std::abs(y_lu[i](j) - y_kr[i](j)));
  }
  std::cout << "ode injection: states=" << y_lu.size()
            << " max|LU-Krylov|=" << maxdiff << std::endl;
  if (maxdiff > 1e-6) {
    std::cerr << "ode injection: FAIL (maxdiff > 1e-6)" << std::endl;
    std::exit(1);
  }
}

int main(){
    krylov_selfcheck();
    ode_injection_demo();

    

    // 初始条件
    State<Scalar> y0(2);
    y0 << 1.0, 0.0;   // Robertson标准初值
    SemiImplicitEulerSolver<Scalar> solver;
    Scalar t0 = 0.0;
    Scalar t1 = 10.0;  // 仿真时间
    Scalar h0 = 0.001; // 初始步长

    std::vector<Scalar> times;
    std::vector<State<Scalar>> states;
    solver.solve(van_der_pol_rhs, t0, t1, y0, h0, times, states);
    tao::json::value root = tao::json::empty_array;
        for (size_t i:std::views::iota((size_t)0,times.size())) {
            root.emplace_back(tao::json::value{
                {"time", times.at(i)},
                {"state", states[i][0]}
            });
        }

        std::string json_str = tao::json::to_string(root, 4); // 生成带缩进的字符串

            std::ofstream file("output.json");
            file << json_str;
            file.close();
 //   std::println("{}",json_str);

}

