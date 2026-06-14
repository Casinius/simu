#include "Eigen/Core"

#include "solve.h"
#include "solve_config.h"

#include <chrono>
#include <iostream>
#include <locale>
/* 一阶验证

int main() {


  // 示例微分方程: dy/dt = -y,  y(0) = 1  (解析解 y = e^{-t})
  auto rhs = [](Scalar t, const State<Scalar> &y) -> State<Scalar> {
    State<Scalar> f(1);
    f(0) = -1000 * y(0);
    return f;
  };

  State<Scalar> y0(1);
  y0 << 1.0;

  std::vector<Scalar> times;
  std::vector<State<Scalar>> states;

  // 使用 RK45
  RK45Solver<Scalar> r45(1e-8, 1e-6, 1e-20, 0.5);
  r45.solve(rhs, 0.0, 5.0, y0, 0.1, times, states);
  std::cout << "RK45: " << times.size() << " steps\n";
  std::cout << "RK45 res: " << states.at(states.size() - 1) << "\n";

  // 使用 BDF2 (固定步长 0.1)
  BDF2Solver<Scalar> bdf2(1e-10, 30);
  times.clear();
  states.clear();
  bdf2.solve(rhs, 0.0, 5.0, y0, 0.08, times, states);
  std::cout << "BDF2 steps: " << times.size() << " steps\n";
  std::cout << "BDF2 res: " << states.at(states.size() - 1) << "\n";

  return 0;
}


*/

// 将二阶系统改写为一阶系统，用于 RK45
// 状态向量 y = [x, v]^T
auto rhs_first_order = [](Scalar t, const State<Scalar> &y) -> State<Scalar> {
  const Scalar omega0 = 1e6;
  const Scalar zeta = 0.5;
  State<Scalar> f(2);
  f(0) = y(1);                                                 // dx/dt = v
  f(1) = -2.0 * zeta * omega0 * y(1) - omega0 * omega0 * y(0); // dv/dt = a
  return f;
};

int main() {
  using Scalar = double;
  using State = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  // 测试高刚度二阶系统转换的一阶形式
  auto rhs = [](Scalar t, const State &y) -> State {
    const Scalar omega0 = 1e6;
    const Scalar zeta = 0.5;
    State f(2);
    f(0) = y(1);
    f(1) = -2.0 * zeta * omega0 * y(1) - omega0 * omega0 * y(0);
    return f;
  };

  State y0(2);
  y0 << 1.0, 0.0;

  std::vector<Scalar> times;
  std::vector<State> states;


  


  IRK2Solver<Scalar> irk2(IRK2Solver<Scalar>::Method::Midpoint, 1e-10, 30);
  irk2.__eps_policy = eps_policy::hybrid<Scalar>();
  std::chrono::high_resolution_clock::time_point irk_s = std::chrono::high_resolution_clock::now();
  irk2.solve(rhs_first_order, 0.0, 0.001, y0, 1e-5, times, states);
  std::chrono::high_resolution_clock::time_point irk_e = std::chrono::high_resolution_clock::now();
  std::cout << "IRK2 time: " << irk_e - irk_s << std::endl;
  std::cout << "Final y: " << states.back().transpose() << std::endl;

  // 隐式中点法



  
  AdaptiveBDF2Solver<Scalar> ors(1e-10, 30);
  ors.__eps_policy= eps_policy::central_diff_optimal<Scalar>();
  times.clear();
  states.clear();
  std::chrono::high_resolution_clock::time_point bdf_s = std::chrono::high_resolution_clock::now();
  ors.solve(rhs_first_order, 0.0, 5.0, y0, 1e-3, times, states);
  std::chrono::high_resolution_clock::time_point bdf_e = std::chrono::high_resolution_clock::now();
  std::cout << "BDF2 time: " << bdf_e-bdf_s << "\n";
  std::cout << "BDF2 res: " << states.back().transpose() << "\n";
  
  
 

  return 0;
}