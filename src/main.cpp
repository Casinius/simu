#include "Eigen/Core"

#include "solve.h"
#include "solve_config.h"

#include <iostream>



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
  bdf2.solve(rhs, 0.0, 5.0, y0, 1e-3, times, states);
  std::cout << "BDF2 steps: " << times.size() << " steps\n";
  std::cout << "BDF2 res: " << states.at(states.size() - 1) << "\n";

  return 0;
}
