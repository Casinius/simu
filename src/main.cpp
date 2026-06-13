#include "Eigen/Core"
#include "solve.h"
#include <iostream>
#include <fpm/fixed.hpp>
int main() {
  //using EMatType = Eigen::Matrix<fpm::fixed_8_24, Eigen::Dynamic, 1>;

    // 示例微分方程: dy/dt = -y,  y(0) = 1  (解析解 y = e^{-t})
    auto rhs = [](double t, const Eigen::VectorXd& y) -> Eigen::VectorXd {
        Eigen::VectorXd f(1);
        f(0) = -y(0);
        return f;
    };

    Eigen::VectorXd y0(1);
    y0 << 1.0;

    std::vector<double> times;
    std::vector<Eigen::VectorXd> states;

    // 使用 RK45
    RK45Solver rk45(1e-8, 1e-6, 1e-20, 0.5);
    rk45.solve(rhs, 0.0, 5.0, y0, 0.1, times, states);
    std::cout << "RK45: " << times.size() << " steps\n";
    std::cout << "RK45 res: "<< states.at(states.size()-1) << "\n";


    // 使用 BDF2 (固定步长 0.1)
    BDF2Solver bdf2(1e-10, 30);
    times.clear(); states.clear();
    bdf2.solve(rhs, 0.0, 5.0, y0, 0.1, times, states);
    std::cout << "BDF2 steps: " << times.size() << " steps\n";
    std::cout << "BDF2 res: "<< states.at(states.size()-1) << "\n";

    return 0;
}