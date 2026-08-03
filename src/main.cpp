// main_vehicle_example.cpp
// 示例：使用 VehicleDynamics 与 ODESolver 进行车辆仿真

#include "solve.h"
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
int main(){
    // 初始条件
    State<Scalar> y0(2);
    y0 << 1.0, 0.0;   // Robertson标准初值
    VerletSolver<Scalar> solver;
    Scalar t0 = 0.0;
    Scalar t1 = 10.0;  // 仿真 10 秒
    Scalar h0 = 1; // 初始步长 10ms

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
}
