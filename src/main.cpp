// main_vehicle_example.cpp
// 示例：使用 VehicleDynamics 与 ODESolver 进行车辆仿真

#include "solve.h"
#include "solve_config.h"
#include "veh.cxx"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace Vehicle;

int main() {
    // 1. 创建车辆参数（使用默认中型轿车参数）
    VehicleParams params = makeDefaultVehicleParams<Scalar>();

    // 2. 创建动力学对象
    VehicleDynamics<Scalar> vehicle(params);
    
    // 3. 设置驾驶员输入（示例：正弦转向 + 恒定油门）
    DriverInput<Scalar> driver;
    driver.delta_cmd = 0.0;   // 初始转向角
    driver.throttle = 0.4;
    //driver.T_engine = 380.0;  // 发动机扭矩 N*m
    //driver.T_brake = {0, 0, 0, 0};
    
    vehicle.setDriverInput(driver);

    // 4. 设置路面（水平路面，附着系数0.8）
    RoadInput road;
    road.z_road = [](Scalar, Scalar) { return Scalar(0.1); };
    road.mu = [](Scalar, Scalar) { return Scalar(0.8); };
    vehicle.setRoadInput(road);

    // 5. 初始状态
    // 15-DOF 状态向量: [Vx, Vy, Vz, p, q, r, z_fR, z_fL, z_rR, z_rL, w_fR, w_fL, w_rR, w_rL, delta]
    State<Scalar> y0(15);
    y0 << 20.0,   // Vx = 20 m/s (72 km/h)
          0.0,    // Vy = 0
          0.0,    // Vz = 0
          0.0,    // p = 0
          0.0,    // q = 0
          0.0,    // r = 0
          0.0,    // z_fR = 0 (平衡位置)
          0.0,    // z_fL = 0
          0.0,    // z_rR = 0
          0.0,    // z_rL = 0
          20.0 / params.R_w,  // w_fR = Vx / R_w (无滑移)
          20.0 / params.R_w,  // w_fL
          20.0 / params.R_w,  // w_rR
          20.0 / params.R_w,  // w_rL
          0.0;    // delta = 0

    // 6. 获取 RHS 函数
    RHSFunc<Scalar> rhs = vehicle.getRHS();

    // 7. 使用 ODESolver 求解（这里用 RK45 自适应步长）
    //RK45Solver<Scalar> solver(1e-6, 1e-4, 1e-9, 0.01, 0.9, 0.2, 4.0);
    BDF2Solver<Scalar> solver;
    Scalar t0 = 0.0;
    Scalar t1 = 10.0;  // 仿真 10 秒
    Scalar h0 = 0.01; // 初始步长 10ms

    std::vector<Scalar> times;
    std::vector<State<Scalar>> states;

    std::cout << "开始仿真..." << std::endl;
    solver.solve(rhs, t0, t1, y0, h0, times, states);
    std::cout << "仿真完成，共 " << times.size() << " 步" << std::endl;

    // 8. 输出结果到 CSV
    std::ofstream csv("vehicle_simulation.csv");
    csv << "time,Vx,Vy,Vz,roll_rate,pitch_rate,yaw_rate,";
    csv << "z_fR,z_fL,z_rR,z_rL,";
    csv << "w_fR,w_fL,w_rR,w_rL,delta,";
    csv << "Fx_FL,Fx_FR,Fx_RL,Fx_RR,Fy_FL,Fy_FR,Fy_RL,Fy_RR,";
    csv << "Fz_FL,Fz_FR,Fz_RL,Fz_RR,";
    csv << "slip_ratio_FL,slip_ratio_FR,slip_ratio_RL,slip_ratio_RR,";
    csv << "slip_angle_FL,slip_angle_FR,slip_angle_RL,slip_angle_RR\n";

    for (size_t i = 0; i < times.size(); ++i) {
        const auto& y = states[i];

        // 计算派生量（轮胎力等）
        std::array<Scalar,4> Fz, Fx, Fy, kappa, alpha;
        vehicle.computeDerived(y, Fz, Fx, Fy, kappa, alpha);

        csv << times[i] << ",";
        for (int j = 0; j < 15; ++j) {
            csv << y(j) << ",";
        }
        for (int j = 0; j < 4; ++j) csv << Fx[j] << ",";
        for (int j = 0; j < 4; ++j) csv << Fy[j] << ",";
        for (int j = 0; j < 4; ++j) csv << Fz[j] << ",";
        for (int j = 0; j < 4; ++j) csv << kappa[j] << ",";
        for (int j = 0; j < 4; ++j) {
            csv << alpha[j];
            if (j < 3) csv << ",";
        }
        csv << "\n";
    }
    csv.close();
    std::cout << "结果已保存到 vehicle_simulation.csv" << std::endl;

    // 9. 简单统计
    if (!states.empty()) {
        const auto& y_final = states.back();
        std::cout << "\n=== 最终状态 ===" << std::endl;
        std::cout << "Vx = " << y_final(0) << " m/s" << std::endl;
        std::cout << "Vy = " << y_final(1) << " m/s" << std::endl;
        std::cout << "yaw_rate = " << y_final(5) << " rad/s" << std::endl;
        std::cout << "steering = " << y_final(14) << " rad" << std::endl;
    }

    return 0;
}
