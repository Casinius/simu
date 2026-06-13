// ODESolver.h
#ifndef ODE_SOLVER_H
#define ODE_SOLVER_H

#include <Eigen/Dense>
#include <functional>
#include <vector>

/**
 * 抽象求解器基类
 * 所有具体求解器必须实现 solve() 方法
 */
class ODESolver {
public:
    using State = Eigen::VectorXd;                     // 状态向量类型
    using RHSFunc = std::function<State(double, const State&)>;  // dy/dt = f(t,y)

    virtual ~ODESolver() = default;

    /**
     * 求解初值问题
     * @param f       右端函数 f(t,y)
     * @param t0      起始时间
     * @param t1      结束时间
     * @param y0      初始状态
     * @param h0      初始步长（对固定步长方法为固定步长，对自适应方法仅为初始值）
     * @param times   输出时间点序列（会被清空并填充）
     * @param states  输出状态序列（与 times 对应）
     */
    virtual void solve(const RHSFunc& f,
                       double t0, double t1,
                       const State& y0, double h0,
                       std::vector<double>& times,
                       std::vector<State>& states) = 0;
};

// ----------------------------------------------------------------------
// RK45 自适应步长求解器
// ----------------------------------------------------------------------
class RK45Solver : public ODESolver {
public:
    RK45Solver(double atol = 1e-8, double rtol = 1e-6,
               double h_min = 1e-12, double h_max = 1.0,
               double safety = 0.9, double fac_min = 0.2, double fac_max = 4.0);

    void solve(const RHSFunc& f,
               double t0, double t1,
               const State& y0, double h0,
               std::vector<double>& times,
               std::vector<State>& states) override;

private:
    // 单步积分，返回误差估计和推荐的下一时间步长
    struct StepResult {
        State y_next;   // 下一时刻状态（5阶精度的值）
        double error;   // 估计的局部截断误差（标量范数）
        double h_next;  // 建议的下一时间步长
    };
    StepResult step(const RHSFunc& f, double t, const State& y, double h);

    double atol_, rtol_;      // 绝对/相对误差容限
    double h_min_, h_max_;    // 步长约束
    double safety_, fac_min_, fac_max_;  // 自适应系数
};

// ----------------------------------------------------------------------
// BDF2 隐式固定步长求解器（使用牛顿迭代 + 数值Jacobian）
// ----------------------------------------------------------------------
class BDF2Solver : public ODESolver {
public:
    explicit BDF2Solver(double newton_tol = 1e-10, int max_iter = 30);

    void solve(const RHSFunc& f,
               double t0, double t1,
               const State& y0, double h0,
               std::vector<double>& times,
               std::vector<State>& states) override;

private:
    // 单步 BDF2：已知 y_{n}, y_{n+1} 和步长 h，求 y_{n+2}
    State bdf2_step(const RHSFunc& f, double t_n, const State& y_n,
                    double t_n1, const State& y_n1, double h);

    double newton_tol_;
    int max_iter_;
};

#endif // ODE_SOLVER_H