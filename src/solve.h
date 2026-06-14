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
template<typename Scalar>
using State = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
template<typename Scalar>
using RHSFunc = std::function<State<Scalar>(Scalar, const State<Scalar>&)>;  // dy/dt = f(t,y)


template<typename Scalar>
class ODESolver {
public:
                    // 状态向量类型


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
    virtual void solve(const RHSFunc<Scalar>& f,
                       Scalar t0, Scalar t1,
                       const State<Scalar>& y0, Scalar h0,
                       std::vector<Scalar>& times,
                       std::vector<State<Scalar>>& states) = 0;
};

// ----------------------------------------------------------------------
// RK45 自适应步长求解器
// ----------------------------------------------------------------------
template<typename Scalar>
class RK45Solver : public ODESolver<Scalar> {
public:
    RK45Solver(Scalar atol = 1e-8, Scalar rtol = 1e-6,
               Scalar h_min = 1e-12, Scalar h_max = 1.0,
               Scalar safety = 0.9, Scalar fac_min = 0.2, Scalar fac_max = 4.0);

    void solve(const RHSFunc<Scalar>& f,
               Scalar t0, Scalar t1,
               const State<Scalar>& y0, Scalar h0,
               std::vector<Scalar>& times,
               std::vector<State<Scalar>>& states) override;

private:
    // 单步积分，返回误差估计和推荐的下一时间步长
    struct StepResult {
        State<Scalar> y_next;   // 下一时刻状态（5阶精度的值）
        Scalar error;   // 估计的局部截断误差（标量范数）
        Scalar h_next;  // 建议的下一时间步长
    };
    StepResult step(const RHSFunc<Scalar>& f, Scalar t, const State<Scalar>& y, Scalar h);

    Scalar atol_, rtol_;      // 绝对/相对误差容限
    Scalar h_min_, h_max_;    // 步长约束
    Scalar safety_, fac_min_, fac_max_;  // 自适应系数
};

// ----------------------------------------------------------------------
// BDF2 隐式固定步长求解器（使用牛顿迭代 + 数值Jacobian）
// ----------------------------------------------------------------------
template<typename Scalar>
class BDF2Solver : public ODESolver<Scalar> {
public:
    explicit BDF2Solver(Scalar newton_tol = 1e-10, int max_iter = 30);

    void solve(const RHSFunc<Scalar>& f,
               Scalar t0, Scalar t1,
               const State<Scalar>& y0, Scalar h0,
               std::vector<Scalar>& times,
               std::vector<State<Scalar>>& states) override;

private:
    // 单步 BDF2：已知 y_{n}, y_{n+1} 和步长 h，求 y_{n+2}
    State<Scalar> bdf2_step(const RHSFunc<Scalar>& f, Scalar t_n, const State<Scalar>& y_n,
                    Scalar t_n1, const State<Scalar>& y_n1, Scalar h);

    Scalar newton_tol_;
    int max_iter_;
};


template<typename Scalar>
class TSIT45Resolver : public ODESolver<Scalar> {
public:
    TSIT45Resolver(Scalar atol = 1e-8, Scalar rtol = 1e-6,
               Scalar h_min = 1e-12, Scalar h_max = 1.0,
               Scalar safety = 0.9, Scalar fac_min = 0.2, Scalar fac_max = 4.0);

    void solve(const RHSFunc<Scalar>& f,
               Scalar t0, Scalar t1,
               const State<Scalar>& y0, Scalar h0,
               std::vector<Scalar>& times,
               std::vector<State<Scalar>>& states) override;

private:
    // 单步积分，返回误差估计和推荐的下一时间步长
    struct StepResult {
        State<Scalar> y_next;   // 下一时刻状态（5阶精度的值）
        Scalar error;   // 估计的局部截断误差（标量范数）
        Scalar h_next;  // 建议的下一时间步长
    };
    StepResult step(const RHSFunc<Scalar>& f, Scalar t, const State<Scalar>& y, Scalar h);

    Scalar atol_, rtol_;      // 绝对/相对误差容限
    Scalar h_min_, h_max_;    // 步长约束
    Scalar safety_, fac_min_, fac_max_;  // 自适应系数
};
#endif // ODE_SOLVER_H
