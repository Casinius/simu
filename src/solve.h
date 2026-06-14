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

template<typename Scalar>
using SecondOrderRHSFunc = std::function<State<Scalar>(Scalar, const State<Scalar>&, const State<Scalar>&)>;
template<typename Scalar>
class SecondOrderODESolver {
public:
    using State = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    virtual ~SecondOrderODESolver() = default;

    /**
     * 求解二阶初值问题
     * @param rhs       右端函数，返回加速度 a = f(t, q, v)
     * @param t0        起始时间
     * @param t1        结束时间
     * @param q0        初始位移
     * @param v0        初始速度
     * @param h0        固定步长
     * @param times     输出时间点
     * @param states_q  输出位移序列
     * @param states_v  输出速度序列
     */
    virtual void solve(const SecondOrderRHSFunc<Scalar>& rhs,
                       Scalar t0, Scalar t1,
                       const State& q0, const State& v0, Scalar h0,
                       std::vector<Scalar>& times,
                       std::vector<State>& states_q,
                       std::vector<State>& states_v) = 0;
};
template<typename Scalar>
class GeneralizedAlphaSolver : public SecondOrderODESolver<Scalar> {
public:
    using State = typename SecondOrderODESolver<Scalar>::State;
    using SecondOrderRHSFunc = typename SecondOrderODESolver<Scalar>::SecondOrderRHSFunc;

    /**
     * 构造函数
     * @param rho_inf  高频耗散参数 ∈ [0,1]；0 = 最大耗散，1 = 无耗散（保守）
     * @param newton_tol  牛顿迭代容差
     * @param max_iter    最大牛顿迭代次数
     */
    explicit GeneralizedAlphaSolver(Scalar rho_inf = 0.5,
                                    Scalar newton_tol = 1e-10,
                                    int max_iter = 30);

    void solve(const SecondOrderRHSFunc& rhs,
               Scalar t0, Scalar t1,
               const State& q0, const State& v0, Scalar h0,
               std::vector<Scalar>& times,
               std::vector<State>& states_q,
               std::vector<State>& states_v) override;

private:
    // 计算算法参数
    void compute_parameters(Scalar rho_inf);

    // 单步更新 (隐式求解加速度 a_{n+1})
    bool step(const SecondOrderRHSFunc& rhs,
              Scalar t_n, Scalar dt,
              const State& q_n, const State& v_n, const State& a_n,
              State& q_n1, State& v_n1, State& a_n1);

    Scalar rho_inf_;
    Scalar alpha_m_, alpha_f_, beta_, gamma_;
    Scalar newton_tol_;
    int max_iter_;
};

#endif // ODE_SOLVER_H
