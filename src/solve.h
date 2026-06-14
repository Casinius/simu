// ODESolver.h
#ifndef ODE_SOLVER_H
#define ODE_SOLVER_H

#include <Eigen/Dense>
#include <Eigen/LU>
#include <functional>
#include <vector>
/**
 * 抽象求解器基类
 * 所有具体求解器必须实现 solve() 方法
 */
template <typename Scalar>
using State = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;






template<typename Scalar>
using EpsPolicy = std::function<Scalar(const State<Scalar>& y, int j, Scalar f_norm)>;

namespace eps_policy {

// 1. 固定扰动（兼容你现在的 1e-8）
template<typename Scalar>
EpsPolicy<Scalar> fixed(Scalar eps) {
    return [eps](const State<Scalar>&, int, Scalar) -> Scalar { return eps; };
}

// 2. 相对扰动: eps_j = scale * max(1, |y_j|)
//    默认 scale = sqrt(eps_mach) ≈ 1.5e-8 (double)
template<typename Scalar>
EpsPolicy<Scalar> relative(Scalar scale = std::sqrt(std::numeric_limits<Scalar>::epsilon())) {
    return [scale](const State<Scalar>& y, int j, Scalar) -> Scalar {
        return scale * std::max(Scalar(1.0), std::abs(y(j)));
    };
}

// 3. 混合扰动: 同时考虑状态量和函数值尺度
//    适合 Pacejka 这种 f 值远大于 y 值的情况
template<typename Scalar>
EpsPolicy<Scalar> hybrid(Scalar scale = std::sqrt(std::numeric_limits<Scalar>::epsilon())) {
    return [scale](const State<Scalar>& y, int j, Scalar f_norm) -> Scalar {
        Scalar y_abs = std::abs(y(j));
        Scalar f_scale = (y.norm() > 0) ? (f_norm / y.norm()) : Scalar(1.0);
        return scale * std::max({Scalar(1.0), y_abs, f_scale});
    };
}

// 4. 中心差分理论最优: eps ~ eps_mach^(1/3) ≈ 6e-6 (double)
//    你的代码用中心差分，这是理论最优值
template<typename Scalar>
EpsPolicy<Scalar> central_diff_optimal() {
    const Scalar scale = std::cbrt(std::numeric_limits<Scalar>::epsilon());
    return [scale](const State<Scalar>& y, int j, Scalar) -> Scalar {
        return scale * std::max(Scalar(1.0), std::abs(y(j)));
    };
}

// 5. 车辆专用保守策略（强烈推荐用于 15DOF）
//    对小量（滑移率、离合器滑差）用绝对下限保护，防止 Pacejka 导数爆炸
//    对大量（转速）用相对扰动
template<typename Scalar>
EpsPolicy<Scalar> vehicle(Scalar scale = std::cbrt(std::numeric_limits<Scalar>::epsilon())) {
    return [scale](const State<Scalar>& y, int j, Scalar f_norm) -> Scalar {
        Scalar y_abs = std::abs(y(j));
        // 小量保护：即使 y_j=0，eps 也不会缩到机器精度以下
        Scalar base = std::max(Scalar(1e-6), y_abs);
        // 如果右端项范数极大（Pacejka 饱和区），适度放大 eps
        Scalar f_factor = (f_norm > 1e3) ? std::log10(f_norm) / 3.0 : Scalar(1.0);
        return scale * base * f_factor;
    };
}

// 6. 有界相对扰动: 传入每个分量的典型物理尺度
//    15DOF 推荐用这个！你可以根据车辆状态量纲配置
template<typename Scalar>
EpsPolicy<Scalar> bounded_relative(const std::vector<Scalar>& typical_scales,
                                    Scalar scale = std::cbrt(std::numeric_limits<Scalar>::epsilon())) {
    // 拷贝到 lambda 里保证生命周期
    return [typical_scales, scale](const State<Scalar>& y, int j, Scalar) -> Scalar {
        Scalar s = (j < (int)typical_scales.size()) ? typical_scales[j] : Scalar(1.0);
        return scale * std::max(s, std::abs(y(j)));
    };
}

} 





template <typename Scalar>
using RHSFunc = std::function<State<Scalar>(
    Scalar, const State<Scalar> &)>; // dy/dt = f(t,y)

template <typename Scalar> class ODESolver {
    
    public:
  // 状态向量类型
EpsPolicy<Scalar> __eps_policy= eps_policy::relative<Scalar>();
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
  virtual void solve(const RHSFunc<Scalar> &f, Scalar t0, Scalar t1,
                     const State<Scalar> &y0, Scalar h0,
                     std::vector<Scalar> &times,
                     std::vector<State<Scalar>> &states) = 0;

};

// ----------------------------------------------------------------------
// RK45 自适应步长求解器
// ----------------------------------------------------------------------
template <typename Scalar> class RK45Solver : public ODESolver<Scalar> {
public:
  RK45Solver(Scalar atol = 1e-8, Scalar rtol = 1e-6, Scalar h_min = 1e-12,
             Scalar h_max = 1.0, Scalar safety = 0.9, Scalar fac_min = 0.2,
             Scalar fac_max = 4.0);

  void solve(const RHSFunc<Scalar> &f, Scalar t0, Scalar t1,
             const State<Scalar> &y0, Scalar h0, std::vector<Scalar> &times,
             std::vector<State<Scalar>> &states);

private:
  // 单步积分，返回误差估计和推荐的下一时间步长
  struct StepResult {
    State<Scalar> y_next; // 下一时刻状态（5阶精度的值）
    Scalar error;         // 估计的局部截断误差（标量范数）
    Scalar h_next;        // 建议的下一时间步长
  };
  StepResult step(const RHSFunc<Scalar> &f, Scalar t, const State<Scalar> &y,
                  Scalar h);

  Scalar atol_, rtol_;                // 绝对/相对误差容限
  Scalar h_min_, h_max_;              // 步长约束
  Scalar safety_, fac_min_, fac_max_; // 自适应系数
};

// ----------------------------------------------------------------------
// BDF2 隐式固定步长求解器（使用牛顿迭代 + 数值Jacobian）
// ----------------------------------------------------------------------
template <typename Scalar> class BDF2Solver : public ODESolver<Scalar> {
public:
  explicit BDF2Solver(Scalar newton_tol = 1e-10, int max_iter = 30);

  void solve(const RHSFunc<Scalar> &f, Scalar t0, Scalar t1,
             const State<Scalar> &y0, Scalar h0, std::vector<Scalar> &times,
             std::vector<State<Scalar>> &states) override;

private:
  // 单步 BDF2：已知 y_{n}, y_{n+1} 和步长 h，求 y_{n+2}
  State<Scalar> bdf2_step(const RHSFunc<Scalar> &f, Scalar t_n,
                          const State<Scalar> &y_n, Scalar t_n1,
                          const State<Scalar> &y_n1, Scalar h);

  Scalar newton_tol_;
  int max_iter_;
  Scalar prev_res_norm;
};

template <typename Scalar> class IRK2Solver : public ODESolver<Scalar> {
public:
  // Method 选择: Midpoint (隐式中点) 或 Trapezoidal (梯形)
  // 在 solve.h 中修改 IRK2Solver 的 Method 枚举
  enum class Method {
    Midpoint
  }; // 移除 Trapezoidal，因为它需要双级 IRK，当前实现错误
  // enum class Method { Midpoint, Trapezoidal };

  explicit IRK2Solver(Method method = Method::Midpoint,
                      Scalar newton_tol = 1e-10, int max_iter = 30);

  void solve(const RHSFunc<Scalar> &f, Scalar t0, Scalar t1,
             const State<Scalar> &y0, Scalar h0, std::vector<Scalar> &times,
             std::vector<State<Scalar>> &states) override;

private:
  struct StepResult {
    State<Scalar> y_next;
    Scalar error; // 未使用，保留占位
    Scalar h_next;
  };
  StepResult step(const RHSFunc<Scalar> &f, Scalar t, const State<Scalar> &y,
                  Scalar h);

  Method method_;
  Scalar newton_tol_;
  int max_iter_;
};





template <typename Scalar> class AdaptiveBDF2Solver : public ODESolver<Scalar> {
public:
  explicit AdaptiveBDF2Solver(Scalar newton_tol = 1e-10, int max_iter = 30);

  void solve(const RHSFunc<Scalar> &f, Scalar t0, Scalar t1,
             const State<Scalar> &y0, Scalar h0, std::vector<Scalar> &times,
             std::vector<State<Scalar>> &states) override;

private:
  // 单步 BDF2：已知 y_{n}, y_{n+1} 和步长 h，求 y_{n+2}
  State<Scalar> abdf2_step(const RHSFunc<Scalar> &f, Scalar t_n,
                          const State<Scalar> &y_n, Scalar t_n1,
                          const State<Scalar> &y_n1, Scalar h,
                          const State<Scalar> &y_init = State<Scalar>());

  Scalar newton_tol_;
  int max_iter_;
};















#endif // ODE_SOLVER_H



