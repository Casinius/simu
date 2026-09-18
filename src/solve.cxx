// ODESolver.cpp — 遗留 ODE 库的现代化实现。
// 模板定义集中于本 TU，文件底部显式实例化 double 版本。
#include "solve.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <print>
#include <stdexcept>
// taskflow v4.1: runtime.hpp/async.hpp 持有 executor 的部分 out-of-line 定义，
// umbrella 头不含它们，须显式引入以满足符号。
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/core/async.hpp>
#include <taskflow/core/runtime.hpp>
#include <taskflow/taskflow.hpp>

namespace rk45_const {

// Fehlberg 4(5) Butcher 表：A 为 6x6 下三角（行 0 为空），c 为节点，
// b4/b5 分别为 4 阶与 5 阶解的权重。
inline constexpr std::array<std::array<double, 6>, 6> A = {{
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {1.0 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0, 0.0},
    {1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0, 0.0},
    {439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0, 0.0},
    {-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0, 0.0},
}};
inline constexpr std::array<double, 6> c = {
    {0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0}};
inline constexpr std::array<double, 6> b4 = {
    {25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0.0}};
inline constexpr std::array<double, 6> b5 = {
    {16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0,
     2.0 / 55.0}};

} // namespace rk45_const

namespace {

// taskflow v4 的 for_each_index 挂在 FlowBuilder 上；包一层 Taskflow +
// run().wait() 提供 Executor 级并行 for。
template <class Fn>
void parallel_for(tf::Executor &ex, int first, int last, int step, Fn &&fn) {
  tf::Taskflow taskflow;
  taskflow.for_each_index(first, last, step, std::forward<Fn>(fn));
  ex.run(std::move(taskflow)).wait();
}

// 固定中心差分扰动（原固定步长变体使用 1e-8）。
template <class Scalar>
Scalar fixed_eps(const State<Scalar> &, int, Scalar) {
  return Scalar(1e-8);
}

// 统一 Newton 迭代：中心差分数值雅可比 + PartialPivLU。
//
//   Residual: State<Scalar>(const State<Scalar>&)          残差 F(y)
//   Eps:       Scalar(const State<Scalar>&, int, Scalar)   第 j 列扰动
//
// refresh_on_slow=false：每次迭代都重算雅可比（原固定 eps 变体）。
// refresh_on_slow=true ：首次迭代重算，之后仅当残差下降过慢
//                        （res_norm > 0.5*prev_res_norm）时重算——原 dyn
//                        变体的惰性雅可比；prev_res_norm 是本次调用内的
//                        局部状态（原成员的跨步残留本就无意义）。
//
// 并行契约：ex 非空且 n >= 64 时雅可比按列并行装配。各列写 J 的不相交列；
// Residual/Eps 闭包必须只读捕获且可并发调用（右端 f 是 const 可调用对象，
// 其内部不得携带跨线程可变状态），否则调用方传 ex=nullptr。
template <class Scalar, class Residual, class Eps>
std::optional<State<Scalar>> newton_solve(const State<Scalar> &y0,
                                          int max_iter, Scalar tol,
                                          Residual &&R, Eps &&eps,
                                          bool refresh_on_slow,
                                          tf::Executor *ex) {
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  const int n = y0.size();
  State<Scalar> y = y0;
  Eigen::PartialPivLU<Mat> lu;
  bool jac_uptodate = false;
  Scalar prev_res_norm = 0;
  for (int iter = 0; iter < max_iter; ++iter) {
    State<Scalar> F = R(y);
    Scalar res_norm = F.norm();
    if (res_norm < tol)
      return y;
    if (!refresh_on_slow || !jac_uptodate) {
      Mat J(n, n);
      auto assemble_column = [&](int j) {
        Scalar e = eps(y, j, res_norm);
        State<Scalar> yp = y, ym = y;
        yp(j) += e;
        ym(j) -= e;
        J.col(j) = (R(yp) - R(ym)) / (Scalar(2) * e);
      };
      if (ex && n >= 64)
        parallel_for(*ex, 0, n, 1, assemble_column);
      else
        for (int j = 0; j < n; ++j)
          assemble_column(j);
      lu.compute(J);
      jac_uptodate = true;
    }
    State<Scalar> delta = lu.solve(-F);
    if (delta.array().isNaN().any())
      return std::nullopt; // 雅可比奇异
    y += delta;
    if (refresh_on_slow && iter > 0 && res_norm > Scalar(0.5) * prev_res_norm)
      jac_uptodate = false; // 收敛过慢，下一轮重算雅可比
    prev_res_norm = res_norm;
  }
  return std::nullopt; // 迭代耗尽未收敛
}

// 隐式 Euler 单步 Newton：F(y) = y - y_curr - h*f(t+h, y)。
template <class Scalar, class Eps>
std::optional<State<Scalar>> implicit_euler_newton(const RHSFunc<Scalar> &f,
                                                   Scalar t, Scalar h,
                                                   const State<Scalar> &y_curr,
                                                   Scalar tol, int max_iter,
                                                   Eps &&eps,
                                                   bool refresh_on_slow,
                                                   tf::Executor *ex) {
  return newton_solve<Scalar>(
      y_curr, max_iter, tol,
      // 显式物化为 State：lambda 若返回 Eigen 表达式模板，其内部引用会指向
      // 本 lambda 帧的临时对象，跨帧求值即悬挂（ASan stack-use-after-return）。
      [&](const State<Scalar> &y) -> State<Scalar> {
        return y - y_curr - h * f(t + h, y);
      },
      std::forward<Eps>(eps), refresh_on_slow, ex);
}

} // namespace

// ======================================================================
// RK45 实现
// ======================================================================
template <class Scalar>
RK45Solver<Scalar>::RK45Solver(Scalar atol, Scalar rtol, Scalar h_min,
                               Scalar h_max, Scalar safety, Scalar fac_min,
                               Scalar fac_max)
    : atol_(atol), rtol_(rtol), h_min_(h_min), h_max_(h_max), safety_(safety),
      fac_min_(fac_min), fac_max_(fac_max) {}

template <class Scalar>
void RK45Solver<Scalar>::solve(const RHSFunc<Scalar> &f, Scalar t0, Scalar t1,
                               const State<Scalar> &y0, Scalar h0,
                               std::vector<Scalar> &times,
                               std::vector<State<Scalar>> &states) {
  times.clear();
  states.clear();

  Scalar t = t0;
  State<Scalar> y = y0;
  Scalar h = std::min(h0, h_max_);
  times.push_back(t);
  states.push_back(y);

  while (t < t1) {
    // 保证最后一步不越过终点
    if (t + h > t1)
      h = t1 - t;
    if (h < h_min_) {
      std::print(stderr, "RK45: step size below minimum! t = {}, h = {}\n", t,
                 h);
      break;
    }

    auto [y_next, error, h_next] = step(f, t, y, h);

    // 误差评估（混合相对/绝对误差）
    Scalar scale = atol_ + rtol_ * y.cwiseAbs().maxCoeff();
    Scalar err_norm = error / scale;

    if (err_norm <= 1.0) { // 接受该步
      t += h;
      y = std::move(y_next);
      times.push_back(t);
      states.push_back(y);
      // 调整下一步长
      if (t >= t1)
        break;
      Scalar fac = safety_ * std::pow(err_norm, -0.2);
      fac = std::clamp(fac, fac_min_, fac_max_);
      h = std::min(h * fac, h_max_);
    } else { // 拒绝，缩小步长重试
      Scalar fac = safety_ * std::pow(err_norm, -0.25);
      fac = std::clamp(fac, fac_min_, 0.9);
      h = std::max(h * fac, h_min_);
    }
  }
}

template <class Scalar>
typename RK45Solver<Scalar>::StepResult
RK45Solver<Scalar>::step(const RHSFunc<Scalar> &f, Scalar t,
                         const State<Scalar> &y, Scalar h) {
  using rk45_const::A;
  using rk45_const::b4;
  using rk45_const::b5;
  using rk45_const::c;

  // stage 循环：ks[i] = f(t + c[i]*h, y + h*Σ_{j<i} A[i][j]*ks[j])。
  // 内积按 j 升序左折叠，与原手写 k1..k6 展开的浮点结合顺序逐项一致。
  std::array<State<Scalar>, 6> ks;
  ks[0] = f(t, y);
  for (int i = 1; i < 6; ++i) {
    State<Scalar> acc = Scalar(A[i][0]) * ks[0];
    for (int j = 1; j < i; ++j)
      acc += Scalar(A[i][j]) * ks[j];
    ks[i] = f(t + Scalar(c[i]) * h, y + h * acc);
  }

  // y4/y5 = y + h*(b·ks)，求和顺序同理
  State<Scalar> s4 = Scalar(b4[0]) * ks[0];
  State<Scalar> s5 = Scalar(b5[0]) * ks[0];
  for (int j = 1; j < 6; ++j) {
    s4 += Scalar(b4[j]) * ks[j];
    s5 += Scalar(b5[j]) * ks[j];
  }
  State<Scalar> y4 = y + h * s4;     // 4阶近似
  State<Scalar> y5 = y + h * s5;     // 5阶近似
  State<Scalar> error_vec = y5 - y4; // 分量误差

  // 计算混合误差范数（分量相对/绝对）
  Scalar err_norm = 0.0;
  for (int i = 0; i < y.size(); ++i) {
    Scalar scale = atol_ + rtol_ * std::abs(y(i));
    Scalar e = std::abs(error_vec(i)) / scale;
    err_norm = std::max(err_norm, e); // 无穷范数，也可使用均方根
  }
  return {std::move(y5), err_norm, h};
}

// ======================================================================
// 标准固定步长 BDF2 实现
// ======================================================================
template <class Scalar>
BDF2Solver<Scalar>::BDF2Solver(Scalar newton_tol, int max_iter)
    : newton_tol_(newton_tol), max_iter_(max_iter) {}

template <class Scalar>
void BDF2Solver<Scalar>::solve(const RHSFunc<Scalar> &f, Scalar t0, Scalar t1,
                               const State<Scalar> &y0, Scalar h0,
                               std::vector<Scalar> &times,
                               std::vector<State<Scalar>> &states) {
  times.clear();
  states.clear();

  // 第一步：使用隐式欧拉生成 y1（固定步长 h）
  Scalar h = std::min(h0, t1 - t0);
  if (h <= 0)
    return;

  Scalar t = t0;
  State<Scalar> y = y0;
  times.push_back(t);
  states.push_back(y);

  auto y1 = implicit_euler_newton(f, t, h, y0, newton_tol_, max_iter_,
                                  fixed_eps<Scalar>, /*refresh_on_slow=*/false,
                                  this->executor_);
  if (!y1.has_value()) {
    std::print(stderr,
               "BDF2: implicit Euler startup did not converge at t={}\n",
               t + h);
    return;
  }
  Scalar t1_bdf = t + h;
  times.push_back(t1_bdf);
  State<Scalar> y_n1 = std::move(*y1);
  states.push_back(y_n1);

  // 后续 BDF2 步进
  State<Scalar> y_n = std::move(y);
  Scalar t_n = t, t_n1 = t1_bdf;

  while (t_n1 < t1 - 1e-12) {
    Scalar t_n2 = t_n1 + h;
    if (t_n2 > t1) {
      // 最后一段，改用隐式欧拉完成
      h = t1 - t_n1;
      t_n2 = t1;
      auto y_n2 = implicit_euler_newton(f, t_n1, h, y_n1, newton_tol_,
                                        max_iter_, fixed_eps<Scalar>,
                                        /*refresh_on_slow=*/false, this->executor_);
      if (!y_n2.has_value()) {
        std::print(stderr,
                   "BDF2: implicit Euler tail did not converge at t={}\n",
                   t_n2);
        break;
      }
      times.push_back(t_n2);
      states.push_back(std::move(*y_n2));
      break;
    } else {
      // 标准 BDF2 步
      auto y_n2 =
          bdf2_step(f, t_n, y_n, t_n1, y_n1, h, /*refresh_on_slow=*/true);
      if (!y_n2.has_value())
        break;
      times.push_back(t_n2);
      y_n = std::move(y_n1);
      y_n1 = std::move(*y_n2);
      states.push_back(y_n1);
      t_n = t_n1;
      t_n1 = t_n2;
    }
  }
}

// 单步 BDF2 Newton：F(y) = y - (4/3 y_{n+1} - 1/3 y_n + 2/3 h f(t_{n+2}, y))。
template <class Scalar>
std::optional<State<Scalar>>
BDF2Solver<Scalar>::bdf2_step(const RHSFunc<Scalar> &f, Scalar /*t_n*/,
                              const State<Scalar> &y_n, Scalar t_n1,
                              const State<Scalar> &y_n1, Scalar h,
                              bool refresh_on_slow) {
  const Scalar t_n2 = t_n1 + h;
  auto result = newton_solve<Scalar>(
      y_n1, max_iter_, newton_tol_,
      [&](const State<Scalar> &y) -> State<Scalar> {
        return y - (4.0 / 3.0 * y_n1 - 1.0 / 3.0 * y_n +
                    (2.0 / 3.0) * h * f(t_n2, y));
      },
      [this, refresh_on_slow](const State<Scalar> &yy, int j, Scalar rn) {
        return refresh_on_slow ? this->eps_policy_(yy, j, rn) : Scalar(1e-8);
      },
      refresh_on_slow, this->executor_);
  if (!result.has_value())
    std::print(stderr, "BDF2: Newton did not converge at t={}\n", t_n2);
  return result;
}

// ======================================================================
// IRK2 实现 (二阶隐式 Runge‑Kutta，隐式中点)
// ======================================================================
template <typename Scalar>
IRK2Solver<Scalar>::IRK2Solver(Scalar newton_tol, int max_iter)
    : newton_tol_(newton_tol), max_iter_(max_iter) {}

template <typename Scalar>
void IRK2Solver<Scalar>::solve(const RHSFunc<Scalar> &f, Scalar t0, Scalar t1,
                               const State<Scalar> &y0, Scalar h0,
                               std::vector<Scalar> &times,
                               std::vector<State<Scalar>> &states) {
  times.clear();
  states.clear();
  Scalar t = t0;
  State<Scalar> y = y0;
  Scalar h = h0;
  times.push_back(t);
  states.push_back(y);

  while (t < t1 - 1e-12) {
    if (t + h > t1)
      h = t1 - t;

    auto y_next = step(f, t, y, h);
    if (!y_next.has_value()) {
      // 失败：尝试减半步长
      h *= 0.5;
      if (h < 1e-15) {
        std::print(stderr, "IRK2Solver: step size too small, aborting at t={}\n",
                   t);
        break;
      }
      std::print(stderr, "IRK2Solver: step failed, retrying with h={}\n", h);
      continue;
    }

    t += h;
    y = std::move(*y_next);
    times.push_back(t);
    states.push_back(y);
  }
}

// 单步隐式中点：F(y_next) = y_next - (y + h f(t + h/2, (y + y_next)/2))。
template <typename Scalar>
std::optional<State<Scalar>>
IRK2Solver<Scalar>::step(const RHSFunc<Scalar> &f, Scalar t,
                         const State<Scalar> &y, Scalar h) {
  if (y.size() == 0 || !f)
    return std::nullopt;

  const Scalar t_mid = t + 0.5 * h;

  State<Scalar> y_next = y + h * f(t, y); // 显式 Euler 预测
  auto result = newton_solve<Scalar>(
      y_next, max_iter_, newton_tol_,
      [&](const State<Scalar> &yn) -> State<Scalar> {
        State<Scalar> y_mid = Scalar(0.5) * (y + yn);
        return yn - (y + h * f(t_mid, y_mid));
      },
      [this](const State<Scalar> &yy, int j, Scalar rn) {
        return this->eps_policy_(yy, j, rn);
      },
      /*refresh_on_slow=*/false, this->executor_);
  if (!result.has_value())
    std::print(stderr, "IRK2: Newton did not converge at t={}\n", t);
  return result;
}

// ======================================================================
// 半隐式 Euler（隐式 Euler）实现
// ======================================================================
template <class Scalar>
SemiImplicitEulerSolver<Scalar>::SemiImplicitEulerSolver(Scalar newton_tol,
                                                         int max_iter)
    : newton_tol_(newton_tol), max_iter_(max_iter) {}

template <class Scalar>
void SemiImplicitEulerSolver<Scalar>::solve(
    const RHSFunc<Scalar> &f, Scalar t0, Scalar t1, const State<Scalar> &y0,
    Scalar h0, std::vector<Scalar> &times, std::vector<State<Scalar>> &states) {
  times.clear();
  states.clear();

  Scalar t = t0;
  State<Scalar> y = y0;
  Scalar h = h0; // 固定步长
  times.push_back(t);
  states.push_back(y);

  while (t < t1) {
    if (t + h > t1)
      h = t1 - t;
    auto y_next = implicit_euler_step(f, t, h, y, /*refresh_on_slow=*/true);
    if (!y_next.has_value()) {
      h *= 0.5;        // 减半步长
      if (h < 1e-12) { // 防止死循环
        std::print(stderr, "Step size too small, aborting.\n");
        break;
      }
      continue; // 重新尝试当前步
    }
    t += h;
    y = std::move(*y_next);
    times.push_back(t);
    states.push_back(y);
  }
}

template <class Scalar>
std::optional<State<Scalar>>
SemiImplicitEulerSolver<Scalar>::implicit_euler_step(
    const RHSFunc<Scalar> &f, Scalar t, Scalar h, const State<Scalar> &y_curr,
    bool refresh_on_slow) {
  auto result = implicit_euler_newton(
      f, t, h, y_curr, newton_tol_, max_iter_,
      [this, refresh_on_slow](const State<Scalar> &yy, int j, Scalar rn) {
        return refresh_on_slow ? this->eps_policy_(yy, j, rn) : Scalar(1e-8);
      },
      refresh_on_slow, this->executor_);
  if (!result.has_value())
    std::print(
        stderr,
        "SemiImplicitEuler: Newton iteration did not converge at t = {}\n",
        t + h);
  return result;
}

// ======================================================================
// Verlet 实现
// ======================================================================
template <class Scalar>
void VerletSolver<Scalar>::solve(const RHSFunc<Scalar> &f, Scalar t0, Scalar t1,
                                 const State<Scalar> &y0, Scalar h0,
                                 std::vector<Scalar> &times,
                                 std::vector<State<Scalar>> &states) {
  times.clear();
  states.clear();

  Scalar t = t0;
  State<Scalar> y = y0;
  Scalar h = h0; // 固定步长
  times.push_back(t);
  states.push_back(y);

  while (t < t1) {
    if (t + h > t1)
      h = t1 - t;
    State<Scalar> y_next = verlet_step(f, t, h, y);
    t += h;
    y = std::move(y_next);
    times.push_back(t);
    states.push_back(y);
  }
}

template <class Scalar>
State<Scalar> VerletSolver<Scalar>::verlet_step(const RHSFunc<Scalar> &f,
                                                Scalar t, Scalar h,
                                                const State<Scalar> &y_curr) {
  if (y_curr.size() % 2 != 0) {
    throw std::invalid_argument("State size must be even for Verlet");
  }
  // y_curr 应包含 [位置, 速度]：前半部分为位置，后半部分为速度
  int n = y_curr.size() / 2;

  State<Scalar> y_new(y_curr.size());

  // 1. 拆解状态
  State<Scalar> x = y_curr.head(n); // 当前位置
  State<Scalar> v = y_curr.tail(n); // 当前速度

  // 2. 计算当前加速度 a(t) = f(t, y_curr) 的后半部分（速度导数）
  State<Scalar> f_curr = f(t, y_curr);
  State<Scalar> a_curr = f_curr.tail(n); // 加速度

  // 3. 更新位置（显式，使用当前加速度）
  State<Scalar> x_new = x + v * h + 0.5 * a_curr * h * h;

  // 4. 构造新状态用于计算新加速度（位置已更新，速度使用半步更新）
  State<Scalar> y_mid(y_curr.size());
  y_mid.head(n) = x_new;

  State<Scalar> v_half = v + 0.5 * a_curr * h;

  y_mid.tail(n) = v_half;

  State<Scalar> f_new = f(t + h, y_mid);
  State<Scalar> a_new = f_new.tail(n);

  // 5. 更新速度（使用基于半步更新的速度计算平均加速度）
  State<Scalar> v_new = v_half + 0.5 * a_new * h;

  // 6. 组装新状态
  y_new.head(n) = x_new;
  y_new.tail(n) = v_new;

  return y_new;
}

// ======================================================================
// 显式实例化（自原 unity-hack 配置头迁移，修复其死链接）
// ======================================================================
using ODEScalar = double;
template class RK45Solver<ODEScalar>;
template class BDF2Solver<ODEScalar>;
template class IRK2Solver<ODEScalar>;
template class SemiImplicitEulerSolver<ODEScalar>;
template class VerletSolver<ODEScalar>;
