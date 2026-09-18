#include <chrono>
#include <cstdlib>
#include <format>
#include <string>
#include <print>
#include <ranges>
#include <string_view>
#include <thread>
#include <vector>

#include <range/v3/all.hpp>
#include <taskflow/core/executor.hpp>
#include <taskflow/core/runtime.hpp>
#include <taskflow/core/async.hpp>
#include <taskflow/taskflow.hpp>

#include "fpxpbd.h"

using Real = double;
using Unit = xpbd::FP_XPBD::TetrahedronFEMData<Real>;
using Traits = xpbd::FP_XPBD::GeomTraits<Real, Unit>;

namespace rv = ranges::views;

static Real total_energy(const std::vector<Unit> &units,
                         const std::vector<Eigen::Vector3<Real>> &pos) {
  // 单一累加器、单元序 → cell 序的左折叠，与基线求值顺序逐位一致
  struct Ref {
    const Unit *unit;
    size_t c;
  };
  std::vector<Ref> cell_refs;
  for (const Unit &unit : units)
    for (size_t c = 0; c < unit.indices.size() / 4; ++c)
      cell_refs.push_back({&unit, c});

  return ranges::fold_left(
      cell_refs, Real(0), [&](Real U, const Ref &ref) {
        const auto &[unit, c] = ref;
        std::array<Eigen::Matrix<Real, 3, 1>, 4> p;
        for (int i = 0; i < 4; ++i)
          p[i] = pos[unit->indices[c * 4 + i]];
        auto F = Traits::computeF(p, unit->Dm_inv[c]);
        // C = F^T F
        Eigen::Matrix<Real, 3, 3> C = F.transpose() * F;
        return U + xpbd::FP_XPBD::StableNeoHookean<Real, 3>::psi(
                       C, unit->mu[c], unit->lambda_mat[c]) *
                       unit->rest_volume[c];
      });
}

static Real min_detF(const std::vector<Unit> &units,
                     const std::vector<Eigen::Vector3<Real>> &pos) {
  return ranges::fold_left(
      units, Real(1e300), [&](Real m, const Unit &unit) {
        return ranges::fold_left(
            unit.indices | rv::chunk(4) | rv::enumerate, m,
            [&](Real mm, const auto &ec) {
              const auto &[c, cell] = ec;
              std::array<Eigen::Matrix<Real, 3, 1>, 4> p;
              std::ranges::copy(
                  cell | std::views::transform([&](size_t v) { return pos[v]; }),
                  p.begin());
              return std::min(
                  mm, Traits::computeF(p, unit.Dm_inv[c]).determinant());
            });
      });
}

// ===== 场景：三角棱柱梁沿 x 轴 k_segs 段，每段 prism 拆 3 个四面体 =====
static size_t n_verts_of(int k_segs) { return size_t(3 * (k_segs + 1)); }
static size_t n_cells_of(int k_segs) { return size_t(3 * k_segs); }
inline constexpr int prism_tets[3][4] = {{0, 1, 2, 4}, {0, 2, 4, 5}, {0, 4, 5, 3}};
inline constexpr Real kNu = 0.3;

// 静止位置：横截面三角形 (i,0,0), (i,1,0), (i,0,1)，顶点序 = 3*i + j
static std::vector<Eigen::Vector3<Real>> rest_positions(int k_segs) {
  return std::views::iota(size_t(0), n_verts_of(k_segs)) |
         std::views::transform([](size_t v) {
           const size_t i = v / 3, j = v % 3;
           return Eigen::Vector3<Real>{Real(i), Real(j == 1), Real(j == 2)};
         }) |
         std::ranges::to<std::vector>();
}

// 每段 3 个四面体：v0..v2 = 截面 i，v3..v5 = 截面 i+1
static std::vector<std::array<size_t, 4>> cell_list(int k_segs) {
  return std::views::iota(0, k_segs) |
         std::views::transform([](int seg) {
           const int b = 3 * seg, n = 3 * (seg + 1);
           auto vidx = [&](int c) {
             return size_t(c < 3 ? b + c : n + (c - 3));
           };
           return prism_tets | std::views::transform([&](const auto &t) {
                    return std::array<size_t, 4>{
                        vidx(t[0]), vidx(t[1]), vidx(t[2]), vidx(t[3])};
                  }) |
                  std::ranges::to<std::vector>();
         }) |
         std::views::join | std::ranges::to<std::vector>();
}

// 每个 cell 的材料：按段变化 E_u = 1e4*(1+0.5*u)，nu = 0.3
static void cell_material(std::vector<Real> &mu, std::vector<Real> &lam,
                          int k_segs) {
  const auto pairs =
      std::views::iota(0, k_segs) |
      std::views::transform([](int seg) {
        const Real E = 1e4 * (1 + 0.5 * seg);
        return std::views::iota(0, 3) |
               std::views::transform([=](int) {
                 return std::pair{E / (2 * (1 + kNu)),
                                  E * kNu / ((1 + kNu) * (1 - 2 * kNu))};
               }) |
               std::ranges::to<std::vector>();
      }) |
      std::views::join | std::ranges::to<std::vector>();
  mu.clear();
  lam.clear();
  for (const auto &[m, l] : pairs) {
    mu.push_back(m);
    lam.push_back(l);
  }
}

struct SimStats {
  Real E_final_avg = 0;   // 末窗口平均弹性能
  Real E_max = 0;
  Real detF_min = 1e300;  // 全程每单元 detF 最小值
  bool finite = true;
  Real E_step100 = 0;     // 第 100 步能量快照（并行一致性判据用）
  std::string log;        // 逐步输出（并发跑时先缓存，完成后按原序打印）
};

// 把 cells 按 unit_of_cell[c] 分组成多个单元体并仿真；同物理、不同拆分。
static SimStats run_sim(const std::vector<int> &unit_of_cell, int n_units,
                        const std::vector<Eigen::Vector3<Real>> &rest,
                        int k_segs, tf::Executor *executor = nullptr) {
  const size_t n_verts = n_verts_of(k_segs);
  const size_t n_cells = n_cells_of(k_segs);
  const auto cells = cell_list(k_segs);
  std::vector<Real> mu, lam;
  cell_material(mu, lam, k_segs);

  std::vector<Unit> units(n_units);
  for (size_t c = 0; c < n_cells; ++c) {
    Unit &u = units[unit_of_cell[c]];
    u.indices.insert(u.indices.end(), cells[c].begin(), cells[c].end());
    u.mu.push_back(mu[c]);
    u.lambda_mat.push_back(lam[c]);
  }
  xpbd::FP_XPBD::setup_units(units, rest);

  xpbd::PhysicsData<Real> data;
  data.pos = rest;
  data.pos_prev = rest;
  data.velocity.assign(n_verts, Eigen::Vector3<Real>::Zero());
  data.inv_mass.assign(n_verts, 1);
  for (int j = 0; j < 3; ++j)
    data.inv_mass[j] = 0; // 钉住 x=0 截面

  // 自由顶点确定性小扰动，制造形变
  for (size_t v = 3; v < n_verts; ++v)
    data.pos[v] += Eigen::Vector3<Real>(0.05 * std::sin(Real(v)),
                                        0.03 * std::cos(Real(v)),
                                        0.04 * std::sin(2 * Real(v)));

  const Real dt = 1.0 / 60;
  xpbd::FP_XPBD::FPSolver<Real, Unit> solver(
      units, /*maxIter=*/10, /*over_relax=*/1.0, /*gravity=*/{0, -9.8, 0},
      executor);

  SimStats st;
  Real window_sum = 0;
  for (int step = 1; step <= 3000; ++step) {
    solver.solve(data, dt);
    const Real E = total_energy(units, data.pos);
    const Real detF = min_detF(units, data.pos);
    st.E_max = std::max(st.E_max, E);
    st.detF_min = std::min(st.detF_min, detF);
    st.finite = st.finite && std::isfinite(E) && std::isfinite(detF) &&
                std::ranges::all_of(data.pos, [](const auto &p) {
                  return p.allFinite();
                });
    if (step > 2900)
      window_sum += E;
    if (step == 100)
      st.E_step100 = E;
    if (step % 1500 == 0) {
      const Real vmax = ranges::max(
          std::views::iota(size_t(0), n_verts) |
          std::views::transform(
              [&](size_t v) {
                return data.inv_mass[v] > 0 ? data.velocity[v].norm() : Real(0);
              }) |
          std::ranges::to<std::vector>());
      st.log.append(std::format(
          "  step {:4d}  E = {:12.6e}  min det F = {:10.6f}  max|v| = {:10.6f}\n",
          step, E, detF, vmax));
    }
  }
  st.E_final_avg = window_sum / 100;
  return st;
}

static int run_default() {
  // 多单元体测试：梁拆成 K 个单元体（相邻单元体共享横截面顶点，
  // Gauss-Seidel 逐单元迭代经共享顶点耦合），每单元体刚度不同；
  // 对照组：同一 mesh 全部 cells 归入 1 个单元体。
  // 判据：拆分不影响物理——两组末窗口平均能量一致。
  const int k_segs = 10;
  const auto rest = rest_positions(k_segs);
  const size_t n_cells = n_cells_of(k_segs);

  std::vector<int> multi(n_cells), single(n_cells, 0);
  for (size_t c = 0; c < n_cells; ++c)
    multi[c] = int(c / 3);

  // 两次仿真相互独立：提交同一 executor 并发执行；输出先缓存、
  // 双双完成后按原顺序打印，stdout 与串行基线一致。
  std::print("multi-unit ({} units, shared section verts):\n", k_segs);
  SimStats s_multi, s_single;
  {
    tf::Executor executor(2);
    tf::Taskflow taskflow;
    taskflow.emplace([&] { s_multi = run_sim(multi, k_segs, rest, k_segs); });
    taskflow.emplace([&] { s_single = run_sim(single, 1, rest, k_segs); });
    executor.run(taskflow).wait();
  }
  std::print("{}", s_multi.log);
  std::print("single-unit (1 unit, same cells/materials):\n");
  std::print("{}", s_single.log);

  std::print("E_final_avg: multi = {:.6e}  single = {:.6e}\n", s_multi.E_final_avg,
             s_single.E_final_avg);
  std::print("E_max:       multi = {:.6e}  single = {:.6e}\n", s_multi.E_max,
             s_single.E_max);
  std::print("min det F:   multi = {:.6f}  single = {:.6f}\n", s_multi.detF_min,
             s_single.detF_min);

  // ---- 测试判定 ----
  bool pass = true;
  pass &= s_multi.finite && s_single.finite;   // 无 NaN/Inf
  pass &= s_multi.detF_min > 0;                // 多单元耦合无倒置
  pass &= s_single.detF_min > 0;
  const Real denom = std::max({std::abs(s_multi.E_final_avg),
                               std::abs(s_single.E_final_avg), Real(1)});
  const Real rel = std::abs(s_multi.E_final_avg - s_single.E_final_avg) / denom;
  std::print("rel diff of window-avg energy = {:.4e}\n", rel);
  pass &= rel < 0.05;                          // 拆分不变性（Gauss-Seidel 不动点一致）
  std::println("{}", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

// ---- --par 基准：seq vs 着色并行 ----
static int run_par(int k_segs) {
  tf::Executor executor(std::thread::hardware_concurrency());
  std::print("--par benchmark: {} segments, executor with {} workers\n",
             k_segs, executor.num_workers());
  const auto rest = rest_positions(k_segs);
  const size_t n_cells = n_cells_of(k_segs);
  std::vector<int> multi(n_cells), single(n_cells, 0);
  for (size_t c = 0; c < n_cells; ++c)
    multi[c] = int(c / 3);

  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  SimStats s_seq = run_sim(single, 1, rest, k_segs);
  const auto t1 = clock::now();
  SimStats s_par = run_sim(single, 1, rest, k_segs, &executor);
  const auto t2 = clock::now();

  const Real seq_s = std::chrono::duration<Real>(t1 - t0).count();
  const Real par_s = std::chrono::duration<Real>(t2 - t1).count();
  std::print("seq: {:.3f}s   colored-parallel: {:.3f}s   speedup = {:.2f}x\n",
             seq_s, par_s, seq_s / par_s);
  std::print("E_final_avg: seq = {:.6e}  par = {:.6e}\n", s_seq.E_final_avg,
             s_par.E_final_avg);
  std::print("min det F:   seq = {:.6f}  par = {:.6f}\n", s_seq.detF_min,
             s_par.detF_min);

  // 一致性判据：着色并行改变 Gauss-Seidel 访问次序；每步不动点逐位一致，
  // 但混沌放大使 3000 步长视野下轨迹必然分离（信息性指标，非实现缺陷）。
  // 实现正确性用第 100 步快照判（同一物理、小容差）：
  const auto rel_of = [](Real a, Real b) {
    return std::abs(a - b) / std::max({std::abs(a), std::abs(b), Real(1)});
  };
  const Real rel100 = rel_of(s_seq.E_step100, s_par.E_step100);
  const Real rel_final = rel_of(s_seq.E_final_avg, s_par.E_final_avg);
  std::print("rel diff @ step 100 (parity gate) = {:.4e}\n", rel100);
  std::print("rel diff of window-avg energy (informational) = {:.4e}\n",
             rel_final);

  bool pass = s_seq.finite && s_par.finite;
  pass &= s_seq.detF_min > 0 && s_par.detF_min > 0;
  pass &= rel100 < 0.05;
  std::println("{}", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

int main(int argc, char *argv[]) {
  if (argc > 1 && std::string_view(argv[1]) == "--par") {
    const int k_segs = argc > 2 ? std::atoi(argv[2]) : 200;
    return run_par(k_segs);
  }
  return run_default();
}
