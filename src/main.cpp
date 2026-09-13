#include <print>
#include <vector>

#include "fpxpbd.h"

using Real = double;
using Unit = xpbd::FP_XPBD::TetrahedronFEMData<Real>;
using Traits = xpbd::FP_XPBD::GeomTraits<Real, Unit>;

static Real total_energy(const std::vector<Unit> &units,
                         const std::vector<Eigen::Vector3<Real>> &pos) {
  Real U = 0;
  for (const auto &unit : units)
    for (size_t c = 0; c < unit.indices.size() / 4; ++c) {
      std::array<Eigen::Matrix<Real, 3, 1>, 4> p;
      for (int i = 0; i < 4; ++i)
        p[i] = pos[unit.indices[c * 4 + i]];
      auto F = Traits::computeF(p, unit.Dm_inv[c]);
      // C = F^T F
      Eigen::Matrix<Real, 3, 3> C = F.transpose() * F;
      U += xpbd::FP_XPBD::StableNeoHookean<Real, 3>::psi(C, unit.mu[c], unit.lambda_mat[c]) *
           unit.rest_volume[c];
    }
  return U;
}

static Real min_detF(const std::vector<Unit> &units,
                     const std::vector<Eigen::Vector3<Real>> &pos) {
  Real m = 1e300;
  for (const auto &unit : units)
    for (size_t c = 0; c < unit.indices.size() / 4; ++c) {
      std::array<Eigen::Matrix<Real, 3, 1>, 4> p;
      for (int i = 0; i < 4; ++i)
        p[i] = pos[unit.indices[c * 4 + i]];
      m = std::min(m, Traits::computeF(p, unit.Dm_inv[c]).determinant());
    }
  return m;
}

// ===== 场景：三角棱柱梁沿 x 轴 K 段，每段 prism 拆 3 个四面体 =====
constexpr int K = 4; // 段数
constexpr size_t n_verts = 3 * (K + 1);
constexpr size_t n_cells = 3 * K;

// 静止位置：横截面三角形 (i,0,0), (i,1,0), (i,0,1)，顶点序 = 3*i + j
static std::vector<Eigen::Vector3<Real>> rest_positions() {
  std::vector<Eigen::Vector3<Real>> rest(n_verts);
  for (int i = 0; i <= K; ++i) {
    rest[3 * i + 0] = {Real(i), 0, 0};
    rest[3 * i + 1] = {Real(i), 1, 0};
    rest[3 * i + 2] = {Real(i), 0, 1};
  }
  return rest;
}

// 每段 3 个四面体：v0..v2 = 截面 i，v3..v5 = 截面 i+1
static std::vector<std::array<size_t, 4>> cell_list() {
  constexpr int prism_tets[3][4] = {{0, 1, 2, 4}, {0, 2, 4, 5}, {0, 4, 5, 3}};
  std::vector<std::array<size_t, 4>> cells;
  cells.reserve(n_cells);
  for (int seg = 0; seg < K; ++seg)
    for (const auto &t : prism_tets) {
      int b = 3 * seg, n = 3 * (seg + 1);
      auto vidx = [&](int c) { return size_t(c < 3 ? b + c : n + (c - 3)); };
      cells.push_back({vidx(t[0]), vidx(t[1]), vidx(t[2]), vidx(t[3])});
    }
  return cells;
}

// 每个 cell 的材料：按段变化 E_u = 1e4*(1+0.5*u)，nu = 0.3
static void cell_material(std::vector<Real> &mu, std::vector<Real> &lam) {
  const Real nu = 0.3;
  mu.resize(n_cells);
  lam.resize(n_cells);
  for (int seg = 0; seg < K; ++seg) {
    const Real E = 1e4 * (1 + 0.5 * seg);
    for (int j = 0; j < 3; ++j) {
      mu[3 * seg + j] = E / (2 * (1 + nu));
      lam[3 * seg + j] = E * nu / ((1 + nu) * (1 - 2 * nu));
    }
  }
}

struct SimStats {
  Real E_final_avg = 0;   // 末窗口平均弹性能
  Real E_max = 0;
  Real detF_min = 1e300;  // 全程每单元 detF 最小值
  bool finite = true;
};

// 把 cells 按 unit_of_cell[c] 分组成多个单元体并仿真；同物理、不同拆分。
static SimStats run_sim(const std::vector<int> &unit_of_cell, int n_units,
                        const std::vector<Eigen::Vector3<Real>> &rest) {
  const auto cells = cell_list();
  std::vector<Real> mu, lam;
  cell_material(mu, lam);

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
      units, /*maxIter=*/10, /*over_relax=*/1.0, /*gravity=*/{0, -9.8, 0});

  SimStats st;
  Real window_sum = 0;
  for (int step = 1; step <= 3000; ++step) {
    solver.solve(data, dt);
    const Real E = total_energy(units, data.pos);
    const Real detF = min_detF(units, data.pos);
    st.E_max = std::max(st.E_max, E);
    st.detF_min = std::min(st.detF_min, detF);
    st.finite = st.finite && std::isfinite(E) && std::isfinite(detF);
    for (const auto &p : data.pos)
      st.finite = st.finite && p.allFinite();
    if (step > 2900)
      window_sum += E;
    if (step % 1500 == 0) {
      Real vmax = 0;
      for (size_t v = 0; v < n_verts; ++v)
        vmax = std::max(vmax, data.inv_mass[v] > 0 ? data.velocity[v].norm() : Real(0));
      std::print("  step {:4d}  E = {:12.6e}  min det F = {:10.6f}  max|v| = {:10.6f}\n",
                 step, E, detF, vmax);
    }
  }
  st.E_final_avg = window_sum / 100;
  return st;
}

int main() {
  // 多单元体测试：梁拆成 K 个单元体（相邻单元体共享横截面顶点，
  // Gauss-Seidel 逐单元迭代经共享顶点耦合），每单元体刚度不同；
  // 对照组：同一 mesh 全部 cells 归入 1 个单元体。
  // 判据：拆分不影响物理——两组末窗口平均能量一致。
  const auto rest = rest_positions();

  std::vector<int> multi(n_cells), single(n_cells, 0);
  for (size_t c = 0; c < n_cells; ++c)
    multi[c] = int(c / 3);

  std::print("multi-unit ({} units, shared section verts):\n", K);
  const SimStats s_multi = run_sim(multi, K, rest);
  std::print("single-unit (1 unit, same cells/materials):\n");
  const SimStats s_single = run_sim(single, 1, rest);

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
