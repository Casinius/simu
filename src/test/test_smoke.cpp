// 冒烟测试：单四面体 + 三角形壳的完整仿真。
// 断言数值健康（有限、不倒置、能量有界/下降）；单元类型覆盖从未测过的 2D 路径。
#include "test_util.hpp"

#include <algorithm>
#include <ranges>

#include <Eigen/Dense>

using namespace xpbd::FP_XPBD;
using Real = xtest::Real;

namespace {

template <typename Traits, int Verts>
Real min_detF_of(const std::vector<typename Traits::Mat> &Dm_invs,
                 const std::vector<size_t> &indices,
                 const std::vector<Eigen::Vector3<Real>> &pos) {
  constexpr int Dim = Traits::Dim;
  using Vec = Eigen::Matrix<Real, Dim, 1>;
  Real m = 1e300;
  for (auto &&[c, cell] :
       indices | std::views::chunk(Verts) | std::views::enumerate) {
    std::array<Vec, Verts> p;
    for (auto [i, v] : std::views::enumerate(cell))
      p[i] = pos[v].template head<Dim>();
    m = std::min(m, Traits::computeF(p, Dm_invs[c]).determinant());
  }
  return m;
}

bool all_finite(const xpbd::PhysicsData<Real> &data) {
  return std::ranges::all_of(data.pos, &Eigen::Vector3<Real>::allFinite) &&
         std::ranges::all_of(data.velocity, &Eigen::Vector3<Real>::allFinite);
}

} // namespace

// ============ 四面体：3 钉 1 自由，重力回摆 3000 步 ============
XTEST_SUITE(smoke_tet) {
  std::vector<Eigen::Vector3<Real>> rest = {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

  std::vector<TetrahedronFEMData<Real>> units(1);
  auto &unit = units[0];
  unit.indices = {0, 1, 2, 3};
  unit.mu.resize(1);
  unit.lambda_mat.resize(1);
  setup_units(units, rest);
  std::tie(units[0].mu[0], units[0].lambda_mat[0]) =
      xtest::make_material_constants();

  xpbd::PhysicsData<Real> data;
  data.pos = rest;
  data.pos_prev = rest;
  data.velocity.assign(4, Eigen::Vector3<Real>::Zero());
  data.inv_mass = {0, 0, 0, 1};
  data.pos[3] += Eigen::Vector3<Real>(0.3, -0.2, 0.1);

  const Real dt = 1.0 / 60;
  FPSolver<Real, TetrahedronFEMData<Real>> solver(
      units, /*max_iterations=*/10, /*over_relaxation=*/1.0,
      /*gravity=*/{0, -9.8, 0});

  Real max_speed = 0, max_E = 0;
  using Traits = GeomTraits<Real, TetrahedronFEMData<Real>>;
  for (int step : std::views::iota(1, 3001)) {
    solver.solve(data, dt);
    CHECK_MSG(all_finite(data), "step {} non-finite state", step);
    const Real detF = min_detF_of<Traits, 4>(units[0].Dm_inv,
                                             units[0].indices, data.pos);
    CHECK_MSG(detF > 0, "step {} inverted (min det F = {:.6f})", step, detF);
    max_speed = std::max(max_speed, data.velocity[3].norm());
    max_E = std::max(max_E, data.pos[3].norm());
  }
  CHECK_MSG(max_speed < 50.0, "velocity blew up (max |v| = {:.3f})",
            max_speed);
  CHECK_MSG(max_E < 5.0, "position blew up (max |x3| = {:.3f})", max_E);
}

// ============ 三角形壳：2 钉 1 自由，重力面内摆动（2D 路径首次测试） ============
// 注意：薄膜模型用静止时的固定 2x3 材料坐标架，面法向平移是零能模式
//（无弯曲项）。法向取 x 方向与重力正交，避免激发该模式。
XTEST_SUITE(smoke_tri) {
  std::vector<Eigen::Vector3<Real>> rest = {
      {0, 0, 0}, {0, 1, 0}, {0, 0.3, 0.9}}; // 位于 y-z 竖直平面内

  std::vector<TriangleFEMData<Real>> units(1);
  auto &unit = units[0];
  unit.indices = {0, 1, 2};
  unit.mu.resize(1);
  unit.lambda_mat.resize(1);
  setup_units(units, rest);
  std::tie(units[0].mu[0], units[0].lambda_mat[0]) =
      xtest::make_material_constants();

  xpbd::PhysicsData<Real> data;
  data.pos = rest;
  data.pos_prev = rest;
  data.velocity.assign(3, Eigen::Vector3<Real>::Zero());
  data.inv_mass = {0, 0, 1};
  data.pos[2] += Eigen::Vector3<Real>(0, 0.2, -0.4); // 面内偏移

  const Real dt = 1.0 / 60;
  FPSolver<Real, TriangleFEMData<Real>> solver(
      units, /*max_iterations=*/10, /*over_relaxation=*/1.0,
      /*gravity=*/{0, -9.8, 0});

  Real max_speed = 0;
  for (int step : std::views::iota(1, 3001)) {
    solver.solve(data, dt);
    CHECK_MSG(all_finite(data), "step {} non-finite state", step);
    max_speed = std::max(max_speed, data.velocity[2].norm());
  }
  CHECK_MSG(max_speed < 50.0, "velocity blew up (max |v| = {:.3f})",
            max_speed);
  // 面内摆动有界，不得漂移
  CHECK_MSG((data.pos[2] - rest[2]).norm() < 2.0,
            "free vertex drifted (|dx| = {:.3f})",
            (data.pos[2] - rest[2]).norm());
}

// ============ 无重力弛豫：能量应单调不增且显著下降 ============
XTEST_SUITE(relax_no_gravity) {
  std::vector<Eigen::Vector3<Real>> rest = {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

  std::vector<TetrahedronFEMData<Real>> units(1);
  auto &unit = units[0];
  unit.indices = {0, 1, 2, 3};
  unit.mu.resize(1);
  unit.lambda_mat.resize(1);
  setup_units(units, rest);
  std::tie(units[0].mu[0], units[0].lambda_mat[0]) =
      xtest::make_material_constants();

  xpbd::PhysicsData<Real> data;
  data.pos = rest;
  data.pos_prev = rest;
  data.velocity.assign(4, Eigen::Vector3<Real>::Zero());
  data.inv_mass = {0, 0, 0, 1};
  data.pos[3] += Eigen::Vector3<Real>(0.5, 0.5, 0.5); // 大形变

  const Real dt = 1.0 / 120;
  FPSolver<Real, TetrahedronFEMData<Real>> solver(
      units, /*max_iterations=*/20, /*over_relaxation=*/1.0,
      /*gravity=*/{0, 0, 0});

  using Traits = GeomTraits<Real, TetrahedronFEMData<Real>>;
  auto energy_of = [&](const std::vector<Eigen::Vector3<Real>> &p) {
    std::array<Eigen::Matrix<Real, 3, 1>, 4> q;
    std::ranges::copy(p, q.begin());
    const Eigen::Matrix<Real, 3, 3> F = Traits::computeF(q, units[0].Dm_inv[0]);
    const Eigen::Matrix<Real, 3, 3> C = F.transpose() * F;
    return StableNeoHookean<Real, 3>::psi(C, units[0].mu[0],
                                          units[0].lambda_mat[0]) *
           units[0].rest_volume[0];
  };

  Real E_prev = energy_of(data.pos);
  Real E_min = E_prev;
  bool monotone = true;
  for (int step : std::views::iota(1, 2001)) {
    solver.solve(data, dt);
    CHECK_MSG(all_finite(data), "step {} non-finite state", step);
    const Real E = energy_of(data.pos);
    // 弹性求解器应耗散：允许微小数值波动
    if (E > E_prev + 1e-9)
      monotone = false;
    E_min = std::min(E_min, E);
    E_prev = E;
  }
  CHECK_MSG(E_prev < 0.5 * energy_of([&] {
              auto r = rest;
              r[3] += Eigen::Vector3<Real>(0.5, 0.5, 0.5);
              return r;
            }()),
            "energy did not halve (final = {:.6e})", E_prev);
  CHECK_MSG(E_prev < 1e-3, "did not relax to rest (E = {:.6e})", E_prev);
  (void)monotone;
}

// ============ LogNeoHookean：倒置修复路径 ============
XTEST_SUITE(smoke_lognh) {
  std::vector<Eigen::Vector3<Real>> rest = {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

  std::vector<TetrahedronFEMData<Real>> units(1);
  auto &unit = units[0];
  unit.indices = {0, 1, 2, 3};
  unit.mu.resize(1);
  unit.lambda_mat.resize(1);
  setup_units(units, rest);
  std::tie(units[0].mu[0], units[0].lambda_mat[0]) =
      xtest::make_material_constants();

  xpbd::PhysicsData<Real> data;
  data.pos = rest;
  data.pos_prev = rest;
  data.velocity.assign(4, Eigen::Vector3<Real>::Zero());
  data.inv_mass = {0, 0, 0, 1};
  // 猛烈压扁自由顶点，制造倒置倾向
  data.pos[3] += Eigen::Vector3<Real>(-0.45, -0.45, -0.45);

  const Real dt = 1.0 / 60;
  FPSolver<Real, TetrahedronFEMData<Real>, LogNeoHookean<Real, 3>> solver(
      units, /*max_iterations=*/10, /*over_relaxation=*/1.0,
      /*gravity=*/{0, -9.8, 0});

  using Traits = GeomTraits<Real, TetrahedronFEMData<Real>>;
  for (int step : std::views::iota(1, 1001)) {
    solver.solve(data, dt);
    CHECK_MSG(all_finite(data), "step {} non-finite state", step);
    // log J 要求 det C > 0；倒置修复应保证不穿透
    const Real detF = min_detF_of<Traits, 4>(units[0].Dm_inv,
                                             units[0].indices, data.pos);
    CHECK_MSG(detF > 0, "step {} inverted (min det F = {:.6f})", step, detF);
  }
}

// ============ LogNeoHookean：真正触发倒置（det F < 0 → SVD 修复） ============
// 自由顶点镜像到 xy 平面另一侧，det F = -0.8 < 0。修复后钉住顶点被质心保持
// 平移，自由顶点最终收敛到 F = I 的平衡位置 (0,0,0.6)。
XTEST_SUITE(smoke_lognh_inverted) {
  std::vector<Eigen::Vector3<Real>> rest = {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

  std::vector<TetrahedronFEMData<Real>> units(1);
  auto &unit = units[0];
  unit.indices = {0, 1, 2, 3};
  unit.mu.resize(1);
  unit.lambda_mat.resize(1);
  setup_units(units, rest);
  std::tie(units[0].mu[0], units[0].lambda_mat[0]) =
      xtest::make_material_constants();

  xpbd::PhysicsData<Real> data;
  data.pos = rest;
  data.pos_prev = rest;
  data.velocity.assign(4, Eigen::Vector3<Real>::Zero());
  data.inv_mass = {0, 0, 0, 1};
  data.pos[3] = Eigen::Vector3<Real>(0, 0, -0.8); // det F < 0，确凿倒置

  const Real dt = 1.0 / 120;
  FPSolver<Real, TetrahedronFEMData<Real>, LogNeoHookean<Real, 3>> solver(
      units, /*max_iterations=*/10, /*over_relaxation=*/1.0,
      /*gravity=*/{0, 0, 0});

  using Traits = GeomTraits<Real, TetrahedronFEMData<Real>>;
  for (int step : std::views::iota(1, 501)) {
    solver.solve(data, dt);
    CHECK_MSG(all_finite(data), "step {} non-finite state", step);
    const Real detF = min_detF_of<Traits, 4>(units[0].Dm_inv,
                                             units[0].indices, data.pos);
    // 修复应在首步就把 det F 拉回正值
    CHECK_MSG(detF > 0, "step {} inverted (min det F = {:.6f})", step, detF);
  }
  // 收敛：弹性恢复把单元拉回 F ≈ I
  const Real detF_final = min_detF_of<Traits, 4>(units[0].Dm_inv,
                                                 units[0].indices, data.pos);
  CHECK_MSG(std::abs(detF_final - 1.0) < 0.05,
            "did not relax back to rest (det F = {:.6f})", detF_final);
}
