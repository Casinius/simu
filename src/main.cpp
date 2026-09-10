
#include <print>
#include <vector>

#include "fpxpbd.h"

using Real = double;

static Real total_energy(const std::vector<xpbd::FP_XPBD::TetrahedronFEMData<Real>> &units,
                         const std::vector<Eigen::Vector3<Real>> &pos) {
  using Traits = xpbd::FP_XPBD::GeomTraits<Real, xpbd::FP_XPBD::TetrahedronFEMData<Real>>;
  Real U = 0;
  const size_t n = units[0].indices.size() / 4;
  for (const auto &unit : units)
    for (size_t c = 0; c < n; ++c) {
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

static Real min_detF(const std::vector<xpbd::FP_XPBD::TetrahedronFEMData<Real>> &units,
                     const std::vector<Eigen::Vector3<Real>> &pos) {
  using Traits = xpbd::FP_XPBD::GeomTraits<Real, xpbd::FP_XPBD::TetrahedronFEMData<Real>>;
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

int main() {
  using namespace xpbd::FP_XPBD;

  // 单四面体：三个钉住顶点 + 一个自由顶点，重力下弯曲回摆
  // 静止位置
  std::vector<Eigen::Vector3<Real>> rest = {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

  TetrahedronFEMData<Real> unit;
  unit.indices = {0, 1, 2, 3};
  unit.mu.resize(1);
  unit.lambda_mat.resize(1);
  std::vector<TetrahedronFEMData<Real>> units = {unit};
  setup_units(units, rest);

  const Real E = 1e4, nu = 0.3;
  units[0].mu[0] = E / (2 * (1 + nu));
  units[0].lambda_mat[0] = E * nu / ((1 + nu) * (1 - 2 * nu));

  xpbd::PhysicsData<Real> data;
  data.pos = rest;
  data.pos_prev = rest;
  data.velocity.assign(4, Eigen::Vector3<Real>::Zero());
  data.inv_mass = {0, 0, 0, 1}; // 仅顶点 3 自由

  // 自由顶点初始偏移，制造形变
  data.pos[3] += Eigen::Vector3<Real>(0.3, -0.2, 0.1);

  const Real dt = 1.0 / 60;
  FPSolver<Real, TetrahedronFEMData<Real>> solver(
      units, /*maxIter=*/10, /*over_relax=*/1.0, /*gravity=*/{0, -9.8, 0});

  for (int step = 1; step <= 3000; ++step) {
    solver.solve(data, dt);
    if (step % 10 == 0)
      std::print("step {:3d}  E = {:12.6e}  min det F = {:10.6f}  |v| = {:10.6f}\n",
                 step, total_energy(units, data.pos), min_detF(units, data.pos),
                 data.velocity[3].norm());
  }
  return 0;
}
