// 解析导数对拍测试：MaterialBase / FPSolver 的解析梯度与 Hessian
// 对照能量函数的中心差分，应变雅可比对照 strain 的位置差分。
// 每个检查都对比"测试侧独立重建的参考实现"，避免与生产代码共享同一处 bug。
#include "test_util.hpp"

#include <Eigen/Eigenvalues>

#include <ranges>

using namespace xpbd::FP_XPBD;
using Real = xtest::Real;

namespace {

// ---------- 测试侧参考实现：独立对称分量 -> C（含镜像写入） ----------
template <int Dim>
Eigen::Matrix<Real, Dim, Dim>
reference_strain_to_cauchy(const Eigen::Matrix<Real, Dim * (Dim + 1) / 2, 1> &s) {
  constexpr int K = Dim * (Dim + 1) / 2;
  Eigen::Matrix<Real, Dim, Dim> C = Eigen::Matrix<Real, Dim, Dim>::Identity();
  for (int k = 0; k < K; ++k) {
    const auto [r, c] = sym_component_indices<Dim>(k);
    if (r == c)
      C(r, c) += Real(2) * s(k);
    else {
      C(r, c) = s(k);
      C(c, r) = s(k);
    }
  }
  return C;
}

// ---------- MaterialBase 独立 C 分量导数 vs psi 的中心差分 ----------
// 扰动算子沿独立分量 (r,c)：offdiag 同变 C(r,c) 与 C(c,r)。
template <typename Material, int Dim>
void check_material_derivs(const Eigen::Matrix<Real, Dim, Dim> &C, Real mu,
                           Real lam, const char *tag) {
  constexpr int K = Dim * (Dim + 1) / 2;
  using MatC = Eigen::Matrix<Real, Dim, Dim>;
  using VecK = Eigen::Matrix<Real, K, 1>;
  using MatK = Eigen::Matrix<Real, K, K>;

  auto psi_at = [&](const MatC &Cm) {
    return Material::template psi<Real>(Cm, mu, lam);
  };
  auto perturbed = [&](int k, Real t) {
    const auto [r, c] = sym_component_indices<Dim>(k);
    MatC Cm = C;
    Cm(r, c) += t;
    if (r != c)
      Cm(c, r) += t;
    return Cm;
  };

  const VecK g = Material::energy_derivatives(C, mu, lam);
  const MatK H = Material::energy_hessian(C, mu, lam);

  const Real hg = Real(1e-5), hh = Real(1e-3);
  bool g_ok = true, H_ok = true;
  double g_max = 0, H_max = 0;
  for (int k : std::views::iota(0, K)) {
    const Real fd = (psi_at(perturbed(k, hg)) - psi_at(perturbed(k, -hg))) /
                    (Real(2) * hg);
    g_max = std::max(g_max, std::abs(g(k) - fd));
    if (std::abs(g(k) - fd) > Real(1e-6) * (1 + std::abs(fd)))
      g_ok = false;
    for (int l : std::views::iota(0, K)) {
      // 四点模板：∂²psi/∂Ck∂Cl（先 k 后 l 各 ±hh 的组合扰动）
      MatC C_pp = perturbed(k, hh);
      MatC C_pm = perturbed(k, hh);
      MatC C_mp = perturbed(k, -hh);
      MatC C_mm = perturbed(k, -hh);
      for (auto *Cm : {&C_pp, &C_pm, &C_mp, &C_mm}) {
        const Real t = (Cm == &C_pp || Cm == &C_mp) ? hh : -hh;
        const auto [r2, c2] = sym_component_indices<Dim>(l);
        Cm->operator()(r2, c2) += t;
        if (r2 != c2)
          Cm->operator()(c2, r2) += t;
      }
      const Real fd2 =
          (psi_at(C_pp) - psi_at(C_pm) - psi_at(C_mp) + psi_at(C_mm)) /
          (Real(4) * hh * hh);
      H_max = std::max(H_max, std::abs(H(k, l) - fd2));
      if (std::abs(H(k, l) - fd2) > Real(1e-5) * (1 + std::abs(fd2)))
        H_ok = false;
    }
  }
  CHECK_MSG(g_ok, "{} Dim={} grad: max |analytic-FD| = {:.3e}", tag, Dim,
            g_max);
  CHECK_MSG(H_ok, "{} Dim={} hess: max |analytic-FD| = {:.3e}", tag, Dim,
            H_max);
}

// ---------- FPSolver 弹性导数（s 空间） vs elastic_energy 的中心差分 ----------
template <typename Solver>
void check_elastic_derivs_at(const typename Solver::StrainVec &s,
                             typename Solver::StrainVec::Scalar mu,
                             typename Solver::StrainVec::Scalar lam,
                             typename Solver::StrainVec::Scalar V0,
                             const char *tag) {
  using StrainVec = typename Solver::StrainVec;
  using StrainMat = typename Solver::StrainMat;
  using Scalar = typename StrainVec::Scalar;
  constexpr int K = Solver::K;

  StrainVec grad;
  StrainMat hess;
  Solver::elastic_derivs(s, mu, lam, V0, &grad, &hess);
  auto E = [&](const StrainVec &sv) {
    return Solver::elastic_energy(sv, mu, lam, V0);
  };
  // 中心差分 + Richardson 外推消 O(h²) 截断项
  auto central1 = [&](int k, Scalar h) {
    StrainVec sp = s, sm = s;
    sp(k) += h;
    sm(k) -= h;
    return (E(sp) - E(sm)) / (Scalar(2) * h);
  };
  auto stencil2 = [&](int k, int l, Scalar h) {
    StrainVec spp = s, spm = s, smp = s, smm = s;
    spp(k) += h; spp(l) += h;
    spm(k) += h; spm(l) -= h;
    smp(k) -= h; smp(l) += h;
    smm(k) -= h; smm(l) -= h;
    return (E(spp) - E(spm) - E(smp) + E(smm)) / (Scalar(4) * h * h);
  };

  const Scalar h = Scalar(1e-2);
  bool g_ok = true, H_ok = true;
  double g_max = 0, H_max = 0;
  for (int k : std::views::iota(0, K)) {
    const Scalar fd = (Scalar(4) * central1(k, h / 2) - central1(k, h)) /
                      Scalar(3);
    g_max = std::max(g_max, std::abs(grad(k) - fd));
    if (std::abs(grad(k) - fd) > Scalar(1e-6) * (1 + std::abs(fd)))
      g_ok = false;
    for (int l : std::views::iota(0, K)) {
      const Scalar fd2 =
          (Scalar(4) * stencil2(k, l, h / 2) - stencil2(k, l, h)) / Scalar(3);
      H_max = std::max(H_max, std::abs(hess(k, l) - fd2));
      if (std::abs(hess(k, l) - fd2) > Scalar(1e-5) * (1 + std::abs(fd2)))
        H_ok = false;
    }
  }
  CHECK_MSG(g_ok, "{} grad: max |analytic-FD| = {:.3e}", tag, g_max);
  CHECK_MSG(H_ok, "{} hess: max |analytic-FD| = {:.3e}", tag, H_max);
}

// ---------- 应变雅可比 vs strain 的位置中心差分 ----------
template <typename Traits, int Verts, typename RNG>
void check_strain_jacobian(
    const std::array<Eigen::Matrix<Real, Traits::Dim, 1>, Verts> &p,
    const Eigen::Matrix<Real, Traits::Dim, Traits::Dim> &Dm_inv,
    RNG & /*rng*/, const char *tag) {
  constexpr int Dim = Traits::Dim;
  using Vec = Eigen::Matrix<Real, Dim, 1>;
  using StrainVec = typename Traits::StrainVec;
  using StrainJac = typename Traits::StrainJac;

  const StrainJac J = Traits::strain_jacobian(p, Dm_inv);
  const Real h = Real(1e-6);
  bool ok = true;
  double max_err = 0;
  for (auto [v, d] : std::views::cartesian_product(std::views::iota(0, Verts),
                                                   std::views::iota(0, Dim))) {
    std::array<Vec, Verts> pp = p, pm = p;
    pp[v](d) += h;
    pm[v](d) -= h;
    const StrainVec fd =
        (Traits::strain(pp, Dm_inv) - Traits::strain(pm, Dm_inv)) /
        (Real(2) * h);
    const auto col = J.col(v * Dim + d);
    max_err = std::max(max_err, (col - fd).cwiseAbs().maxCoeff());
    if ((col - fd).cwiseAbs().maxCoeff() > Real(1e-7) * (1 + fd.norm()))
      ok = false;
  }
  CHECK_MSG(ok, "{} strain_jacobian: max |analytic-FD| = {:.3e}", tag,
            max_err);
}

// 静止态 C = I：梯度应为 0，Hessian 正定
template <typename Material, int Dim>
void check_rest_state(Real mu, Real lam, const char *tag) {
  constexpr int K = Dim * (Dim + 1) / 2;
  const Eigen::Matrix<Real, Dim, Dim> I =
      Eigen::Matrix<Real, Dim, Dim>::Identity();
  CHECK_NEAR((Material::energy_derivatives(I, mu, lam).norm()), 0.0, 1e-12);
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real, K, K>> es(
      Material::energy_hessian(I, mu, lam));
  CHECK_MSG(es.eigenvalues().minCoeff() > 0.0,
            "{} Dim={} rest hessian PD (min ev = {:.3e})", tag, Dim,
            es.eigenvalues().minCoeff());
}

// 随机正定矩阵：R^T R + I
template <int Dim, typename RNG>
Eigen::Matrix<Real, Dim, Dim> random_spd(RNG &rng) {
  Eigen::Matrix<Real, Dim, Dim> R;
  for (auto [i, j] : std::views::cartesian_product(std::views::iota(0, Dim),
                                                   std::views::iota(0, Dim)))
    R(i, j) = std::uniform_real_distribution<Real>(-1.0, 1.0)(rng);
  return R.transpose() * R + Eigen::Matrix<Real, Dim, Dim>::Identity();
}

} // namespace

// ============ 套件：MaterialBase 导数 vs psi 中心差分 ============
XTEST_SUITE(material_derivs_fd) {
  const auto [mu, lam] = xtest::make_material_constants();

  // 静止态 C = I
  check_rest_state<StableNeoHookean<Real, 2>, 2>(mu, lam, "StableNH");
  check_rest_state<LogNeoHookean<Real, 2>, 2>(mu, lam, "LogNH");
  check_rest_state<StableNeoHookean<Real, 3>, 3>(mu, lam, "StableNH");
  check_rest_state<LogNeoHookean<Real, 3>, 3>(mu, lam, "LogNH");

  // 单位 / 对角拉伸 / 随机正定 C 状态
  {
    std::mt19937 rng(7);
    for (int trial : std::views::iota(0, 3)) {
      Eigen::Matrix<Real, 2, 2> C;
      if (trial == 0)
        C = Eigen::Matrix<Real, 2, 2>::Identity();
      else if (trial == 1)
        C = Eigen::DiagonalMatrix<Real, 2>(1.44, 0.81).toDenseMatrix();
      else
        C = random_spd<2>(rng);
      check_material_derivs<StableNeoHookean<Real, 2>>(C, mu, lam, "StableNH");
      check_material_derivs<LogNeoHookean<Real, 2>>(C, mu, lam, "LogNH");
    }
  }
  {
    std::mt19937 rng(11);
    for (int trial : std::views::iota(0, 3)) {
      Eigen::Matrix<Real, 3, 3> C;
      if (trial == 0)
        C = Eigen::Matrix<Real, 3, 3>::Identity();
      else if (trial == 1)
        C = Eigen::DiagonalMatrix<Real, 3>(1.69, 1.0, 0.64).toDenseMatrix();
      else
        C = random_spd<3>(rng);
      check_material_derivs<StableNeoHookean<Real, 3>>(C, mu, lam, "StableNH");
      check_material_derivs<LogNeoHookean<Real, 3>>(C, mu, lam, "LogNH");
    }
  }
}

// ============ 套件：FPSolver 弹性导数（s 空间） vs 能量中心差分 ============
XTEST_SUITE(fpsolver_elastic_derivs_fd) {
  const auto [mu, lam] = xtest::make_material_constants();

  // ---- 四面体：rest / 拉伸 / 随机(det F > 0) ----
  using TetSolver = FPSolver<Real, TetrahedronFEMData<Real>>;
  using TetTraits = GeomTraits<Real, TetrahedronFEMData<Real>>;
  {
    std::array<Eigen::Matrix<Real, 3, 1>, 4> p0 = {
        Eigen::Matrix<Real, 3, 1>{0, 0, 0}, Eigen::Matrix<Real, 3, 1>{1, 0, 0},
        Eigen::Matrix<Real, 3, 1>{0.2, 1, 0.1},
        Eigen::Matrix<Real, 3, 1>{0.1, 0.3, 0.9}};
    Eigen::Matrix<Real, 3, 3> Dm = TetTraits::shape_matrix(p0);
    const Eigen::Matrix<Real, 3, 3> Dm_inv = Dm.inverse();
    const Real V0 = std::abs(Dm.determinant()) / 6.0;

    check_elastic_derivs_at<TetSolver>(TetSolver::StrainVec::Zero(), mu, lam,
                                       V0, "tet/rest");

    auto p = p0;
    p[3] += Eigen::Matrix<Real, 3, 1>{0.3, -0.2, 0.1};
    check_elastic_derivs_at<TetSolver>(TetTraits::strain(p, Dm_inv), mu, lam,
                                       V0, "tet/stretched");

    std::mt19937 rng(42);
    int checked = 0;
    for (int trial = 0; trial < 50 && checked < 5; ++trial) {
      auto q = p0;
      xtest::perturb<3, 4>(q, 0.25, rng);
      if (TetTraits::computeF(q, Dm_inv).determinant() <= 0)
        continue;
      ++checked;
      check_elastic_derivs_at<TetSolver>(TetTraits::strain(q, Dm_inv), mu, lam,
                                         V0, "tet/random");
    }
    CHECK_MSG(checked >= 3, "tet random states checked = {}", checked);
  }

  // ---- 三角形（局部坐标）：rest / 拉伸 / 随机 ----
  using TriSolver = FPSolver<Real, TriangleFEMData<Real>>;
  using TriTraits = GeomTraits<Real, TriangleFEMData<Real>>;
  {
    std::array<Eigen::Matrix<Real, 2, 1>, 3> p0 = {
        Eigen::Matrix<Real, 2, 1>{0, 0}, Eigen::Matrix<Real, 2, 1>{1, 0},
        Eigen::Matrix<Real, 2, 1>{0.3, 0.8}};
    Eigen::Matrix<Real, 2, 2> Dm = TriTraits::shape_matrix(p0);
    const Eigen::Matrix<Real, 2, 2> Dm_inv = Dm.inverse();
    const Real V0 = std::abs(Dm.determinant()) / 2.0;

    check_elastic_derivs_at<TriSolver>(TriSolver::StrainVec::Zero(), mu, lam,
                                       V0, "tri/rest");

    auto p = p0;
    p[2] += Eigen::Matrix<Real, 2, 1>{0.2, -0.3};
    check_elastic_derivs_at<TriSolver>(TriTraits::strain(p, Dm_inv), mu, lam,
                                       V0, "tri/stretched");

    std::mt19937 rng(43);
    int checked = 0;
    for (int trial = 0; trial < 50 && checked < 5; ++trial) {
      auto q = p0;
      xtest::perturb<2, 3>(q, 0.25, rng);
      if (TriTraits::computeF(q, Dm_inv).determinant() <= 0)
        continue;
      ++checked;
      check_elastic_derivs_at<TriSolver>(TriTraits::strain(q, Dm_inv), mu, lam,
                                         V0, "tri/random");
    }
    CHECK_MSG(checked >= 3, "tri random states checked = {}", checked);
  }
}

// ============ 套件：应变雅可比 vs 位置中心差分 ============
XTEST_SUITE(strain_jacobian_fd) {
  using TetTraits = GeomTraits<Real, TetrahedronFEMData<Real>>;
  using TriTraits = GeomTraits<Real, TriangleFEMData<Real>>;
  std::mt19937 rng(99);

  {
    std::array<Eigen::Matrix<Real, 3, 1>, 4> p = {
        Eigen::Matrix<Real, 3, 1>{0, 0, 0}, Eigen::Matrix<Real, 3, 1>{1, 0, 0},
        Eigen::Matrix<Real, 3, 1>{0.2, 1, 0.1},
        Eigen::Matrix<Real, 3, 1>{0.1, 0.3, 0.9}};
    Eigen::Matrix<Real, 3, 3> Dm_inv = TetTraits::shape_matrix(p).inverse();
    check_strain_jacobian<TetTraits, 4>(p, Dm_inv, rng, "tet/initial");
    for (int trial = 0; trial < 3; ++trial) {
      auto q = p;
      xtest::perturb<3, 4>(q, 0.3, rng);
      check_strain_jacobian<TetTraits, 4>(q, Dm_inv, rng, "tet/random");
    }
  }
  {
    std::array<Eigen::Matrix<Real, 2, 1>, 3> p = {
        Eigen::Matrix<Real, 2, 1>{0, 0}, Eigen::Matrix<Real, 2, 1>{1, 0},
        Eigen::Matrix<Real, 2, 1>{0.3, 0.8}};
    Eigen::Matrix<Real, 2, 2> Dm_inv = TriTraits::shape_matrix(p).inverse();
    check_strain_jacobian<TriTraits, 3>(p, Dm_inv, rng, "tri/initial");
    for (int trial = 0; trial < 3; ++trial) {
      auto q = p;
      xtest::perturb<2, 3>(q, 0.3, rng);
      check_strain_jacobian<TriTraits, 3>(q, Dm_inv, rng, "tri/random");
    }
  }
}

// ============ 套件：strain -> C roundtrip 与能量端到端一致性 ============
// roundtrip 直接检验 offdiag 是否写满镜像（Bug 1 的回归测试）。
XTEST_SUITE(strain_roundtrip) {
  using TetSolver = FPSolver<Real, TetrahedronFEMData<Real>>;
  using TetTraits = GeomTraits<Real, TetrahedronFEMData<Real>>;
  using TriSolver = FPSolver<Real, TriangleFEMData<Real>>;
  using TriTraits = GeomTraits<Real, TriangleFEMData<Real>>;
  std::mt19937 rng(123);

  {
    std::array<Eigen::Matrix<Real, 3, 1>, 4> p0 = {
        Eigen::Matrix<Real, 3, 1>{0, 0, 0}, Eigen::Matrix<Real, 3, 1>{1, 0, 0},
        Eigen::Matrix<Real, 3, 1>{0.2, 1, 0.1},
        Eigen::Matrix<Real, 3, 1>{0.1, 0.3, 0.9}};
    Eigen::Matrix<Real, 3, 3> Dm_inv = TetTraits::shape_matrix(p0).inverse();
    const auto [mu, lam] = xtest::make_material_constants();
    const Real V0 = std::abs(TetTraits::shape_matrix(p0).determinant()) / 6.0;
    for (int trial = 0; trial < 5; ++trial) {
      auto q = p0;
      if (trial > 0)
        xtest::perturb<3, 4>(q, 0.3, rng);
      const Eigen::Matrix<Real, 3, 3> F = TetTraits::computeF(q, Dm_inv);
      const Eigen::Matrix<Real, 3, 3> C_ref = F.transpose() * F;
      const Eigen::Matrix<Real, 3, 3> C =
          TetSolver::strain_to_cauchy(TetTraits::strain(q, Dm_inv));
      for (auto [i, j] : std::views::cartesian_product(std::views::iota(0, 3),
                                                       std::views::iota(0, 3)))
          CHECK_NEAR(C(i, j), C_ref(i, j), 1e-12);
      // 端到端：elastic_energy(s) == V0 * psi(F^T F)
      CHECK_NEAR(TetSolver::elastic_energy(TetTraits::strain(q, Dm_inv), mu,
                                           lam, V0),
                 (V0 * StableNeoHookean<Real, 3>::psi(C_ref, mu, lam)), 1e-9);
    }
  }
  {
    std::array<Eigen::Matrix<Real, 2, 1>, 3> p0 = {
        Eigen::Matrix<Real, 2, 1>{0, 0}, Eigen::Matrix<Real, 2, 1>{1, 0},
        Eigen::Matrix<Real, 2, 1>{0.3, 0.8}};
    Eigen::Matrix<Real, 2, 2> Dm_inv = TriTraits::shape_matrix(p0).inverse();
    const auto [mu, lam] = xtest::make_material_constants();
    const Real V0 = std::abs(TriTraits::shape_matrix(p0).determinant()) / 2.0;
    for (int trial = 0; trial < 5; ++trial) {
      auto q = p0;
      if (trial > 0)
        xtest::perturb<2, 3>(q, 0.3, rng);
      const Eigen::Matrix<Real, 2, 2> F = TriTraits::computeF(q, Dm_inv);
      const Eigen::Matrix<Real, 2, 2> C_ref = F.transpose() * F;
      const Eigen::Matrix<Real, 2, 2> C =
          TriSolver::strain_to_cauchy(TriTraits::strain(q, Dm_inv));
      for (auto [i, j] : std::views::cartesian_product(std::views::iota(0, 2),
                                                       std::views::iota(0, 2)))
          CHECK_NEAR(C(i, j), C_ref(i, j), 1e-12);
      CHECK_NEAR(TriSolver::elastic_energy(TriTraits::strain(q, Dm_inv), mu,
                                           lam, V0),
                 (V0 * StableNeoHookean<Real, 2>::psi(C_ref, mu, lam)), 1e-9);
    }
  }
}
