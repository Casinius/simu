#ifndef FPXPBD_SOLVER_H
#define FPXPBD_SOLVER_H

#include "xpbd_base.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>
#include <vector>

namespace xpbd {

namespace FP_XPBD {

// 前置声明：几何 traits 的各单元特化在下方给出
template <typename Scalar, typename UnitData> struct GeomTraits;

// 容纳某种 FEM 单元数据的容器
//（vector<TetrahedronFEMData<Scalar>> / vector<TriangleFEMData<Scalar>>，
//  且元素须有对应 Scalar 维度的几何 traits 特化）
template <typename C, typename Scalar>
concept UnitContainer = std::ranges::range<C> && requires(C &c) {
  typename GeomTraits<Scalar, std::remove_cvref_t<decltype(*c.begin())>>;
};

// FEM 单元类型：具有对应 Scalar 维度的几何 traits 特化
template <typename Scalar, typename UnitData>
concept FEMUnit = requires {
  typename GeomTraits<Scalar, UnitData>::Mat;
  typename GeomTraits<Scalar, UnitData>::Vec;
};

// 把一段单元顶点索引（views::chunk 的元素）固化为 std::array
template <std::size_t Verts, std::ranges::input_range R>
  requires std::convertible_to<std::ranges::range_value_t<R>, std::size_t>
[[nodiscard]] constexpr std::array<std::size_t, Verts> cell_indices(R &&cell) {
  std::array<std::size_t, Verts> out{};
  std::ranges::copy(cell, out.begin());
  return out;
}

// ========== 数据结构：四面体 / 三角形 FEM 单元 ==========
template <typename Scalar> struct TetrahedronFEMData {
  std::vector<size_t> indices; // [i0, i1, i2, i3, ...] 每 4 个一单元

  // 每单元参考构型逆矩阵（3x3）
  std::vector<Mat3<Scalar>> Dm_inv;

  // 每单元材料参数（拉梅常数）
  std::vector<Scalar> mu;
  std::vector<Scalar> lambda_mat;

  // 每单元静止体积 = |det(Dm)| / 6
  std::vector<Scalar> rest_volume;
};

template <typename Scalar> struct TriangleFEMData {
  std::vector<size_t> indices; // [i0, i1, i2, ...] 每 3 个一单元

  // 每单元参考构型逆矩阵（2x2，材料局部坐标下）
  std::vector<Mat2<Scalar>> Dm_inv;

  // 每单元材料参数
  std::vector<Scalar> mu;
  std::vector<Scalar> lambda_mat;

  // 每单元静止面积 = |det(Dm_local)| / 2（单位厚度）
  std::vector<Scalar> rest_volume;

  // 每单元 2x3 材料坐标架：行 = {单位化静止边 e1, e1 正交补内单位化静止边 e2}
  // 世界 -> 局部：x_loc = frame * p（正交行 => 逆映射 = frame^T）
  std::vector<Eigen::Matrix<Scalar, 2, 3>> frames;
};

// ============ 材料策略：作用在 Green 应变上的约化能量 ============
// C = 2E + I，trC = ||F||^2，J = sqrt(det C)。
// StableNeoHookean（GPBD 论文式 21）：
//   U = V [ mu/2 (trC - Dim) - mu (J-1) + lambda/2 (J-1)^2 ]
// LogNeoHookean（GPBD 论文式 20，带倒置屏障）：
//   U = V [ mu/2 (trC - Dim) - mu log J + lambda/2 (log J)^2 ]
// psi 接收 C 矩阵；det C > 0 由倒置修复保证。

// 独立对称分量序：对角在前 (0..Dim-1)，offdiag 按行序 (a<b)；返回 (row, col)。
template <int Dim> constexpr std::array<int, 2>
sym_component_indices(int component_index) {  if (component_index < Dim)
    return {component_index, component_index};
  int rest = component_index - Dim;
  int component_row = 0;
  while (rest >= Dim - 1 - component_row) {
    rest -= Dim - 1 - component_row;
    ++component_row;
  }
  return {component_row, component_row + 1 + rest};
}

// 编译期枚举全部独立对称分量 (row, col)（对角在前，offdiag 按行序）。
template <int Dim>
consteval std::array<std::array<int, 2>, Dim * (Dim + 1) / 2>
sym_component_table() {
  constexpr int K = Dim * (Dim + 1) / 2;
  std::array<std::array<int, 2>, K> table{};
  for (int k = 0; k < K; ++k)
    table[k] = sym_component_indices<Dim>(k);
  return table;
}

// 对称 C 空间的 NeoHookean 族解析导数。Derived 须提供：
//   static Scalar j_potential_deriv(Scalar J, Scalar mu, Scalar lam)
//     —— d/dJ [体积势项]（StableNH: lam*(J-1)-mu；LogNH: (lam*log(J)-mu)/J）
//   static Scalar j_potential_second_deriv(Scalar J, Scalar mu, Scalar lam)
//     —— d²/dJ² [体积势项]（StableNH: lam；LogNH: (lam - lam*log(J) + mu)/J²）
//   template <class T> static T psi(...)
template <std::floating_point Scalar, int Dim, typename Derived>
struct MaterialBase {
  static constexpr int SymComponentCount = Dim * (Dim + 1) / 2;

  // 能量在 det C < 0 是否可评估；true 时求解器启用 SVD 倒置修复
  //（LogNH 的 log J 需要正 det，StableNH 的 (J-1)² 天然屏障无需修复）。
  static constexpr bool needs_inversion_fix = false;

  // 求解器坐标约定：offdiag 为独立单坐标（同变 C_ij 与 C_ji），
  // 相对逐分量偏导 ∂U/∂C_ij = ∂U/∂C_ji 差因子 2。
  static constexpr Scalar component_scale(int component_row,
                                          int component_col) {
    return component_row == component_col ? Scalar(1) : Scalar(2);
  }

  // dU/dC 按独立对称分量 (c00, c11, c22, c01, c02, c12) 序，即"全矩阵坐标"约定。
  // U = mu/2 (trC - Dim) + V(J)：剪切项给 mu/2·I，体积项经 dJ/dC = (J/2)·Cinv（对称 C）。
  // offdiag 乘 component_scale(=2) 折叠镜像项。
  // 用法约定：调用方保证 C 与此处分量坐标一致（offdiag 单坐标同时携带镜像）。
  [[nodiscard]] static Eigen::Matrix<Scalar, SymComponentCount, 1>
  energy_derivatives(const Eigen::Matrix<Scalar, Dim, Dim> &deformation_cauchy,
                     Scalar shear_modulus, Scalar bulk_modulus) {
    const Eigen::Matrix<Scalar, Dim, Dim> cauchy_inverse =
        deformation_cauchy.inverse();
    const Scalar jacobian = sqrt(deformation_cauchy.determinant());
    const Scalar j_potential_coefficient =
        Derived::j_potential_deriv(jacobian, shear_modulus, bulk_modulus);
    constexpr auto component_table = sym_component_table<Dim>();
    Eigen::Matrix<Scalar, SymComponentCount, 1> gradient;
    for (auto [component_index, component] :
         std::views::enumerate(component_table)) {
      const auto [component_row, component_col] = component;
      Scalar entry = cauchy_inverse(component_row, component_col) *
                     (j_potential_coefficient * jacobian / Scalar(2));
      if (component_row == component_col)
        entry += shear_modulus / Scalar(2);
      gradient(component_index) =
          entry * component_scale(component_row, component_col);
    }
    return gradient;
  }

  // d²U/dC²_{ij,kl} = V''·dJ_{ij}·dJ_{kl} + V'·d²J_{ij,kl}，同序。
  // dJ_{ij} = (J/2)·Cinv_{ij}；
  // d²J_{ij,kl} = (J/4)·Cinv_{ij}·Cinv_{kl} - (J/4)·(Cinv_{ik}·Cinv_{jl} +
  //               Cinv_{il}·Cinv_{jk})，(ij)(kl) 为独立分量 (row, col)。
  // 同 energy_derivatives，行列各乘 component_scale 折叠镜像项。
  [[nodiscard]] static Eigen::Matrix<Scalar, SymComponentCount, SymComponentCount>
  energy_hessian(const Eigen::Matrix<Scalar, Dim, Dim> &deformation_cauchy,
                 Scalar shear_modulus, Scalar bulk_modulus) {
    const Eigen::Matrix<Scalar, Dim, Dim> cauchy_inverse =
        deformation_cauchy.inverse();
    const Scalar jacobian = sqrt(deformation_cauchy.determinant());
    const Scalar j_potential_first =
        Derived::j_potential_deriv(jacobian, shear_modulus, bulk_modulus);
    const Scalar j_potential_second =
        Derived::j_potential_second_deriv(jacobian, shear_modulus, bulk_modulus);
    constexpr auto component_table = sym_component_table<Dim>();
    Eigen::Matrix<Scalar, SymComponentCount, SymComponentCount> hessian;
    for (auto [component_row, r1c1] : std::views::enumerate(component_table)) {
      const auto [r1, c1] = r1c1;
      for (int component_col : std::views::iota(0, component_row + 1)) {
        const auto [r2, c2] = component_table[component_col];
        const Scalar d_jacobian_row = jacobian / Scalar(2) *
                                      cauchy_inverse(r1, c1);
        const Scalar d_jacobian_col = jacobian / Scalar(2) *
                                      cauchy_inverse(r2, c2);
        // d²J_{ij,kl} = (J/4)·Cinv_{ij}·Cinv_{kl}
        //             - (J/4)·(Cinv_{ik}·Cinv_{jl} + Cinv_{il}·Cinv_{jk})，
        // 分量为对称 C 的独立分量 (row, col)。
        const Scalar cross_term =
            cauchy_inverse(r1, r2) * cauchy_inverse(c1, c2) +
            cauchy_inverse(r1, c2) * cauchy_inverse(c1, r2);
        const Scalar d2_jacobian =
            jacobian / Scalar(4) * cauchy_inverse(r1, c1) *
                cauchy_inverse(r2, c2) -
            jacobian / Scalar(4) * cross_term;
        const Scalar entry =
            j_potential_second * d_jacobian_row * d_jacobian_col +
            j_potential_first * d2_jacobian;
        const Scalar scale_factor =
            component_scale(r1, c1) * component_scale(r2, c2);
        hessian(component_row, component_col) = entry * scale_factor;
        hessian(component_col, component_row) = entry * scale_factor;
      }
    }
    return hessian;
  }
};

template <typename M, typename Scalar, int Dim>
concept NeoHookeanMaterial =
    std::floating_point<Scalar> &&
    requires(const Eigen::Matrix<Scalar, Dim, Dim> &C, Scalar J, Scalar mu,
             Scalar lam) {
      { M::j_potential_deriv(J, mu, lam) } -> std::same_as<Scalar>;
      { M::j_potential_second_deriv(J, mu, lam) } -> std::same_as<Scalar>;
      { M::template psi<Scalar>(C, mu, lam) } -> std::same_as<Scalar>;
    } && std::is_base_of_v<MaterialBase<Scalar, Dim, M>, M>;

template <std::floating_point Scalar, int Dim>
struct StableNeoHookean
    : MaterialBase<Scalar, Dim, StableNeoHookean<Scalar, Dim>> {
  [[nodiscard]] static Scalar j_potential_deriv(Scalar J, Scalar mu,
                                                Scalar lam) {
    return lam * (J - Scalar(1)) - mu;
  }
  [[nodiscard]] static Scalar
  j_potential_second_deriv(Scalar J, Scalar mu, Scalar lam) {
    return lam;
  }
  template <std::floating_point T>
  [[nodiscard]] static T psi(const Eigen::Matrix<T, Dim, Dim> &C, T mu, T lam) {
    T trC = C.trace();
    T J = sqrt(C.determinant());
    T arg = J - T(1);
    return mu / T(2) * (trC - T(Dim)) - mu * arg + lam / T(2) * arg * arg;
  }
};

template <std::floating_point Scalar, int Dim>
struct LogNeoHookean : MaterialBase<Scalar, Dim, LogNeoHookean<Scalar, Dim>> {
  static constexpr bool needs_inversion_fix = true;

  // d/dJ [ -mu log J + lam/2 (log J)^2 ] = (lam log J - mu)/J
  [[nodiscard]] static Scalar j_potential_deriv(Scalar J, Scalar mu,
                                                Scalar lam) {
    return (lam * log(J) - mu) / J;
  }
  // d²/dJ² = (lam - lam log J + mu)/J²
  [[nodiscard]] static Scalar
  j_potential_second_deriv(Scalar J, Scalar mu, Scalar lam) {
    return (lam - lam * log(J) + mu) / (J * J);
  }
  template <std::floating_point T>
  [[nodiscard]] static T psi(const Eigen::Matrix<T, Dim, Dim> &C, T mu, T lam) {
    T trC = C.trace();
    T J = sqrt(C.determinant());
    T lj = log(J);
    return mu / T(2) * (trC - T(Dim)) - mu * lj + lam / T(2) * lj * lj;
  }
};

// ============ 单元几何特化 ============
// 通用单纯形 FEM 几何（Dim 维、Verts = Dim + 1 顶点）：
// F = Ds·Dm⁻¹，对称 Green 应变及其解析雅可比。
// Derived 须提供：static Mat shape_matrix(const std::array<Vec, Verts> &p)
//（单纯形构造——顶点 p₁..p_Dim 相对 p₀ 取差——留在各特化）。
template <std::floating_point Scalar, int Dim, int Verts, typename Derived>
  requires(Verts == Dim + 1)
struct GeomTraitsBase {
  static constexpr int StrainSize = Dim * (Dim + 1) / 2;
  using Mat = Eigen::Matrix<Scalar, Dim, Dim>;
  using Vec = Eigen::Matrix<Scalar, Dim, 1>;
  using StrainVec = Eigen::Matrix<Scalar, StrainSize, 1>;
  using StrainJac = Eigen::Matrix<Scalar, StrainSize, Dim * Verts>;

  // 变形梯度 F = Ds * reference_inverse（参考构型形状矩阵之逆）
  [[nodiscard]] static Mat computeF(const std::array<Vec, Verts> &pos,
                                    const Mat &reference_inverse) {
    return Derived::shape_matrix(pos) * reference_inverse;
  }

  // 对称 Green 应变 s = C - I 的分量（C = F^T F；能量以 C = s + I 接收）
  // 顺序：exx eyy ezz exy exz eyz（2D：exx eyy exy）
  [[nodiscard]] static StrainVec strain(const std::array<Vec, Verts> &pos,
                                        const Mat &reference_inverse) {
    const Mat F = computeF(pos, reference_inverse);
    StrainVec strain_vector;
    int component_index = Dim;
    for (int a : std::views::iota(0, Dim)) {
      strain_vector(a) = F.col(a).squaredNorm() - Scalar(1);
      for (int b : std::views::iota(a + 1, Dim))
        strain_vector(component_index++) = F.col(a).dot(F.col(b));
    }
    return strain_vector;
  }

  // 解析 ds/dx，在当前 x 处取值（k x Dim*Verts）
  // dF_ab/dp_{0,a} = -sum_j reference_inverse(j,b)
  // dF_ab/dp_{v,a} =  reference_inverse(v-1,b)   (v = 1..Dim；仅对角块非零)
  // ds_aa/dF_cd    = 2 F(c,a) delta_ac
  // ds_ab/dF_cd    = F(c,b) delta_ac + F(c,a) delta_bc (a < b)
  [[nodiscard]] static StrainJac
  strain_jacobian(const std::array<Vec, Verts> &pos,
                  const Mat &reference_inverse) {
    Eigen::Matrix<Scalar, Dim * Dim, Dim * Verts> dFdx; // (ab, vd)
    dFdx.setZero();
    for (int a : std::views::iota(0, Dim))
      for (int b : std::views::iota(0, Dim)) {
        dFdx(a * Dim + b, a) = -reference_inverse.col(b).sum(); // 顶点 0
        for (int v : std::views::iota(1, Dim + 1))
          dFdx(a * Dim + b, v * Dim + a) = reference_inverse(v - 1, b);
      }
    const Mat F = computeF(pos, reference_inverse);
    Eigen::Matrix<Scalar, StrainSize, Dim * Dim> dsdF;
    dsdF.setZero();
    for (int a : std::views::iota(0, Dim))
      for (int c : std::views::iota(0, Dim))
        dsdF(a, c * Dim + a) = Scalar(2) * F(c, a);
    int component_index = Dim;
    for (int a : std::views::iota(0, Dim))
      for (int b : std::views::iota(a + 1, Dim)) {
        for (int c : std::views::iota(0, Dim)) {
          dsdF(component_index, c * Dim + a) += F(c, b);
          dsdF(component_index, c * Dim + b) += F(c, a);
        }
        ++component_index;
      }
    return dsdF * dFdx;
  }
};

// 四面体特化（Dim=3, k=6：exx eyy ezz exy exz eyz）
template <typename Scalar>
struct GeomTraits<Scalar, TetrahedronFEMData<Scalar>>
    : GeomTraitsBase<Scalar, 3, 4,
                     GeomTraits<Scalar, TetrahedronFEMData<Scalar>>> {
  static constexpr int Dim = 3;
  static constexpr int Verts = 4;
  static constexpr int StrainSize = 6;
  using Mat = typename GeomTraitsBase<Scalar, 3, 4,
                                      GeomTraits<Scalar, TetrahedronFEMData<Scalar>>>::Mat;
  using Vec = typename GeomTraitsBase<Scalar, 3, 4,
                                      GeomTraits<Scalar, TetrahedronFEMData<Scalar>>>::Vec;

  // 形状矩阵：列 = p_i - p_0
  [[nodiscard]] static Mat shape_matrix(const std::array<Vec, Verts> &p) {
    Mat Dm;
    Dm.col(0) = p[1] - p[0];
    Dm.col(1) = p[2] - p[0];
    Dm.col(2) = p[3] - p[0];
    return Dm;
  }
};

// 三角形壳特化（Dim=2, k=3：exx eyy exy；世界 -> 局部由 frame 完成）
template <typename Scalar>
struct GeomTraits<Scalar, TriangleFEMData<Scalar>>
    : GeomTraitsBase<Scalar, 2, 3,
                     GeomTraits<Scalar, TriangleFEMData<Scalar>>> {
  static constexpr int Dim = 2;
  static constexpr int Verts = 3;
  static constexpr int StrainSize = 3;
  using Mat = typename GeomTraitsBase<Scalar, 2, 3,
                                      GeomTraits<Scalar, TriangleFEMData<Scalar>>>::Mat;
  using Vec = typename GeomTraitsBase<Scalar, 2, 3,
                                      GeomTraits<Scalar, TriangleFEMData<Scalar>>>::Vec;

  [[nodiscard]] static Mat shape_matrix(const std::array<Vec, Verts> &p) {
    Mat Dm;
    Dm.col(0) = p[1] - p[0];
    Dm.col(1) = p[2] - p[0];
    return Dm;
  }
};


// ============ 工厂：从静止顶点位置填充 Dm_inv / rest_volume / frames ============
template <std::floating_point Scalar, UnitContainer<Scalar> UnitData>
void setup_units(UnitData &units,
                 const std::vector<Eigen::Vector3<Scalar>> &rest_positions) {
  using Elem = std::remove_cvref_t<decltype(*std::declval<UnitData &>().begin())>;
  using T = GeomTraits<Scalar, Elem>;
  constexpr int Dim = T::Dim;
  constexpr int Verts = T::Verts;
  for (auto &unit : units) {
    const size_t n_cells = unit.indices.size() / Verts;
    unit.Dm_inv.reserve(n_cells);
    if constexpr (Dim == 2)
      unit.frames.reserve(n_cells);

    for (auto [c, cell] :
         unit.indices | std::views::chunk(Verts) | std::views::enumerate) {
      std::array<Eigen::Vector3<Scalar>, Verts> pw;
      for (auto [i, vertex] : std::views::enumerate(cell))
        pw[i] = rest_positions[vertex];

      if constexpr (Dim == 3) {
        const typename T::Mat Dm = T::shape_matrix(pw);
        unit.Dm_inv.push_back(Dm.inverse());
        unit.rest_volume.push_back(std::abs(Dm.determinant()) / Scalar(6));
      } else {
        // 2x3 材料坐标架：e1 = 单位化静止边 (p1-p0)；e2 在 e1 正交补内单位化
        Eigen::Matrix<Scalar, 2, 3> frame;
        Eigen::Matrix<Scalar, 3, 1> e1 = (pw[1] - pw[0]).normalized();
        Eigen::Matrix<Scalar, 3, 1> e2 = pw[2] - pw[0];
        e2 = (e2 - e1 * e1.dot(e2)).normalized();
        frame.row(0) = e1;
        frame.row(1) = e2;
        unit.frames.push_back(frame);
        // 局部静止坐标（平移不变，只差公共偏移）
        std::array<Eigen::Matrix<Scalar, 2, 1>, Verts> p;
        for (int i = 0; i < Verts; ++i)
          p[i] = frame * pw[i];
        const typename T::Mat Dm = T::shape_matrix(p);
        unit.Dm_inv.push_back(Dm.inverse());
        unit.rest_volume.push_back(std::abs(Dm.determinant()) / Scalar(2));
      }
    }
  }
}
// ============ FP-PXPBD 求解器（GPBD Algorithm 1，Gauss-Seidel 形式） ============
// 目标 (19)：min_{Dl in R^k} 1/2 ||d + W S^T Dl||^2_{W^{-1}} + U_hat(x + W S^T Dl)
// 其中 U_hat(s) = rest_volume * Material::psi(C = s + I, mu, lambda_mat)。
// 注：autodiff v1.1.2 与 Eigen 5.0.1 不兼容，Û 的梯度/Hessian 用
// MaterialBase::energy_derivatives / energy_hessian 的 C 空间解析式
//（C = s + I ⇒ dc/ds = I，无链式因子）；应变雅可比 S 为解析式。
template <std::floating_point Scalar, typename UnitData,
          NeoHookeanMaterial<Scalar, GeomTraits<Scalar, UnitData>::Dim>
              Material = StableNeoHookean<
                  Scalar, GeomTraits<Scalar, UnitData>::Dim>>
  requires FEMUnit<Scalar, UnitData>
class FPSolver : public ISolver<PhysicsData<Scalar>, Scalar> {
public:
  using Traits = GeomTraits<Scalar, UnitData>;
  static constexpr int Dim = Traits::Dim;
  static constexpr int Verts = Traits::Verts;
  static constexpr int K = Traits::StrainSize;
  static constexpr int N = Dim * Verts;
  using Vec = typename Traits::Vec;
  using Mat = typename Traits::Mat;
  using StrainVec = Eigen::Matrix<Scalar, K, 1>;
  using StrainMat = Eigen::Matrix<Scalar, K, K>;
  using StrainJac = Eigen::Matrix<Scalar, K, N>;
  using Dvec = Eigen::Matrix<Scalar, N, 1>;

  FPSolver(std::vector<UnitData> &units, int max_iterations = 10,
           Scalar over_relaxation = Scalar(1),
           Vec3<Scalar> gravity = Vec3<Scalar>::Zero())
      : units_(units), max_iterations_(max_iterations),
        over_relaxation_(over_relaxation), gravity_(std::move(gravity)) {}

  void solve(PhysicsData<Scalar> &data, Scalar dt) override {
    // 1. 预测：备份 pos_prev（速度恢复基准），半隐式欧拉 + 重力
    data.pos_prev = data.pos;
    integrate_velocity_verlet(data, gravity_, dt);

    // 2. 每单元位移缓存清零
    d_.assign(units_.size(), {});
    for (auto &&[u, unit] : std::views::enumerate(units_))
      d_[u].assign(unit.indices.size() / Verts, Dvec::Zero());

    // 3. GPBD 迭代：逐单元 Gauss-Seidel
    for (int iter = 0; iter < max_iterations_; ++iter)
      for (auto &&[u, unit] : std::views::enumerate(units_))
        for (auto &&[c, cell] : unit.indices | std::views::chunk(Verts) |
                                    std::views::enumerate)
          process_unit(data, unit, c,
                       cell_indices<Verts>(cell), d_[u][c], dt);

    // 4. 速度恢复
    update_velocities(data, dt);
  }

  // ---- 解析导数与能量工具（public：无状态静态函数，供测试对拍复用） ----
public:
  // ---- 对称 C = s + I 的重组（s = C − I = 2E；elastic_energy / elastic_derivs 共用） ----
  [[nodiscard]] static Eigen::Matrix<Scalar, Dim, Dim>
  strain_to_cauchy(const StrainVec &strain_vector) {
    Eigen::Matrix<Scalar, Dim, Dim> deformation_cauchy =
        Eigen::Matrix<Scalar, Dim, Dim>::Identity();
    for (auto [component_index, component] :
         std::views::enumerate(sym_component_table<Dim>())) {
      const auto [component_row, component_col] = component;
      Scalar entry = strain_vector(component_index);
      if (component_row == component_col) {
        deformation_cauchy(component_row, component_col) += entry;
      } else {
        // offdiag 单坐标同时驱动 C(r,c) 与 C(c,r)
        deformation_cauchy(component_row, component_col) = entry;
        deformation_cauchy(component_col, component_row) = entry;
      }
    }
    return deformation_cauchy;
  }

  // ---- U_hat(s) = V0 * psi(s + I) ----
  [[nodiscard]] static Scalar elastic_energy(const StrainVec &s, Scalar mu,
                                             Scalar lam, Scalar V0) {
    return V0 * Material::template psi<Scalar>(strain_to_cauchy(s), mu, lam);
  }

  // ---- Û 的梯度 / Hessian：解析。C = s + I ⇒ dc/ds = I（独立分量坐标），
  // 无额外链式因子；offdiag 镜像折算已由 MaterialBase 的 component_scale 处理 ----
  static void elastic_derivs(const StrainVec &strain_vector, Scalar mu,
                             Scalar lam, Scalar V0, StrainVec *elastic_gradient,
                             StrainMat *elastic_hessian) {
    const Eigen::Matrix<Scalar, Dim, Dim> deformation_cauchy =
        strain_to_cauchy(strain_vector);
    if (elastic_gradient)
      *elastic_gradient =
          V0 * Material::energy_derivatives(deformation_cauchy, mu, lam);
    if (elastic_hessian)
      *elastic_hessian =
          V0 * Material::energy_hessian(deformation_cauchy, mu, lam);
  }

  // ---- 倒置修复：SVD 最小奇异值取反 + 质心保持 ----
  // 在局部坐标 pos 上原位修复并返回是否触发。
private:
  [[nodiscard]] static bool fix_inversion(std::array<Vec, Verts> &pos,
                                          const Mat &Dm_inv) {
    const Mat F = Traits::computeF(pos, Dm_inv);
    if (F.determinant() >= Scalar(0))
      return false;
    const Eigen::JacobiSVD<Mat> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat Sigma = svd.singularValues().asDiagonal().toDenseMatrix();
    Sigma(Dim - 1, Dim - 1) = -Sigma(Dim - 1, Dim - 1); // 最小奇异值取反
    const Mat Ds2 = svd.matrixU() * Sigma * svd.matrixV().transpose() *
                    Dm_inv.inverse(); // Ds' = F' * Dm
    std::array<Vec, Verts> q = pos;
    for (int j : std::views::iota(0, Dim))
      q[j + 1] = pos[0] + Ds2.col(j);
    Vec c0 = Vec::Zero(), cq = Vec::Zero();
    for (const auto &p : pos)
      c0 += p;
    for (const auto &p : q)
      cq += p;
    const Vec shift = (c0 - cq) / Scalar(Verts);
    for (auto [p, q_i] : std::views::zip(pos, q))
      p = q_i + shift;
    return true;
  }

  // ---- 每单元（顶点索引 vidx，容器内序号 cell）GPBD 子问题求解与提交 ----
  void process_unit(PhysicsData<Scalar> &data, UnitData &unit, size_t cell,
                    const std::array<size_t, Verts> &vidx, Dvec &d,
                    Scalar dt) {
    std::array<Vec, Verts> pos;
    Eigen::Matrix<Scalar, 3, 2> frame_t; // 局部 -> 世界（仅三角形用）
    if constexpr (Dim == 3) {
      std::ranges::transform(vidx, pos.begin(),
                             [&](size_t v) { return data.pos[v]; });
    } else {
      const auto &frame = unit.frames[cell];
      frame_t = frame.transpose();
      Eigen::Matrix<Scalar, 2, Verts> loc;
      for (auto [i, v] : std::views::enumerate(vidx))
        loc.col(i) = frame * data.pos[v];
      for (auto [i, p_i] : std::views::enumerate(pos))
        p_i = loc.col(i);
    }

    const Mat &Dm_inv = unit.Dm_inv[cell];
    const Scalar V0 = unit.rest_volume[cell];
    const Scalar mu = unit.mu[cell];
    const Scalar lam = unit.lambda_mat[cell];

    if constexpr (Material::needs_inversion_fix) {
      if (fix_inversion(pos, Dm_inv)) {
        // 修复写回世界坐标（三角形：frame^T；四面体：恒等）
        for (auto [i, v] : std::views::enumerate(vidx)) {
          if constexpr (Dim == 3)
            data.pos[v] = pos[i];
          else
            data.pos[v] = frame_t * pos[i];
        }
      }
    }

    const StrainVec s = Traits::strain(pos, Dm_inv);
    const StrainJac S0 = Traits::strain_jacobian(pos, Dm_inv);

    // W = diag(dt^2 * inv_mass)，固定粒子为 0
    Dvec w;
    for (auto [i, v] : std::views::enumerate(vidx))
      w.template segment<Dim>(i * Dim)
          .setConstant(dt * dt * data.inv_mass[v]);

    // ---- 子问题 (19)：f(Dl) 及其 k 维梯度/Hessian ----
    // y(Dl) = W S0^T Dl；u = d + y；x~ = x + y
    // f   = 1/2 u^T W^{-1} u + U_hat(s(x~))
    // g   = S0 W u + S0 W S(x~)^T grad_s
    // H   = S0 W S0^T + (S0 W S(x~)^T) H_s (S0 W S(x~)^T)^T
    auto eval_f = [&](const StrainVec &Dl, StrainVec *grad,
                      StrainMat *hess) -> Scalar {
      const Dvec y = w.asDiagonal() * (S0.transpose() * Dl);
      std::array<Vec, Verts> pt = pos;
      for (auto &&[p_i, y_i] :
           std::views::zip(pt, y | std::views::chunk(Dim)))
        p_i += Eigen::Map<const Eigen::Matrix<Scalar, Dim, 1>>(y_i.data());
      const StrainVec s_new = Traits::strain(pt, Dm_inv);

      Scalar U = elastic_energy(s_new, mu, lam, V0);
      // S(x~) 与材料导数在 grad / hess 间共用
      StrainJac St;
      StrainVec gs;
      StrainMat Hs;
      if (grad || hess) {
        St = Traits::strain_jacobian(pt, Dm_inv);
        elastic_derivs(s_new, mu, lam, V0, grad ? &gs : nullptr,
                       hess ? &Hs : nullptr);
      }

      const Dvec u = d + y;
      const Scalar quad = std::ranges::fold_left(
          std::views::iota(0, N), Scalar(0), [&](Scalar acc, int i) {
            return w(i) > Scalar(0) ? acc + u(i) * u(i) / w(i) : acc;
          });
      if (grad)
        *grad = S0 * w.asDiagonal() * u +
                S0 * w.asDiagonal() * (St.transpose() * gs);
      if (hess) {
        const StrainMat M = S0 * w.asDiagonal() * St.transpose();
        *hess = S0 * w.asDiagonal() * S0.transpose() + M.transpose() * Hs * M;
      }
      return Scalar(0.5) * quad + U;
    };

    // Newton（<= 8 步，从 Dl = 0 起）+ 特征值截断 + 回溯线搜索
    StrainVec Dl = StrainVec::Zero();
    StrainVec grad;
    StrainMat H;
    Scalar f = eval_f(Dl, &grad, &H);
    for (int it = 0; it < 8; ++it) {
      if (grad.norm() < Scalar(1e-10))
        break;
      // 正定化：SelfAdjointEigenSolver 特征值下限截断
      Eigen::SelfAdjointEigenSolver<StrainMat> es(H);
      Scalar mean_abs = es.eigenvalues().cwiseAbs().mean();
      Scalar floor_e = std::max(Scalar(1e-8) * mean_abs, Scalar(1e-12));
      StrainVec ev = es.eigenvalues().cwiseMax(floor_e);
      StrainMat Hreg =
          es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
      StrainVec step = Hreg.ldlt().solve(-grad);
      // 回溯线搜索（每次减半，<= 16 次）
      Scalar alpha = Scalar(1);
      bool accepted = false;
      for (int ls = 0; ls < 16; ++ls) {
        StrainVec Dl_new = Dl + alpha * step;
        StrainVec g2;
        Scalar f2 = eval_f(Dl_new, &g2, nullptr);
        if (f2 < f) {
          Dl = Dl_new;
          f = f2;
          grad = g2;
          accepted = true;
          break;
        }
        alpha *= Scalar(0.5);
      }
      if (!accepted)
        break;
    }

    // 提交：Delta d = W S0^T Dl；位置更新（局部 -> 世界），带过松弛
    const Dvec dd = w.asDiagonal() * (S0.transpose() * Dl);
    d += dd;
    for (auto [i, v] : std::views::enumerate(vidx)) {
      if constexpr (Dim == 3)
        data.pos[v] += over_relaxation_ * dd.template segment<Dim>(i * Dim);
      else
        data.pos[v] +=
            over_relaxation_ * (frame_t * dd.template segment<Dim>(i * Dim));
    }
  }

  std::vector<UnitData> &units_;
  int max_iterations_;
  Scalar over_relaxation_;
  Vec3<Scalar> gravity_;
  std::vector<std::vector<Dvec>> d_;
};

} // namespace FP_XPBD

} // namespace xpbd
#endif
