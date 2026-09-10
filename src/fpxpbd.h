#ifndef FPXPBD_SOLVER_H
#define FPXPBD_SOLVER_H

#include "xpbd_base.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace xpbd {

namespace FP_XPBD {

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
template <typename Scalar, int Dim> struct StableNeoHookean {
  template <class T>
  static T psi(const Eigen::Matrix<T, Dim, Dim> &C, T mu, T lam) {
    T trC = C.trace();
    T J = sqrt(C.determinant());
    T arg = J - T(1);
    return mu / T(2) * (trC - T(Dim)) - mu * arg + lam / T(2) * arg * arg;
  }
};

template <typename Scalar, int Dim> struct LogNeoHookean {
  template <class T>
  static T psi(const Eigen::Matrix<T, Dim, Dim> &C, T mu, T lam) {
    T trC = C.trace();
    T J = sqrt(C.determinant());
    T lj = log(J);
    return mu / T(2) * (trC - T(Dim)) - mu * lj + lam / T(2) * lj * lj;
  }
};

// ============ 单元几何特化 ============
template <typename Scalar, typename UnitData> struct GeomTraits;

// 四面体特化（Dim=3, k=6：exx eyy ezz exy exz eyz）
template <typename Scalar> struct GeomTraits<Scalar, TetrahedronFEMData<Scalar>> {
  static constexpr int Dim = 3;
  static constexpr int Verts = 4;
  static constexpr int StrainSize = 6;
  using Mat = Eigen::Matrix<Scalar, Dim, Dim>;
  using Vec = Eigen::Matrix<Scalar, Dim, 1>;
  using StrainVec = Eigen::Matrix<Scalar, StrainSize, 1>;
  using StrainJac = Eigen::Matrix<Scalar, StrainSize, Dim * Verts>;

  // 形状矩阵：列 = p_i - p_0
  static Mat shape_matrix(const std::array<Vec, Verts> &p) {
    Mat Dm;
    Dm.col(0) = p[1] - p[0];
    Dm.col(1) = p[2] - p[0];
    Dm.col(2) = p[3] - p[0];
    return Dm;
  }

  // 变形梯度 F = Ds * Dm_inv
  static Mat computeF(const std::array<Vec, Verts> &pos, const Mat &Dm_inv) {
    return shape_matrix(pos) * Dm_inv;
  }

  // 对称 Green 应变 s = C - I 的分量（C = F^T F；能量以 C = 2s + I 接收）
  // 顺序：exx eyy ezz exy exz eyz
  static StrainVec strain(const std::array<Vec, Verts> &pos, const Mat &Dm_inv) {
    Mat F = computeF(pos, Dm_inv);
    StrainVec s;
    int comp = Dim;
    for (int a = 0; a < Dim; ++a) {
      s(a) = F.col(a).squaredNorm() - Scalar(1);
      for (int b = a + 1; b < Dim; ++b)
        s(comp++) = F.col(a).dot(F.col(b));
    }
    return s;
  }

  // 解析 ds/dx，在当前 x 处取值（k x Dim*Verts）
  // dF_ab/dp_{0,d} = -delta_{ad} * sum_j Dm_inv(j,b)
  // dF_ab/dp_{v,d} =  delta_{ad} * Dm_inv(v-1,b)   (v = 1..Dim)
  // ds_aa/dF_cd    = 2 F(c,a) delta_ac
  // ds_ab/dF_cd    = F(c,b) delta_ac + F(c,a) delta_bc (a < b)
  static StrainJac strain_jacobian(const std::array<Vec, Verts> &pos,
                                   const Mat &Dm_inv) {
    StrainJac S;
    Eigen::Matrix<Scalar, Dim * Dim, Dim * Verts> dFdx; // (ab, vd)
    dFdx.setZero();
    for (int a = 0; a < Dim; ++a)
      for (int b = 0; b < Dim; ++b)
        for (int d = 0; d < Dim; ++d) {
          Scalar s0 = Scalar(0);
          for (int j = 0; j < Dim; ++j)
            s0 -= Dm_inv(j, b);
          if (a == d) {
            dFdx(a * Dim + b, 0 * Dim + d) += s0;
            for (int v = 1; v <= Dim; ++v)
              dFdx(a * Dim + b, v * Dim + d) += Dm_inv(v - 1, b);
          }
        }
    Mat F = computeF(pos, Dm_inv);
    Eigen::Matrix<Scalar, StrainSize, Dim * Dim> dsdF;
    dsdF.setZero();
    for (int a = 0; a < Dim; ++a)
      for (int c = 0; c < Dim; ++c)
        dsdF(a, c * Dim + a) = Scalar(2) * F(c, a);
    int comp = Dim;
    for (int a = 0; a < Dim; ++a)
      for (int b = a + 1; b < Dim; ++b) {
        for (int c = 0; c < Dim; ++c) {
          dsdF(comp, c * Dim + a) += F(c, b);
          dsdF(comp, c * Dim + b) += F(c, a);
        }
        ++comp;
      }
    S = dsdF * dFdx;
    return S;
  }
};

// 三角形壳特化（Dim=2, k=3：exx eyy exy；世界 -> 局部由 frame 完成）
template <typename Scalar> struct GeomTraits<Scalar, TriangleFEMData<Scalar>> {
  static constexpr int Dim = 2;
  static constexpr int Verts = 3;
  static constexpr int StrainSize = 3;
  using Mat = Eigen::Matrix<Scalar, Dim, Dim>;
  using Vec = Eigen::Matrix<Scalar, Dim, 1>;
  using StrainVec = Eigen::Matrix<Scalar, StrainSize, 1>;
  using StrainJac = Eigen::Matrix<Scalar, StrainSize, Dim * Verts>;

  static Mat shape_matrix(const std::array<Vec, Verts> &p) {
    Mat Dm;
    Dm.col(0) = p[1] - p[0];
    Dm.col(1) = p[2] - p[0];
    return Dm;
  }

  static Mat computeF(const std::array<Vec, Verts> &pos, const Mat &Dm_inv) {
    return shape_matrix(pos) * Dm_inv;
  }

  static StrainVec strain(const std::array<Vec, Verts> &pos, const Mat &Dm_inv) {
    Mat F = computeF(pos, Dm_inv);
    StrainVec s;
    int comp = Dim;
    for (int a = 0; a < Dim; ++a) {
      s(a) = F.col(a).squaredNorm() - Scalar(1);
      for (int b = a + 1; b < Dim; ++b)
        s(comp++) = F.col(a).dot(F.col(b));
    }
    return s;
  }

  static StrainJac strain_jacobian(const std::array<Vec, Verts> &pos,
                                   const Mat &Dm_inv) {
    StrainJac S;
    Eigen::Matrix<Scalar, Dim * Dim, Dim * Verts> dFdx; // (ab, vd)
    dFdx.setZero();
    for (int a = 0; a < Dim; ++a)
      for (int b = 0; b < Dim; ++b)
        for (int d = 0; d < Dim; ++d) {
          Scalar s0 = Scalar(0);
          for (int j = 0; j < Dim; ++j)
            s0 -= Dm_inv(j, b);
          if (a == d) {
            dFdx(a * Dim + b, 0 * Dim + d) += s0;
            for (int v = 1; v <= Dim; ++v)
              dFdx(a * Dim + b, v * Dim + d) += Dm_inv(v - 1, b);
          }
        }
    Mat F = computeF(pos, Dm_inv);
    Eigen::Matrix<Scalar, StrainSize, Dim * Dim> dsdF;
    dsdF.setZero();
    for (int a = 0; a < Dim; ++a)
      for (int c = 0; c < Dim; ++c)
        dsdF(a, c * Dim + a) = Scalar(2) * F(c, a);
    int comp = Dim;
    for (int a = 0; a < Dim; ++a)
      for (int b = a + 1; b < Dim; ++b) {
        for (int c = 0; c < Dim; ++c) {
          dsdF(comp, c * Dim + a) += F(c, b);
          dsdF(comp, c * Dim + b) += F(c, a);
        }
        ++comp;
      }
    S = dsdF * dFdx;
    return S;
  }
};


// ============ 工厂：从静止顶点位置填充 Dm_inv / rest_volume / frames ============
template <typename Scalar, typename UnitData>
void setup_units(UnitData &units,
                 const std::vector<Eigen::Vector3<Scalar>> &rest_positions) {
  using Elem = std::remove_cvref_t<decltype(*std::declval<UnitData &>().begin())>;
  using T = GeomTraits<Scalar, Elem>;
  constexpr int Dim = T::Dim;
  constexpr int Verts = T::Verts;
  for (auto &unit : units) {
    const size_t n_cells = unit.indices.size() / Verts;
    unit.Dm_inv.reserve(n_cells);
    constexpr bool is_tri = (Dim == 2);
    if constexpr (is_tri)
      unit.frames.reserve(n_cells);

    for (size_t c = 0; c < n_cells; ++c) {
      std::array<Eigen::Vector3<Scalar>, Verts> pw;
      for (int i = 0; i < Verts; ++i)
        pw[i] = rest_positions[unit.indices[c * Verts + i]];

      if constexpr (Dim == 3) {
        std::array<Eigen::Matrix<Scalar, 3, 1>, Verts> p;
        for (int i = 0; i < Verts; ++i)
          p[i] = pw[i];
        typename T::Mat Dm = T::shape_matrix(p);
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
        typename T::Mat Dm = T::shape_matrix(p);
        unit.Dm_inv.push_back(Dm.inverse());
        unit.rest_volume.push_back(std::abs(Dm.determinant()) / Scalar(2));
      }
    }
  }
}
// ============ FP-PXPBD 求解器（GPBD Algorithm 1，Gauss-Seidel 形式） ============
// 目标 (19)：min_{Dl in R^k} 1/2 ||d + W S^T Dl||^2_{W^{-1}} + U_hat(x + W S^T Dl)
// 其中 U_hat(s) = rest_volume * Material::psi(C = 2s + I, mu, lambda_mat)。
// 注：autodiff v1.1.2 与 Eigen 5.0.1 不兼容，Û 的梯度/Hessian 按预案用中心差分
//（k <= 6，代价可忽略）；应变雅可比 S 为解析式。
template <typename Scalar, typename UnitData,
          typename Material =
              StableNeoHookean<Scalar, GeomTraits<Scalar, UnitData>::Dim>>
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

  FPSolver(std::vector<UnitData> &units, int maxIter = 10,
           Scalar over_relax = Scalar(1),
           Vec3<Scalar> gravity = Vec3<Scalar>::Zero())
      : units_(units), maxIter_(maxIter), over_relax_(over_relax),
        gravity_(gravity) {}

  void solve(PhysicsData<Scalar> &data, Scalar dt) override {
    // 1. 预测：备份 pos_prev（速度恢复基准），半隐式欧拉 + 重力
    for (size_t i = 0; i < data.pos.size(); ++i)
      data.pos_prev[i] = data.pos[i];
    integrate_velocity_verlet(data, gravity_, dt);

    // 2. 每单元位移缓存清零
    const size_t n_units = units_.size();
    d_.assign(n_units, {});
    for (size_t u = 0; u < n_units; ++u)
      d_[u].assign(units_[u].indices.size() / Verts, Dvec::Zero());

    // 3. GPBD 迭代：逐单元 Gauss-Seidel
    for (int iter = 0; iter < maxIter_; ++iter)
      for (size_t u = 0; u < n_units; ++u) {
        const size_t n_cells = units_[u].indices.size() / Verts;
        for (size_t c = 0; c < n_cells; ++c)
          process_unit(data, units_[u], c, d_[u][c], dt);
      }

    // 4. 速度恢复
    update_velocities(data, dt);
  }

private:
  // ---- U_hat(s) = V0 * psi(2s + I) ----
  static Scalar elastic_energy(const StrainVec &s, Scalar mu, Scalar lam,
                               Scalar V0) {
    Eigen::Matrix<Scalar, Dim, Dim> C =
        Eigen::Matrix<Scalar, Dim, Dim>::Identity();
    for (int a = 0; a < Dim; ++a)
      C(a, a) += Scalar(2) * s(a);
    int comp = Dim;
    for (int a = 0; a < Dim; ++a)
      for (int b = a + 1; b < Dim; ++b)
        C(a, b) = C(b, a) = s(comp++);
    return V0 * Material::template psi<Scalar>(C, mu, lam);
  }

  // ---- Û 的梯度 / Hessian：中心差分（autodiff 与 Eigen 5 不兼容的预案回退） ----
  static constexpr Scalar FD_EPS = Scalar(1e-4);
  static void elastic_derivs(const StrainVec &s, Scalar mu, Scalar lam,
                             Scalar V0, StrainVec *grad, StrainMat *hess) {
    if (grad) {
      for (int i = 0; i < K; ++i) {
        StrainVec sp = s, sm = s;
        sp(i) += FD_EPS;
        sm(i) -= FD_EPS;
        (*grad)(i) = (elastic_energy(sp, mu, lam, V0) -
                      elastic_energy(sm, mu, lam, V0)) /
                     (Scalar(2) * FD_EPS);
      }
    }
    if (hess) {
      for (int i = 0; i < K; ++i)
        for (int j = 0; j <= i; ++j) {
          StrainVec spp = s, spm = s, smp = s, smm = s;
          spp(i) += FD_EPS; spp(j) += FD_EPS;
          spm(i) += FD_EPS; spm(j) -= FD_EPS;
          smp(i) -= FD_EPS; smp(j) += FD_EPS;
          smm(i) -= FD_EPS; smm(j) -= FD_EPS;
          Scalar v = (elastic_energy(spp, mu, lam, V0) -
                      elastic_energy(spm, mu, lam, V0) -
                      elastic_energy(smp, mu, lam, V0) +
                      elastic_energy(smm, mu, lam, V0)) /
                     (Scalar(4) * FD_EPS * FD_EPS);
          (*hess)(i, j) = v;
          (*hess)(j, i) = v;
        }
    }
  }

  // ---- 倒置修复：SVD 最小奇异值取反 + 质心保持 ----
  // 在局部坐标 pos 上原位修复并返回是否触发。
  static bool fix_inversion(std::array<Vec, Verts> &pos, const Mat &Dm_inv) {
    Mat F = Traits::computeF(pos, Dm_inv);
    if (F.determinant() >= Scalar(0))
      return false;
    Eigen::JacobiSVD<Mat> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat Sigma = svd.singularValues().asDiagonal().toDenseMatrix();
    Sigma(Dim - 1, Dim - 1) = -Sigma(Dim - 1, Dim - 1); // 最小奇异值取反
    Mat F2 = svd.matrixU() * Sigma * svd.matrixV().transpose();
    Mat Ds2 = F2 * Dm_inv.inverse(); // Ds' = F' * Dm
    Vec c0 = Vec::Zero(), cq = Vec::Zero();
    std::array<Vec, Verts> q = pos;
    q[0] = pos[0];
    for (int j = 0; j < Dim; ++j)
      q[j + 1] = pos[0] + Ds2.col(j);
    for (int i = 0; i < Verts; ++i) {
      c0 += pos[i];
      cq += q[i];
    }
    Vec shift = (c0 - cq) / Scalar(Verts);
    for (int i = 0; i < Verts; ++i)
      pos[i] = q[i] + shift;
    return true;
  }

  // ---- 每单元（第 cell 个）GPBD 子问题求解与提交 ----
  void process_unit(PhysicsData<Scalar> &data, UnitData &unit, size_t cell,
                    Dvec &d, Scalar dt) {
    std::array<size_t, Verts> vidx;
    std::array<Vec, Verts> pos;
    Eigen::Matrix<Scalar, 3, 2> frame_t; // 局部 -> 世界（仅三角形用）
    if constexpr (Dim == 3) {
      for (int i = 0; i < Verts; ++i) {
        vidx[i] = unit.indices[cell * Verts + i];
        pos[i] = data.pos[vidx[i]];
      }
    } else {
      const auto &frame = unit.frames[cell];
      frame_t = frame.transpose();
      Eigen::Matrix<Scalar, 2, Verts> loc;
      for (int i = 0; i < Verts; ++i) {
        vidx[i] = unit.indices[cell * Verts + i];
        loc.col(i) = frame * data.pos[vidx[i]];
      }
      for (int i = 0; i < Verts; ++i)
        pos[i] = loc.col(i);
    }

    const Mat &Dm_inv = unit.Dm_inv[cell];
    const Scalar V0 = unit.rest_volume[cell];
    const Scalar mu = unit.mu[cell];
    const Scalar lam = unit.lambda_mat[cell];

    if constexpr (std::is_same_v<Material, LogNeoHookean<Scalar, Dim>>) {
      if (fix_inversion(pos, Dm_inv)) {
        // 修复写回世界坐标（三角形：frame^T；四面体：恒等）
        for (int i = 0; i < Verts; ++i) {
          if constexpr (Dim == 3)
            data.pos[vidx[i]] = pos[i];
          else
            data.pos[vidx[i]] = frame_t * pos[i];
        }
      }
    }

    StrainVec s = Traits::strain(pos, Dm_inv);
    StrainJac S0 = Traits::strain_jacobian(pos, Dm_inv);

    // W = diag(dt^2 * inv_mass)，固定粒子为 0
    Dvec w;
    for (int i = 0; i < Verts; ++i) {
      Scalar im = dt * dt * data.inv_mass[vidx[i]];
      for (int a = 0; a < Dim; ++a)
        w(i * Dim + a) = im;
    }

    // ---- 子问题 (19)：f(Dl) 及其 k 维梯度/Hessian ----
    // y(Dl) = W S0^T Dl；u = d + y；x~ = x + y
    // f   = 1/2 u^T W^{-1} u + U_hat(s(x~))
    // g   = S0 W u + S0 W S(x~)^T grad_s
    // H   = S0 W S0^T + (S0 W S(x~)^T) H_s (S0 W S(x~)^T)^T
    auto eval_f = [&](const StrainVec &Dl, StrainVec *grad,
                      StrainMat *hess) -> Scalar {
      Dvec y = w.asDiagonal() * (S0.transpose() * Dl);
      std::array<Vec, Verts> pt = pos;
      for (int i = 0; i < Verts; ++i)
        pt[i] += y.template segment<Dim>(i * Dim);
      StrainVec s_new = Traits::strain(pt, Dm_inv);

      StrainVec gs;
      StrainMat Hs;
      Scalar U;
      if (grad || hess)
        elastic_derivs(s_new, mu, lam, V0, &gs, hess ? &Hs : nullptr);
      U = elastic_energy(s_new, mu, lam, V0);

      Dvec u = d + y;
      Scalar quad = Scalar(0);
      for (int i = 0; i < N; ++i)
        if (w(i) > Scalar(0))
          quad += u(i) * u(i) / w(i);
      if (grad) {
        StrainJac St = Traits::strain_jacobian(pt, Dm_inv);
        *grad = S0 * w.asDiagonal() * u +
                S0 * w.asDiagonal() * (St.transpose() * gs);
      }
      if (hess) {
        StrainJac St = Traits::strain_jacobian(pt, Dm_inv);
        StrainMat M = S0 * w.asDiagonal() * St.transpose();
        *hess = S0 * w.asDiagonal() * S0.transpose() + M.transpose() * Hs * M;
      }
      (void)U;
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
    Dvec dd = w.asDiagonal() * (S0.transpose() * Dl);
    d += dd;
    for (int i = 0; i < Verts; ++i) {
      if constexpr (Dim == 3)
        data.pos[vidx[i]] += over_relax_ * dd.template segment<Dim>(i * Dim);
      else
        data.pos[vidx[i]] +=
            over_relax_ * (frame_t * dd.template segment<Dim>(i * Dim));
    }
  }

  std::vector<UnitData> &units_;
  int maxIter_;
  Scalar over_relax_;
  Vec3<Scalar> gravity_;
  std::vector<std::vector<Dvec>> d_;
};

} // namespace FP_XPBD

} // namespace xpbd
#endif
