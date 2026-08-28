#ifndef FPXPBD_SOLVER_H
#define FPXPBD_SOLVER_H

#include "xpbd_base.h"
#include <cstdlib>
#include <vector>

namespace xpbd {

namespace FP_XPBD {
// ========== 四面体超弹性 (FP-PXPBD) ==========
template <typename Scalar> struct TetrahedronFEMData {
  std::vector<size_t> indices; // [i0, i1, i2, i3, ...]

  // 核心：参考构型逆矩阵（3x3），直接用 Eigen 矩阵数组
  std::vector<Mat3<Scalar>> Dm_inv;

  // 材料参数（拉梅常数）
  std::vector<Scalar> mu;         // 剪切模量
  std::vector<Scalar> lambda_mat; // 拉梅第一常数（注意不要与拉格朗日乘子重名）

  // 柔度与拉格朗日乘子
  std::vector<Scalar> compliance;
  std::vector<Scalar> lambda;
};

// ========== 三角形超弹性壳 (FP-PXPBD) ==========
template <typename Scalar> struct TriangleFEMData {
  std::vector<size_t> indices; // [i0, i1, i2, ...]

  // 参考构型逆矩阵（2x2）
  std::vector<Mat2<Scalar>> Dm_inv;

  // 材料参数
  std::vector<Scalar> mu;
  std::vector<Scalar> lambda_mat;

  // 壳厚度（用于计算体积）
  std::vector<Scalar> thickness;

  // 柔度与拉格朗日乘子
  std::vector<Scalar> compliance;
  std::vector<Scalar> lambda;
};

// ============ 材料模型基类 ============
template <typename Scalar, int Dim>
class MaterialModel {
public:
    using Mat = Eigen::Matrix<Scalar, Dim, Dim>;
    virtual Scalar psi(const Mat& F, Scalar mu, Scalar lambda) const = 0;
    virtual ~MaterialModel() = default;
};

// Neo‑Hookean 实现 (Dim=2 或 3)
template <typename Scalar, int Dim>
class NeoHookean : public MaterialModel<Scalar, Dim> {
public:
    using Mat = typename MaterialModel<Scalar, Dim>::Mat;
    Scalar psi(const Mat& F, Scalar mu, Scalar lambda) const override {
        Scalar J = F.determinant();
        Scalar I1 = (F.transpose() * F).trace();
        if constexpr (Dim == 3) {
            return 0.5 * mu * (I1 - 3) - mu * std::log(J) + 0.5 * lambda * std::log(J) * std::log(J);
        } else { // 平面应力简化
            return 0.5 * mu * (I1 - 2) - mu * std::log(J) + 0.5 * lambda * std::log(J) * std::log(J);
        }
    }
};

// ============ 单元几何特化 ============
template <typename Scalar, typename UnitData>
struct GeomTraits;

// 四面体特化
template <typename Scalar>
struct GeomTraits<Scalar, FP_XPBD::TetrahedronFEMData<Scalar>> {
    static constexpr int Dim = 3;
    static constexpr int Verts = 4;
    using Mat = Eigen::Matrix<Scalar, Dim, Dim>;
    using Vec = Eigen::Matrix<Scalar, Dim, 1>;

    // 计算变形梯度 F = Ds * Dm_inv
    static Mat computeF(const std::array<Vec, Verts>& pos, const Mat& Dm_inv) {
        Mat Ds;
        Ds.col(0) = pos[1] - pos[0];
        Ds.col(1) = pos[2] - pos[0];
        Ds.col(2) = pos[3] - pos[0];
        return Ds * Dm_inv;
    }

    // 参考体积（为简化直接返回1，实际可存储）
    static Scalar volume(const Mat& Dm_inv) { return 1.0; }
};

// 三角形壳特化
template <typename Scalar>
struct GeomTraits<Scalar, FP_XPBD::TriangleFEMData<Scalar>> {
    static constexpr int Dim = 2;
    static constexpr int Verts = 3;
    using Mat = Eigen::Matrix<Scalar, Dim, Dim>;
    using Vec = Eigen::Matrix<Scalar, Dim, 1>;

    static Mat computeF(const std::array<Vec, Verts>& pos, const Mat& Dm_inv) {
        Mat Ds;
        Ds.col(0) = pos[1] - pos[0];
        Ds.col(1) = pos[2] - pos[0];
        return Ds * Dm_inv;
    }

    static Scalar volume(const Mat& Dm_inv) { return 1.0; } // 可乘厚度
};

// ============ 能量计算器（使用 AutoDiff） ============
template <typename Scalar, typename UnitData, typename Material>
class EnergyFunctor {
public:
    using Traits = GeomTraits<Scalar, UnitData>;
    static constexpr int Dim = Traits::Dim;
    static constexpr int Verts = Traits::Verts;
    using Vec = typename Traits::Vec;
    using Mat = typename Traits::Mat;

    // 输入：顶点坐标数组（普通 Scalar），输出：能量值（Scalar）及其梯度和海森
    struct Result {
        Scalar energy;
        Eigen::Matrix<Scalar, Dim * Verts, 1> gradient;   // 对顶点位置的梯度
        Eigen::Matrix<Scalar, Dim * Verts, Dim * Verts> hessian; // 海森矩阵
    };

    static Result compute(const std::array<Vec, Verts>& pos,
                          const Mat& Dm_inv,
                          Scalar mu, Scalar lambda,
                          const Material& material) {
        constexpr int NVars = Dim * Verts;
        using ADScalar = Eigen::AutoDiffScalar<Eigen::Matrix<Scalar, NVars, 1>>;
        using ADVec = Eigen::Matrix<ADScalar, Dim, 1>;
        using ADMat = Eigen::Matrix<ADScalar, Dim, Dim>;

        // 1. 构造自变量（每个顶点坐标为一个独立变量）
        std::array<ADVec, Verts> ad_pos;
        int var_idx = 0;
        for (int i = 0; i < Verts; ++i) {
            for (int d = 0; d < Dim; ++d) {
                ADScalar var = pos[i][d];
                var.derivatives().setZero();
                var.derivatives()(var_idx) = 1.0; // 单位偏导数
                ad_pos[i][d] = var;
                ++var_idx;
            }
        }

        // 2. 计算变形梯度 F（自动微分）
        ADMat Ds;
        for (int i = 0; i < Dim; ++i) {
            Ds.col(i) = ad_pos[i+1] - ad_pos[0]; // 注意：四面体i=0..2，三角形i=0..1
        }
        ADMat F = Ds * Dm_inv.template cast<ADScalar>();

        // 3. 计算应变能密度
        ADScalar psi_ad = material.psi(F, mu, lambda);
        // 乘以体积得到总能量
        Scalar V = Traits::volume(Dm_inv);
        ADScalar energy_ad = psi_ad * V;

        // 4. 提取结果
        Result res;
        res.energy = energy_ad.value();
        res.gradient = energy_ad.derivatives();

        // 5. 计算海森矩阵（对梯度再求导）
        // 方法：对每个变量方向扰动一次，用数值差分？但为了利用AutoDiff，可采用嵌套AD。
        // 简便起见，这里使用有限差分近似（二阶精度），但也可使用AutoDiff嵌套。
        // 为了展示AutoDiff，我们提供嵌套方案（简化版，仅示意）。
        // 实际项目中，建议使用两次AD或解析推导。
        // 此处为实现完整功能，我们采用数值海森（中心差分）。
        const Scalar eps = 1e-6;
        Eigen::Matrix<Scalar, NVars, 1> g0 = res.gradient;
        res.hessian.setZero();
        for (int i = 0; i < NVars; ++i) {
            // 正向扰动
            std::array<Vec, Verts> pos_plus = pos;
            pos_plus[i / Dim][i % Dim] += eps;
            Result r_plus = compute_energy_only(pos_plus, Dm_inv, mu, lambda, material);
            // 负向扰动
            std::array<Vec, Verts> pos_minus = pos;
            pos_minus[i / Dim][i % Dim] -= eps;
            Result r_minus = compute_energy_only(pos_minus, Dm_inv, mu, lambda, material);
            res.hessian.col(i) = (r_plus.gradient - r_minus.gradient) / (2 * eps);
        }
        return res;
    }

private:
    // 仅计算能量和梯度（用于海森数值差分）
    static Result compute_energy_only(const std::array<Vec, Verts>& pos,
                                      const Mat& Dm_inv,
                                      Scalar mu, Scalar lambda,
                                      const Material& material) {
        // 直接调用AutoDiff计算（可复用，但为了简化，此处重新实现）
        constexpr int NVars = Dim * Verts;
        using ADScalar = Eigen::AutoDiffScalar<Eigen::Matrix<Scalar, NVars, 1>>;
        using ADVec = Eigen::Matrix<ADScalar, Dim, 1>;
        using ADMat = Eigen::Matrix<ADScalar, Dim, Dim>;

        std::array<ADVec, Verts> ad_pos;
        int var_idx = 0;
        for (int i = 0; i < Verts; ++i) {
            for (int d = 0; d < Dim; ++d) {
                ADScalar var = pos[i][d];
                var.derivatives().setZero();
                var.derivatives()(var_idx) = 1.0;
                ad_pos[i][d] = var;
                ++var_idx;
            }
        }
        ADMat Ds;
        for (int i = 0; i < Dim; ++i) {
            Ds.col(i) = ad_pos[i+1] - ad_pos[0];
        }
        ADMat F = Ds * Dm_inv.template cast<ADScalar>();
        ADScalar psi_ad = material.psi(F, mu, lambda);
        Scalar V = Traits::volume(Dm_inv);
        ADScalar energy_ad = psi_ad * V;
        Result res;
        res.energy = energy_ad.value();
        res.gradient = energy_ad.derivatives();
        return res;
    }
};

// ============ FP-PXPBD 求解器 ============
template <typename Scalar, typename UnitData, typename Material = NeoHookean<Scalar, GeomTraits<Scalar, UnitData>::Dim>>
class FPSolver : public ISolver<PhysicsData<Scalar>, Scalar> {
public:
    using Traits = GeomTraits<Scalar, UnitData>;
    static constexpr int Dim = Traits::Dim;
    static constexpr int Verts = Traits::Verts;
    using Vec = typename Traits::Vec;
    using Mat = typename Traits::Mat;

    FPSolver(std::vector<UnitData>& units,
             std::shared_ptr<Material> material,
             int maxIter = 10)
        : units_(units), material_(material), maxIter_(maxIter) {}

    void solve(PhysicsData<Scalar>& data, Scalar dt) override {
        // 1. 预测位置 (半隐式欧拉)
        for (size_t i = 0; i < data.pos.size(); ++i) {
            if (data.inv_mass[i] == 0) continue;
            data.vel[i] += /* 外力 */ Vec3<Scalar>::Zero() * dt;
            data.pos[i] += data.vel[i] * dt;
            data.pos_prev[i] = data.pos[i]; // 备份用于速度更新
        }

        // 2. XPBD 迭代
        for (int iter = 0; iter < maxIter_; ++iter) {
            for (auto& unit : units_) {
                processUnit(data, unit);
            }
        }

        // 3. 更新速度
        for (size_t i = 0; i < data.pos.size(); ++i) {
            if (data.inv_mass[i] == 0) continue;
            data.vel[i] = (data.pos[i] - data.pos_prev[i]) / dt;
        }
    }

private:
    void processUnit(PhysicsData<Scalar>& data, UnitData& unit) {
        // 收集当前顶点位置
        std::array<Vec, Verts> pos;
        for (int i = 0; i < Verts; ++i) {
            pos[i] = data.pos[unit.indices[i]];
        }

        // 使用 AutoDiff 计算能量、梯度（内力）和海森（刚度）
        using Functor = EnergyFunctor<Scalar, UnitData, Material>;
        auto result = Functor::compute(pos, unit.Dm_inv, unit.mu, unit.lambda_mat, *material_);

        const int NVars = Dim * Verts;
        // 梯度 g = dE/dx (即内力负方向)
        Eigen::Matrix<Scalar, NVars, 1> g = result.gradient;
        // 刚度 K = d2E/dx2
        Eigen::Matrix<Scalar, NVars, NVars> K = result.hessian;

        // 计算逆质量矩阵对角块
        Eigen::Matrix<Scalar, NVars, NVars> Minv = Eigen::Matrix<Scalar, NVars, NVars>::Zero();
        for (int i = 0; i < Verts; ++i) {
            Scalar invm = data.inv_mass[unit.indices[i]];
            for (int d = 0; d < Dim; ++d) {
                Minv(i*Dim + d, i*Dim + d) = invm;
            }
        }

        // XPBD 更新公式（针对标量约束 C = 能量）
        // C = energy, ∇C = g
        Scalar C = result.energy;
        // 计算分母：∇C^T M^{-1} ∇C + α
        Scalar denom = g.transpose() * Minv * g + unit.compliance;
        // 计算 Δλ = (-C - α * λ) / denom
        Scalar delta_lambda = (-C - unit.compliance * unit.lambda) / denom;
        // 更新位置：Δx = M^{-1} ∇C * Δλ
        Eigen::Matrix<Scalar, NVars, 1> dx = Minv * g * delta_lambda;
        for (int i = 0; i < Verts; ++i) {
            for (int d = 0; d < Dim; ++d) {
                data.pos[unit.indices[i]][d] += dx(i*Dim + d);
            }
        }
        // 更新拉格朗日乘子
        unit.lambda += delta_lambda;
    }

    std::vector<UnitData>& units_;
    std::shared_ptr<Material> material_;
    int maxIter_;
};


} // namespace FP_XPBD

} // namespace xpbd
#endif