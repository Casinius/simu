

#ifndef CXPBD_SOLVER_H
#define CXPBD_SOLVER_H
#include "xpbd_base.h"
namespace xpbd {
namespace Cheby {

namespace ChebyCons {

// ========== 距离约束 ==========
template <typename Scalar> struct DistanceConstraintData {
  std::vector<size_t> indices; // [i0, i1, ...]
  std::vector<Scalar> rest_length;
  std::vector<Scalar> compliance;
  std::vector<Scalar> lambda;
};

// ========== 二面角弯曲约束 ==========
template <typename Scalar> struct BendingConstraintData {
  std::vector<size_t> indices;  // [i0, i1, i2, i3, ...]
  std::vector<Scalar> rest_cos; // 静止二面角余弦值
  std::vector<Scalar> compliance;
  std::vector<Scalar> lambda;
};

// ========== 三角形膜约束 ==========
template <typename Scalar> struct MembraneConstraintData {
  std::vector<size_t> indices;        // [i0, i1, i2, ...]
  std::vector<Scalar> rest_length_sq; // 静止边长平方（或存储协方差矩阵）
  std::vector<Scalar> stretch_compliance;
  std::vector<Scalar> shear_compliance;
  std::vector<Scalar> lambda;
};

// ========== 固定/附着约束 ==========
template <typename Scalar> struct AttachConstraintData {
  std::vector<size_t> particle_indices;       // [i0, i1, ...]
  std::vector<Vec3<Scalar>> target_positions; // 世界坐标锚点（Eigen向量数组）
  std::vector<Scalar> compliance;
  std::vector<Scalar> lambda;
};

// ========== 刚性附着约束 ==========
template <typename Scalar> struct RigidAttachConstraintData {
  std::vector<size_t> indices; // [i0, i1, ...]
  std::vector<Scalar> rest_distance;
  std::vector<Scalar> compliance;
  std::vector<Scalar> lambda;
};

// ========== 体积守恒约束（标量） ==========
template <typename Scalar> struct VolumeConstraintData {
  std::vector<size_t> indices; // [i0, i1, i2, i3, ...]
  std::vector<Scalar> rest_volume;
  std::vector<Scalar> compliance;
  std::vector<Scalar> lambda;
};

// ========== 球体碰撞约束 ==========
template <typename Scalar> struct SphereCollisionData {
  std::vector<size_t> particle_indices;
  std::vector<Vec3<Scalar>> center; // 球心 Eigen 向量数组
  std::vector<Scalar> radius;
  std::vector<Scalar> friction;
  std::vector<Scalar> compliance;
  std::vector<Scalar> lambda; // 可选，如需摩擦持久化
};

// ========== 粒子自碰撞约束 ==========
template <typename Scalar> struct ParticleCollisionData {
  std::vector<size_t> indices;       // [i0, i1, ...]
  std::vector<Scalar> rest_distance; // 通常为半径和
  std::vector<Scalar> compliance;
  std::vector<Scalar> lambda;
};

// ==================== 通用投影器（使用稀疏有限差分） ====================
template <class Scalar, class ConstraintFunc>
void project_xpbd_fd(
    ConstraintFunc&& compute_constraint,  // 接收 const Eigen::VectorX<Scalar>&，返回 Scalar
    Eigen::VectorX<Scalar>& x,            // 所有粒子位置 (3N)
    const std::vector<Scalar>& masses,    // 每个粒子的质量
    const std::vector<int>& active_dofs,  // 需要求导的自由度索引（如 [3*i0, 3*i0+1, ...]）
    Scalar dt,
    Scalar compliance,
    Scalar& lambda_out)
{
    // 1. 计算当前约束值 C
    Scalar C = compute_constraint(x);

    // 2. 通过中心差分计算梯度（仅针对 active_dofs）
    Eigen::VectorX<Scalar> grad = Eigen::VectorX<Scalar>::Zero(x.size());
    Eigen::VectorX<Scalar> x_orig = x;  // 备份原始位置
    const Scalar eps = 1e-7;

    for (int dof : active_dofs) {
        int p_idx = dof / 3;
        // 固定粒子（质量无穷大）不参与求导
        if (masses[p_idx] == 0) continue;

        // 自适应步长
        Scalar h = eps * std::max(Scalar(1.0), std::abs(x(dof)));

        // C(x + h)
        x(dof) = x_orig(dof) + h;
        Scalar C_plus = compute_constraint(x);

        // C(x - h)
        x(dof) = x_orig(dof) - h;
        Scalar C_minus = compute_constraint(x);

        // 中心差分
        grad(dof) = (C_plus - C_minus) / (2 * h);

        // 恢复位置
        x(dof) = x_orig(dof);
    }

    // 3. 【完全复用】计算 ∇C * M^{-1} * ∇C^T
    Scalar invMassNorm = 0;
    for (int i = 0; i < grad.size() / 3; ++i) {
        Scalar w = (masses[i] > 0) ? Scalar(1) / masses[i] : Scalar(0);
        Eigen::Matrix<Scalar, 3, 1> g = grad.segment<3>(3 * i);
        invMassNorm += w * g.squaredNorm();
    }

    // 4. XPBD 更新公式（与 AutoDiff 版本完全一致）
    Scalar alpha_tilde = compliance / (dt * dt);
    Scalar delta_lambda = -(C + alpha_tilde * lambda_out) / (invMassNorm + alpha_tilde);

    // 5. 更新所有粒子位置（梯度为 0 的粒子自然不动）
    for (int i = 0; i < grad.size() / 3; ++i) {
        Scalar w = (masses[i] > 0) ? Scalar(1) / masses[i] : Scalar(0);
        Eigen::Matrix<Scalar, 3, 1> dx = delta_lambda * w * grad.segment<3>(3 * i);
        x.segment<3>(3 * i) += dx;
    }

    lambda_out += delta_lambda;
}

constexpr std::vector<int> make_dof_list(const std::vector<int>& particle_indices) {
    std::vector<int> dofs;
    dofs.reserve(particle_indices.size() * 3);
    
    for (int p_idx : particle_indices) {
        int base = p_idx * 3;
        dofs.push_back(base);
        dofs.push_back(base + 1);
        dofs.push_back(base + 2);
    }
    
    // 【关键】排序并去重，防止同一个粒子被重复处理（例如自碰撞约束中 i0 == i1）
    // 虽然排序有小开销，但粒子数量极少（通常 ≤ 4），完全可忽略。
    std::sort(dofs.begin(), dofs.end());
    dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
    return dofs;
}


template <class Scalar>
void project_distance(
    Eigen::VectorX<Scalar>& x,
    const std::vector<Scalar>& masses,
    Scalar dt, Scalar compliance, Scalar& lambda_out,
    size_t i0, size_t i1, Scalar rest_length)
{
    // 约束计算函数（接收普通向量，返回标量）
    auto func = [&](const Eigen::VectorX<Scalar>& pos) -> Scalar {
        auto p0 = pos.segment<3>(3 * i0);
        auto p1 = pos.segment<3>(3 * i1);
        return (p0 - p1).norm() - rest_length;
    };

    // 构造活跃自由度列表（仅两个粒子，6 个自由度）
    const auto dofs = make_dof_list({static_cast<int>(i0),static_cast<int>(i1)});

    project_xpbd_fd(func, x, masses, dofs, dt, compliance, lambda_out);
}

// 弯曲约束同理，只需把活跃自由度扩展到 4 个粒子（12 个自由度）
template <class Scalar>
void project_bend(
    Eigen::VectorX<Scalar>& x,
    const std::vector<Scalar>& masses,
    Scalar dt, Scalar compliance, Scalar& lambda_out,
    size_t i0, size_t i1,size_t i2,size_t i3, Scalar rest_cos)
{
    // 约束计算函数（接收普通向量，返回标量）
    auto func = [&](const Eigen::VectorX<Scalar>& pos) -> Scalar {

        auto p0 = pos.segment<3>(3 * i0);
        auto p1 = pos.segment<3>(3 * i1);
        auto p2 = pos.segment<3>(3 * i2);
        auto p3 = pos.segment<3>(3 * i3);

        auto b1 = (p1 - p0).normalized();
        auto b2 = (p2 - p0).normalized();
        auto b3 = (p3 - p0).normalized();
        auto n1 = b1.cross(b2);
        auto n2 = b3.cross(b2);
        return n1.dot(n2) / (n1.norm() * n2.norm() + 1e-12) - rest_cos;
    };

    const auto dofs = make_dof_list({static_cast<int>(i0),static_cast<int>(i1),static_cast<int>(i2),static_cast<int>(i3)});

    project_xpbd_fd(func, x, masses, dofs, dt, compliance, lambda_out);
}
} // namespace ChebyCons

namespace ChebyAutoDiff {

template <class Scalar, class ConstraintFunc>
inline void project_xpbd(
    ConstraintFunc &&compute_constraint,  // 核心：你只需写 C(x) 的计算公式
    Eigen::VectorX<Scalar> &x,            // 所有粒子的位置 (3N x 1)
    const Eigen::VectorX<Scalar> &masses, // 每个粒子的质量
    Scalar dt, Scalar compliance, Scalar &lambda_out) {
  using ADScalar =
      Eigen::AutoDiffScalar<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>;
  using VectorXAD = Eigen::Matrix<ADScalar, Eigen::Dynamic, 1>;
  // 1. 将当前 x 转换为 AutoDiff 类型（用于自动求导）
  VectorXAD x_ad(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    x_ad(i).value() = x(i);
    x_ad(i).derivatives() =
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Unit(x.size(), i);
  }

  // 2. 调用你定义的约束函数，自动计算 C 和梯度 ∇C
  ADScalar C_ad = ConstraintFunc(x_ad);
  Scalar C = C_ad.value();
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> grad = C_ad.derivatives(); // 1 x 3N

  // 3. 【自动组装】计算 ∇C * M^{-1} * ∇C^T （遍历所有粒子）
  Scalar invMassNorm = 0;
  for (int i = 0; i < grad.size() / 3; ++i) {
    Scalar w = (masses.at(i) > 0) ? Scalar(1) / masses.at(i) : Scalar(0);
    Eigen::Matrix<Scalar, 3, 1> g = grad.segment<3>(3 * i);
    invMassNorm += w * g.squaredNorm();
  }

  // 4. 标准的 XPBD 更新公式
  Scalar alpha_tilde = compliance / (dt * dt);
  Scalar delta_lambda =
      -(C + alpha_tilde * lambda_out) / (invMassNorm + alpha_tilde);

  // 5. 【自动更新】根据梯度更新所有粒子位置
  for (int i = 0; i < grad.size() / 3; ++i) {
    Scalar w = (masses.at(i) > 0) ? Scalar(1) / masses.at(i) : Scalar(0);
    Eigen::Matrix<Scalar, 3, 1> dx = delta_lambda * w * grad.segment<3>(3 * i);
    x.segment<3>(3 * i) += dx;
  }

  lambda_out += delta_lambda;
}

template <class Scalar, typename... TA>
inline void
project_distance(Eigen::VectorX<Scalar> &x, // 所有粒子的位置 (3N x 1)
                 const Eigen::VectorX<Scalar> &masses, // 每个粒子的质量
                 Scalar dt, Scalar compliance, Scalar &lambda_out, auto i0,
                 auto i1, Scalar rest_length) {
  project_xpbd(
      [i0, i1, rest_length](const auto &x_ad) -> decltype(auto) {
        auto p0 = x_ad.segment<3>(3 * i0);
        auto p1 = x_ad.segment<3>(3 * i1);
        return (p0 - p1).norm() - rest_length;
      },
      x, masses, dt, compliance, lambda_out);
}

template <class Scalar, typename... TA>
inline void project_bend(Eigen::VectorX<Scalar> &x, // 所有粒子的位置 (3N x 1)
                         const Eigen::VectorX<Scalar> &masses, // 每个粒子的质量
                         Scalar dt, Scalar compliance, Scalar &lambda_out,
                         auto i0, auto i1, auto i2, auto i3, Scalar rest_cos) {
  project_xpbd(
      [i0, i1, i2, i3, rest_cos](const auto &x_ad) -> decltype(auto) {
        auto p0 = x_ad.segment<3>(3 * i0);
        auto p1 = x_ad.segment<3>(3 * i1);
        auto p2 = x_ad.segment<3>(3 * i2);
        auto p3 = x_ad.segment<3>(3 * i3);

        auto b1 = (p1 - p0).normalized();
        auto b2 = (p2 - p0).normalized();
        auto b3 = (p3 - p0).normalized();
        auto n1 = b1.cross(b2);
        auto n2 = b3.cross(b2);
        return n1.dot(n2) / (n1.norm() * n2.norm() + 1e-12) - rest_cos;
      },
      x, masses, dt, compliance, lambda_out);
}

template <class Scalar>
void project_attach(Eigen::VectorX<Scalar> &x,
                    const std::vector<Scalar> &masses, Scalar dt,
                    Scalar compliance, Scalar &lambda_out, size_t idx,
                    const Vec3<Scalar> &target) {
  // 对每个分量分别调用，或者一次投影向量约束（需修改 project_xpbd
  // 支持向量输出） 这里示范三个分量作为三个独立约束（每个分量一个 lambda）
  // 但为了简单，你可以将三个分量合并为距离约束（但那样不会固定方向）
  // 更好的做法：约束为 (p - target).norm() =
  // 0，但会丢失方向，通常附着都用距离=0即可。 这里采用距离=0：
  project_xpbd(
      [idx, target](const auto &x_ad) -> decltype(auto) {
        auto p = x_ad.segment<3>(3 * idx);
        return (p - target).norm(); // 约束 C = |p - target|
      },
      x, masses, dt, compliance, lambda_out);
}

} // namespace ChebyAutoDiff

// 为了让代码干净，定义具体的物理精度类型（例如单精度实时）
template <class Scalar> struct ConstraintRegistry {
  ChebyCons::DistanceConstraintData<Scalar> distances;
  ChebyCons::BendingConstraintData<Scalar> bendings;
  ChebyCons::MembraneConstraintData<Scalar> membranes;
  ChebyCons::AttachConstraintData<Scalar> attachments;
  ChebyCons::RigidAttachConstraintData<Scalar> rigid_attachments;
  ChebyCons::VolumeConstraintData<Scalar> volumes;
  ChebyCons::SphereCollisionData<Scalar> sphere_collisions;
  ChebyCons::ParticleCollisionData<Scalar> particle_collisions;

  void clear() {
    // 对每个成员调用 .clear() 释放数据，但保留容量可选（调用 shrink_to_fit
    // 需谨慎） 或只 clear，不改变 capacity 以复用内存
  }
};

// 切比雪夫加速参数计算（基于残差缩减率估计）
template <typename Scalar> struct ChebyshevParams {
  Scalar rho;   // 谱半径估计
  Scalar sigma; // 缩放因子
};

} // namespace Cheby

} // namespace xpbd
#endif