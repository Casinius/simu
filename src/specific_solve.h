
// #include "solve_config.h"
#include "Eigen/Core"

#include <array>
#include <cstddef>
#include <unistd.h>

#ifndef SPAR_SOLVER_H
#define SPAR_SOLVER_H
#include <Eigen/Dense>
#include <Eigen/LU>
#include <vector>
/**
 * 抽象求解器基类
 * 所有具体求解器必须实现 solve() 方法
 */
template <typename Scalar>
using State = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <class Scalar, size_t n_element>
struct ring_array : std::array<Scalar, n_element> {
  size_t default_set_position = 0;
  void push_back(Scalar val) {
    this->at(default_set_position) = val;
    if (default_set_position < n_element)
      this->default_set_position += 1;
    else
      this->default_set_position = 0;
  }
};

namespace xpbd {
template <class Scalar> struct PhysicsData {
  std::vector<Eigen::Vector3<Scalar>> pos;
  std::vector<Eigen::Vector3<Scalar>> velocity;
  std::vector<Eigen::Vector3<Scalar>> pos_prev;
  std::vector<Scalar> inv_mass;
};

template <typename Data, class Scalar>
void integrate(Data &data, Scalar dt, const Eigen::Vector3<Scalar> &force);

template <class Data, class Scalar> struct ISolver {
  virtual void solve(Data &data, Scalar dt) = 0;
};

template <typename Data, class Scalar>
class XPBDGaussSeidelSolver : public ISolver<Data, Scalar> {};

template <typename Data, class Scalar>
class XPBDChebyshevSolver : public ISolver<Data, Scalar> {}; // 你关心的加速版
// 核心标量类型
template <typename Scalar>
using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

template <typename Scalar>
using Vec2 = Eigen::Matrix<Scalar, 2, 1>;

template <typename Scalar>
using Mat3 = Eigen::Matrix<Scalar, 3, 3>;

template <typename Scalar>
using Mat2 = Eigen::Matrix<Scalar, 2, 2>;

namespace constraint{

  // ========== 距离约束 ==========
template <typename Scalar>
struct DistanceConstraintData {
    std::vector<size_t> indices;        // [i0, i1, ...]
    std::vector<Scalar> rest_length;
    std::vector<Scalar> compliance;
    std::vector<Scalar> lambda;
};

// ========== 二面角弯曲约束 ==========
template <typename Scalar>
struct BendingConstraintData {
    std::vector<size_t> indices;        // [i0, i1, i2, i3, ...]
    std::vector<Scalar> rest_cos;       // 静止二面角余弦值
    std::vector<Scalar> compliance;
    std::vector<Scalar> lambda;
};

// ========== 三角形膜约束 ==========
template <typename Scalar>
struct MembraneConstraintData {
    std::vector<size_t> indices;        // [i0, i1, i2, ...]
    std::vector<Scalar> rest_length_sq; // 静止边长平方（或存储协方差矩阵）
    std::vector<Scalar> stretch_compliance;
    std::vector<Scalar> shear_compliance;
    std::vector<Scalar> lambda;
};

// ========== 固定/附着约束 ==========
template <typename Scalar>
struct AttachConstraintData {
    std::vector<size_t> particle_indices; // [i0, i1, ...]
    std::vector<Vec3<Scalar>> target_positions; // 世界坐标锚点（Eigen向量数组）
    std::vector<Scalar> compliance;
    std::vector<Scalar> lambda;
};

// ========== 刚性附着约束 ==========
template <typename Scalar>
struct RigidAttachConstraintData {
    std::vector<size_t> indices;        // [i0, i1, ...]
    std::vector<Scalar> rest_distance;
    std::vector<Scalar> compliance;
    std::vector<Scalar> lambda;
};

// ========== 体积守恒约束（标量） ==========
template <typename Scalar>
struct VolumeConstraintData {
    std::vector<size_t> indices;        // [i0, i1, i2, i3, ...]
    std::vector<Scalar> rest_volume;
    std::vector<Scalar> compliance;
    std::vector<Scalar> lambda;
};

// ========== 球体碰撞约束 ==========
template <typename Scalar>
struct SphereCollisionData {
    std::vector<size_t> particle_indices;
    std::vector<Vec3<Scalar>> center;   // 球心 Eigen 向量数组
    std::vector<Scalar> radius;
    std::vector<Scalar> friction;
    std::vector<Scalar> compliance;
    std::vector<Scalar> lambda;         // 可选，如需摩擦持久化
};

// ========== 粒子自碰撞约束 ==========
template <typename Scalar>
struct ParticleCollisionData {
    std::vector<size_t> indices;        // [i0, i1, ...]
    std::vector<Scalar> rest_distance;  // 通常为半径和
    std::vector<Scalar> compliance;
    std::vector<Scalar> lambda;
};


}

namespace complex_constraint{
  // ========== 四面体超弹性 (FP-PXPBD) ==========
template <typename Scalar>
struct TetrahedronFEMData {
    std::vector<size_t> indices;        // [i0, i1, i2, i3, ...]
    
    // 核心：参考构型逆矩阵（3x3），直接用 Eigen 矩阵数组
    std::vector<Mat3<Scalar>> Dm_inv;   
    
    // 材料参数（拉梅常数）
    std::vector<Scalar> mu;             // 剪切模量
    std::vector<Scalar> lambda_mat;     // 拉梅第一常数（注意不要与拉格朗日乘子重名）
    
    // 柔度与拉格朗日乘子
    std::vector<Scalar> compliance;
    std::vector<Scalar> lambda;
};

// ========== 三角形超弹性壳 (FP-PXPBD) ==========
template <typename Scalar>
struct TriangleFEMData {
    std::vector<size_t> indices;        // [i0, i1, i2, ...]
    
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
}

// 为了让代码干净，定义具体的物理精度类型（例如单精度实时）
template<class Scalar>
struct ConstraintRegistry {
    constraint::DistanceConstraintData<Scalar> distances;
    constraint::BendingConstraintData<Scalar> bendings;
    constraint::MembraneConstraintData<Scalar> membranes;
    constraint::AttachConstraintData<Scalar> attachments;
    constraint::RigidAttachConstraintData<Scalar> rigid_attachments;
    constraint::VolumeConstraintData<Scalar> volumes;
    constraint::SphereCollisionData<Scalar> sphere_collisions;
    constraint::ParticleCollisionData<Scalar> particle_collisions;
    
    // FP 专用
    complex_constraint::TetrahedronFEMData<Scalar> tetra_fem;
    complex_constraint::TriangleFEMData<Scalar> triangle_fem;

    void clear() {
        // 对每个成员调用 .clear() 释放数据，但保留容量可选（调用 shrink_to_fit 需谨慎）
        // 或只 clear，不改变 capacity 以复用内存
    }
};




// ---------- 积分器（Verlet 或 Velocity-Verlet） ----------
template <typename Scalar>
void integrate_verlet(PhysicsData<Scalar>& data, const Vec3<Scalar>& force, Scalar dt) {
    const Scalar dt2 = dt * dt;
    for (size_t i = 0; i < data.num_particles(); ++i) {
        if (data.invMass[i] == 0) continue;
        Vec3<Scalar> new_x = data.x[i] + (data.x[i] - data.x_prev[i]) + force * dt2;
        data.x_prev[i] = data.x[i];
        data.x[i] = new_x;
        // 速度可以后续再更新，或者不维护
    }
}

template <typename Scalar>
void integrate_velocity_verlet(PhysicsData<Scalar>& data, const Vec3<Scalar>& force, Scalar dt) {
    // 半隐式欧拉（与 velocity-Verlet 类似）
    for (size_t i = 0; i < data.num_particles(); ++i) {
        if (data.invMass[i] == 0) continue;
        data.v[i] += force * dt;
        data.x[i] += data.v[i] * dt;
    }
}

// 后处理：用位置修正更新速度
template <typename Scalar>
void update_velocities(PhysicsData<Scalar>& data, Scalar dt) {
    for (size_t i = 0; i < data.num_particles(); ++i) {
        if (data.invMass[i] == 0) continue;
        data.v[i] = (data.x[i] - data.x_prev[i]) / dt;
        data.x_prev[i] = data.x[i];
    }
}


namespace cheby {
  // 切比雪夫加速参数计算（基于残差缩减率估计）
template <typename Scalar>
struct ChebyshevParams {
    Scalar rho;   // 谱半径估计
    Scalar sigma; // 缩放因子
};

// 简单版本：固定参数，通常使用经验值，或者根据迭代次数动态调整
template <typename Scalar>
ChebyshevParams<Scalar> compute_chebyshev_params(int iteration, Scalar omega = 1.0) {
    // 这里采用简单的Jacobi松弛参数，切比雪夫加速需要两层参数
    // 更准确的实现请参考论文 "Chebyshev加速的XPBD"
    ChebyshevParams<Scalar> params;
    // 伪代码：迭代开始时rho=1，然后逐步逼近最优
    // 实际使用中常用固定松弛因子
    params.rho = 0.9;   // 例如
    params.sigma = 1.0;
    return params;
}

// ---------- 距离约束的切比雪夫投影 ----------
template <typename Scalar>
void project_distance_chebyshev(
    PhysicsData<Scalar>& data,
    const constraint::DistanceConstraintData<Scalar>& cons,
    Scalar dt,
    const ChebyshevParams<Scalar>& params
) {
    const size_t num = cons.indices.size() / 2;
    #pragma omp parallel for
    for (size_t k = 0; k < num; ++k) {
        const size_t i0 = cons.indices[2*k];
        const size_t i1 = cons.indices[2*k + 1];
        const Scalar w0 = data.invMass[i0];
        const Scalar w1 = data.invMass[i1];
        const Scalar total_inv = w0 + w1;
        if (total_inv == 0) continue;

        Vec3<Scalar> p0 = data.x[i0];
        Vec3<Scalar> p1 = data.x[i1];
        Vec3<Scalar> diff = p0 - p1;
        Scalar dist = diff.norm();
        if (dist < 1e-12) continue;

        Scalar C = dist - cons.rest_length[k];
        Scalar alpha = cons.compliance[k] / (dt * dt);
        Scalar lambda_old = cons.lambda[k];
        Scalar delta_lambda = -(C + alpha * lambda_old) / (total_inv + alpha);
        cons.lambda[k] = lambda_old + delta_lambda;

        Vec3<Scalar> correction = delta_lambda * diff / dist;
        // 切比雪夫加速：对修正量进行加权
        Scalar scale = params.rho;  // 或者更复杂的 sigma 组合
        data.x[i0] += w0 * correction * scale;
        data.x[i1] -= w1 * correction * scale;
    }
}

// ---------- 主求解器（循环调用切比雪夫加速） ----------
template <typename Scalar>
void solve_chebyshev_xpbd(
    PhysicsData<Scalar>& data,
    const ConstraintRegistry<Scalar>& registry,
    Scalar dt,
    size_t num_iterations
) {
    for (size_t iter = 0; iter < num_iterations; ++iter) {
        // 计算当前迭代的切比雪夫参数（可根据迭代次数调整）
        ChebyshevParams<Scalar> params = compute_chebyshev_params<Scalar>(iter);

        // 并行求解各类约束（可按任意顺序，因为都是独立计算修正量）
        if (!registry.distances.indices.empty()) {
            project_distance_chebyshev(data, registry.distances, dt, params);
        }
        if (!registry.bendings.indices.empty()) {
            // project_bending_chebyshev(...); // 类似实现
        }
        if (!registry.membranes.indices.empty()) {
            // project_membrane_chebyshev(...);
        }
        // 其他约束同理
    }
}



}


} // namespace xpbd
#endif
