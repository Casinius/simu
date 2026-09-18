
#include "Eigen/Core"

#include <array>
#include <cstddef>

#ifndef SPAR_SOLVER_H
#define SPAR_SOLVER_H
#include <Eigen/Dense>

#include <vector>

namespace xpbd {
template <class Scalar> struct PhysicsData {
  std::vector<Eigen::Vector3<Scalar>> pos;
  std::vector<Eigen::Vector3<Scalar>> velocity;
  std::vector<Eigen::Vector3<Scalar>> pos_prev;
  std::vector<Scalar> inv_mass;
};


template <class Data, class Scalar> struct ISolver {
  virtual ~ISolver() = default;
  virtual void solve(Data &data, Scalar dt) = 0;
};


// 核心标量类型
template <typename Scalar> using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

template <typename Scalar> using Vec2 = Eigen::Matrix<Scalar, 2, 1>;

template <typename Scalar> using Mat3 = Eigen::Matrix<Scalar, 3, 3>;

template <typename Scalar> using Mat2 = Eigen::Matrix<Scalar, 2, 2>;

// ---------- 积分器（半隐式欧拉） ----------
template <typename Scalar>
void integrate_semi_implicit_euler(PhysicsData<Scalar> &data,
                                   const Vec3<Scalar> &force, Scalar dt) {
  for (size_t i = 0; i < data.pos.size(); ++i) {
    if (data.inv_mass[i] == 0)
      continue;
    data.velocity[i] += force * dt;
    data.pos[i] += data.velocity[i] * dt;
  }
}

// 后处理：用位置修正更新速度
template <typename Scalar>
void update_velocities(PhysicsData<Scalar> &data, Scalar dt) {
  for (size_t i = 0; i < data.pos.size(); ++i) {
    if (data.inv_mass[i] == 0)
      continue;
    data.velocity[i] = (data.pos[i] - data.pos_prev[i]) / dt;
    data.pos_prev[i] = data.pos[i];
  }
}


} // namespace xpbd
#endif
