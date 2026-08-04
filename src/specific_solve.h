
// #include "solve_config.h"
#include "solve.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <ranges>
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
using index_t = size_t;
template <class Scalar> struct Particle {
  Eigen::Vector3d x;      // 当前位置
  Eigen::Vector3d x_prev; // 上一帧位置（用于算速度和 Verlet 预测）
  Scalar invMass;         // 1/m
};
template <class Scalar> struct Constraint {
  index_t id;
  index_t p1, p2; // 粒子索引（距离约束举例）
  Scalar rest_length;
  Scalar compliance; // 柔度 (0 = 完全刚性)
  Scalar lambda;     // 拉格朗日乘子（累计值，每帧需要持久化）
};

} // namespace xpbd
#endif
