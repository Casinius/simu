
// #include "solve_config.h"
#include "Eigen/Core"
#include "solve.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <unistd.h>
#include <unordered_map>

#ifndef SPAR_SOLVER_H
#define SPAR_SOLVER_H
#include <Eigen/Dense>
#include <Eigen/LU>
#include <set>
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
template <class Scalar, size_t N> struct Constraint {
  index_t id;
  std::array<index_t, N> particle_indices; // 粒子索引（距离约束举例）
  Scalar rest_length;
  Scalar compliance; // 柔度 (0 = 完全刚性)
  Scalar lambda;     // 拉格朗日乘子（累计值，每帧需要持久化）
};

template <class Scalar>
void integrate_symlect_euler(std::vector<Particle<Scalar>> &particles,
                             const Eigen::Vector3d &accelerate, Scalar dt) {
  for (auto &p : particles) {
    if (p.invMass == 0.0)
      continue;
    p.v += accelerate * dt;
    p.x += p.v * dt;
  }
}

template <class Scalar>
void integrate_verlet(std::vector<Particle<Scalar>> &particles,
                      const Eigen::Vector3d &accelerate, Scalar dt) {
  for (auto &p : particles) {
    if (p.invMass == 0.0)
      continue;
    Eigen::Vector3d accel = accelerate;
    Eigen::Vector3f new_x = p.x + (p.x - p.x_prev) + accel * dt * dt;
    p.x_prev = p.x;
    p.x = new_x;
  }
}

template <class Scalar>
void integrate_velocity_verlet(std::vector<Particle<Scalar>> &particles,
                               const Eigen::Vector3d &gravity, double dt) {
  for (auto &p : particles) {
    if (p.invMass == 0.0)
      continue;
    // 1. 更新位置（用当前速度 + 半个步长的加速度）
    p.x += p.v * dt + 0.5 * gravity * dt * dt;
    // 2. 更新速度（先加半步加速度）
    p.v += 0.5 * gravity * dt;
    // 注意：在 XPBD 中，我们不会在这里再次计算外力加速度，
    // 因为约束校正会改变位置。速度的最终修正放在后处理中。
  }
}

struct ConstraintGraph {
  size_t num_constraints;
  std::vector<std::vector<index_t>> adjacency; // 邻接表
  std::vector<index_t> colors;                 // 结果：每个约束的颜色索引
};

template <class Scalar, size_t N> //
ConstraintGraph
brutal_constraint_graph(const std::vector<Constraint<Scalar, N>> cons) {
  size_t n = cons.size();
  ConstraintGraph graph;
  graph.num_constraints = n;
  graph.adjacency.resize(n);
  std::unordered_map<index_t, std::vector<index_t>> particles_to_cons;

  // 建立粒子->约束列表的下标映射
  for (auto i : std::views::iota(n)) {
    for (auto idx : cons.at(i).particle_indices) {
      particles_to_cons[idx].push_back(i);
    }
  }
  for (auto i : std::views::iota(n)) {
    std::set<index_t> neighbors;
    for (int p_idx : cons[i].particle_indices) {
      for (int j : particles_to_cons[p_idx]) {
        if (j != i)
          neighbors.insert(j);
      }
    }
    graph.adjacency[i].assign(neighbors.begin(), neighbors.end());
  }

  return graph;
}

void greedy_coloring(ConstraintGraph &graph) {
  int n = graph.num_constraints;
  graph.colors.assign(n, -1);
  std::vector<bool> used_color(n, false);
  for (auto i : std::views::iota(n)) {
    for (auto neighbor : graph.adjacency[i]) {
      if (graph.colors[neighbor] != -1) {
        used_color[graph.colors[neighbor]] = true;
      }
    }
    size_t color = 0;
    while (color < n && used_color[color])
      color++;
    graph.colors[i] = color;

    for (auto neighbor : graph.adjacency[i]) {
      if (graph.colors[neighbor] != 1) {
        used_color[graph.colors[neighbor]] = false;
      }
    }
  }
}


std::vector<std::vector<index_t>> buildColorGroups(const ConstraintGraph& graph) {
    size_t num_colors = 0;
    for (auto c : graph.colors) num_colors = std::max(num_colors, c + 1);
    
    std::vector<std::vector<index_t>> groups(num_colors);
    for (auto i = 0; i < graph.num_constraints; ++i) {
        groups[graph.colors[i]].push_back(i);
    }
    return groups;
}



namespace SoA {
template <class Scalar> struct ParticlesSoA {
  std::vector<Scalar> x;
  std::vector<Scalar> y;
  std::vector<Scalar> z;
  std::vector<Scalar> invMass;
};

template <class Scalar>
void soa_integrate_symlect_euler(ParticlesSoA<Scalar> &particles,
                                 const Eigen::Vector3d &accelerate, Scalar dt) {
  for (auto &p : particles) {
    if (p.invMass == 0.0)
      continue;
    p.v += accelerate * dt;
    p.x += p.v * dt;
  }
}

template <class Scalar>
void soa_integrate_verlet(ParticlesSoA<Scalar> &particles,
                          const Eigen::Vector3d &accelerate, Scalar dt) {
  for (auto &p : particles) {
    if (p.invMass == 0.0)
      continue;
    Eigen::Vector3d accel = accelerate;
    Eigen::Vector3f new_x = p.x + (p.x - p.x_prev) + accel * dt * dt;
    p.x_prev = p.x;
    p.x = new_x;
  }
}

} // namespace SoA

} // namespace xpbd
#endif
