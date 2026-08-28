#ifndef FPXPBD_SOLVER_H
#define FPXPBD_SOLVER_H

#include "xpbd_base.h"
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
} // namespace FP_XPBD

} // namespace xpbd
#endif