// SPDX-License-Identifier: Apache-2.0
// kpalg — a small Kompute-backed GPU linear-algebra library mirroring the
// Eigen API subset used by this project.
//
//   #include "gpu/GpuLinAlg.hpp"
//   kpalg::GpuMatrixXd A = ..., b = ...;
//   kpalg::PartialPivLU<kpalg::GpuMatrixXd::value_type> lu(A);
//   auto x = lu.solve(b);
//
// Features (all executed as Vulkan compute kernels via Kompute):
//   * GpuVector / GpuMatrix — Eigen-like dense containers (column-major)
//   * PartialPivLU          — LU decomposition with partial pivoting,
//                             compute()/solve()/determinant()/matrixLU()
//   * JacobiSVD             — one-sided Jacobi SVD (thin U/V),
//                             solve() (min-norm least squares), rank(),
//                             singularValues(), matrixU(), matrixV()
//   * matrix/vector products, axpy/scal, dot/norm, cwiseAbs().maxCoeff(),
//     array().isNaN().any(), segment/head/tail, Zero/Unit/Identity factories
#pragma once

#include "SpvCode.hpp"
#include "Context.hpp"
#include "Dense.hpp"
#include "PartialPivLU.hpp"
#include "JacobiSVD.hpp"
