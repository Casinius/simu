// SPDX-License-Identifier: Apache-2.0
// kpalg::PartialPivLU — a Kompute-backed mirror of Eigen::PartialPivLU.
//
// A = P^-1 * L * U  with partial pivoting, computed in-place on the GPU:
// per pivot column k the algorithm runs four kernels
//   1. PivotSearch  — argmax_{i>=k} |A(i,k)|            (device-side reduction)
//   2. RowSwap      — swaps rows k/p and tracks the permutation sign
//   3. LuScale      — A(i,k) /= A(k,k) for i > k        (the L factor, unit diag)
//   4. LuUpdate     — rank-1 update of the trailing submatrix
//
// The packed n×n buffer holds L (strictly lower part, implicit unit diagonal)
// and U (diagonal and above) exactly like Eigen's matrixLU().
//
// solve() mirrors Eigen's semantics: forward substitution with the unit lower
// factor (P·b already applied via the swaps), then backward substitution with
// U. No singularity check is performed — matching Eigen; use try_solve() for a
// checked variant.
#pragma once

#include "Dense.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>

namespace kpalg {

template <std::floating_point T>
class PartialPivLU final {
 public:
  PartialPivLU() = default;
  explicit PartialPivLU(const GpuMatrix<T>& A) { compute(A); }

  /// (Re)factor A; mirrors `lu.compute(J)` — safe to call repeatedly.
  PartialPivLU& compute(const GpuMatrix<T>& A) {
    if (A.rows() != A.cols()) {
      throw std::invalid_argument("kpalg::PartialPivLU: matrix must be square");
    }
    if (A.rows() == 0) {
      throw std::invalid_argument("kpalg::PartialPivLU: empty matrix");
    }
    n_ = A.rows();
    lu_ = A.clone();  // deep copy: the factorization destroys the input layout
    meta_ = Context::instance().tensor<std::int32_t>({0, 1});  // [0]=pivot, [1]=sign
    pval_ = Context::instance().tensor<T>({T{0}});
    sign_ = 1;

    // The whole factorization is recorded as ONE submission: 4 kernels per
    // pivot column with explicit memory barriers between them (dispatch_seq).
    // This avoids 4n driver round-trips and the read-after-write races that
    // Kompute's plain dispatch sequence would otherwise contain.
    auto& ctx = Context::instance();
    std::vector<Context::Step> steps;
    steps.reserve(4 * n_);
    for (std::size_t k = 0; k < n_; ++k) {
      // 1. pivot search (single workgroup; writes pidx[0] on the device)
      steps.push_back({spv::Kernel::PivotSearch, {lu_.dev(), meta_, pval_},
                       {1, 1, 1}, {i32(n_), i32(k)}});

      // 2. row swap across all columns (also flips the sign flag on device)
      steps.push_back({spv::Kernel::RowSwap, {lu_.dev(), meta_},
                       {u32((n_ + 127) / 128), 1, 1},
                       {i32(n_), i32(n_), i32(k)}});

      // 3. scale column k below the diagonal
      steps.push_back({spv::Kernel::LuScale, {lu_.dev()},
                       {u32((n_ + 127) / 128), 1, 1}, {i32(n_), i32(k)}});

      // 4. rank-1 update of the trailing submatrix
      if (k + 1 < n_) {
        steps.push_back(
            {spv::Kernel::LuUpdate, {lu_.dev()},
             {u32((n_ - k - 1 + 15) / 16), u32((n_ - k - 1 + 15) / 16), 1},
             {i32(n_), i32(k)}});
      }
    }
    ctx.dispatch_seq(steps);
    lu_.mark_device_dirty();

    // read back the sign flag (single int)
    ctx.download({meta_});
    sign_ = (meta_->vector()[1] != 0) ? -1 : 1;
    computed_ = true;
    return *this;
  }

  /// x = A⁻¹ b (Eigen-style, unchecked).
  [[nodiscard]] GpuVector<T> solve(GpuVector<T> b) const {
    ensure_computed();
    if (b.size() != n_) {
      throw std::invalid_argument("kpalg::PartialPivLU::solve: size mismatch");
    }
    substitute_in_place(b);
    return b;
  }

  /// Checked variant: returns nullopt when the solve produced NaN/Inf.
  [[nodiscard]] std::optional<GpuVector<T>> try_solve(GpuVector<T> b) const {
    GpuVector<T> x = solve(std::move(b));
    if (x.array().isNaN().any()) return std::nullopt;
    return x;
  }

  /// determinant = sign * prod(diag(U))
  [[nodiscard]] T determinant() const {
    ensure_computed();
    const auto packed = matrixLU();
    T d{1};
    for (std::size_t i = 0; i < n_; ++i) d *= packed(i, i);
    return static_cast<T>(sign_) * d;
  }

  /// Packed in-place factors: L strictly below the diagonal, U on/above.
  [[nodiscard]] GpuMatrix<T> matrixLU() const {
    ensure_computed();
    return lu_.clone();
  }

  [[nodiscard]] bool is_computed() const noexcept { return computed_; }

 private:
  friend class GpuMatrix<T>;

  void ensure_computed() const {
    if (!computed_) {
      throw std::logic_error("kpalg::PartialPivLU: compute() not called");
    }
  }

  /// Forward/back substitution directly on the device-resident factors.
  /// The 2n dependent steps are recorded as ONE submission (dispatch_seq
  /// inserts the required write->read barriers between steps).
  void substitute_in_place(GpuVector<T>& x) const {
    auto& ctx = Context::instance();
    const auto& m = lu_.dev();
    const auto& xd = x.dev();

    std::vector<Context::Step> steps;
    steps.reserve(2 * n_);
    // L y = P b   (unit diagonal, swaps already applied to the right-hand side
    // by pivoting the factor columns — identical to Eigen's approach)
    for (std::size_t j = 0; j < n_; ++j) {
      steps.push_back({spv::Kernel::TrsvStep, {m, xd, xd}, {1, 1, 1},
                       {i32(n_), i32(j), i32(0), i32(j), 1}});
    }
    // U x = y
    for (std::size_t j = n_; j-- > 0;) {
      steps.push_back({spv::Kernel::TrsvStep, {m, xd, xd}, {1, 1, 1},
                       {i32(n_), i32(j), i32(j + 1), i32(n_), 0}});
    }
    ctx.dispatch_seq(steps);
    x.state_ = GpuVector<T>::SyncState::DeviceDirty;
  }

  [[nodiscard]] std::int32_t i32(std::size_t v) const {
    if (v > static_cast<std::size_t>(INT32_MAX)) {
      throw std::overflow_error("kpalg: size does not fit int32");
    }
    return static_cast<std::int32_t>(v);
  }

  std::size_t n_ = 0;
  GpuMatrix<T> lu_{};
  std::shared_ptr<kp::TensorT<std::int32_t>> meta_;
  std::shared_ptr<kp::TensorT<T>> pval_;
  int sign_ = 1;
  bool computed_ = false;
};

// GpuMatrix::determinant() — declared in Dense.hpp.
template <std::floating_point T>
T GpuMatrix<T>::determinant() const {
  if (rows_ != cols_) {
    throw std::invalid_argument("kpalg::determinant: matrix must be square");
  }
  return PartialPivLU<T>(*this).determinant();
}

}  // namespace kpalg
