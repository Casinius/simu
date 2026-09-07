// SPDX-License-Identifier: Apache-2.0
// kpalg::JacobiSVD — a Kompute-backed mirror of Eigen::JacobiSVD (thin mode).
//
// One-sided Jacobi orthogonalization on the GPU:
//   * W starts as a copy of A (or A^T when A has fewer rows than columns —
//     the same pre-transposition Eigen performs internally)
//   * each round processes a disjoint batch of column pairs in parallel
//     (svd_pairs kernel, one workgroup per pair); rotations are accumulated
//     into V
//   * sweeps repeat until every column pair is orthogonal (metric readback
//     once per sweep) or a sweep cap is hit
//   * svd_finalize normalizes W into U and produces the singular values
//
// A = U * Sigma * V^T with singular values sorted in descending order
// (Eigen's convention).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "Dense.hpp"

namespace kpalg {

/// Mirrors Eigen::ComputeThinU / Eigen::ComputeThinV.
enum class SvdOptions : unsigned {
  None = 0,
  ComputeThinU = 1 << 0,
  ComputeThinV = 1 << 1,
};

[[nodiscard]] constexpr SvdOptions operator|(SvdOptions a, SvdOptions b) noexcept {
  return static_cast<SvdOptions>(static_cast<unsigned>(a) | static_cast<unsigned>(b));
}
[[nodiscard]] constexpr bool has_option(SvdOptions opts, SvdOptions flag) noexcept {
  return (static_cast<unsigned>(opts) & static_cast<unsigned>(flag)) != 0;
}

template <std::floating_point T>
class JacobiSVD final {
 public:
  JacobiSVD() = default;

  explicit JacobiSVD(const GpuMatrix<T>& A,
                     SvdOptions opts = SvdOptions::ComputeThinU |
                                       SvdOptions::ComputeThinV) {
    compute(A, opts);
  }

  JacobiSVD& compute(const GpuMatrix<T>& A, SvdOptions opts) {
    want_u_ = has_option(opts, SvdOptions::ComputeThinU);
    want_v_ = has_option(opts, SvdOptions::ComputeThinV);
    if (!want_u_ && !want_v_) {
      throw std::invalid_argument("kpalg::JacobiSVD: at least one of "
                                  "ComputeThinU/ComputeThinV required");
    }

    // One-sided Jacobi needs a tall-or-square matrix. Eigen pre-transposes
    // wide matrices and swaps the roles of U and V — we do the same.
    transposed_ = A.rows() < A.cols();
    GpuMatrix<T> B = transposed_ ? A.transpose() : A.clone();
    m_ = B.rows();
    n_ = B.cols();
    if (n_ == 0 || m_ == 0) {
      throw std::invalid_argument("kpalg::JacobiSVD: empty matrix");
    }

    auto& ctx = Context::instance();
    w_ = std::move(B);                                  // device copy, destroyed
    v_ = GpuMatrix<T>::Identity(n_);                    // accumulates rotations
    sigma_ = ctx.tensor<T>(std::vector<T>(n_, T{0}));
    u_ = GpuMatrix<T>(m_, n_);

    // metric buffer: one |gamma|/(||w_i||*||w_j||) per pair, read once per sweep
    const std::size_t max_pairs = n_ / 2;
    metric_ = ctx.tensor<T>(std::vector<T>(max_pairs, T{0}));
    pairs_ = ctx.tensor<std::int32_t>(std::vector<std::int32_t>(2 * max_pairs, 0));

    constexpr int kMaxSweeps = 60;
    const T conv_tol = static_cast<T>(std::numeric_limits<T>::epsilon()) *
                       static_cast<T>(1000);
    for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
      T worst{0};
      for (std::uint32_t shift = 1; shift < n_; ++shift) {
        worst = std::max(worst, run_round(shift, conv_tol));
      }
      if (worst <= conv_tol) break;  // all column pairs orthogonal
    }

    // normalize columns of W into U, produce sigma
    ctx.dispatch(spv::Kernel::SvdFinalize, {w_.dev(), sigma_, u_.dev_},
                 {u32((n_ + 127) / 128), 1, 1}, {i32(m_), i32(n_)},
                 {w_.dev()}, {sigma_});
    u_.mark_device_dirty();

    sort_singular_values();

    // map the factors of B back to the SVD of A:
    //   B = A^T = U_B Sigma V_B^T   =>   A = V_B Sigma U_B^T
    u_out_ = transposed_ ? std::move(v_sorted_) : std::move(u_sorted_);
    v_out_ = transposed_ ? std::move(u_sorted_) : std::move(v_sorted_);
    computed_ = true;
    return *this;
  }

  /// Singular values, descending (Eigen convention).
  [[nodiscard]] GpuVector<T> singularValues() const {
    ensure_computed();
    return sigma_host_.clone();
  }

  /// Thin U (m×n). Requires ComputeThinU.
  [[nodiscard]] GpuMatrix<T> matrixU() const {
    ensure_computed();
    if (!want_u_) {
      throw std::logic_error("kpalg::JacobiSVD: matrixU requires ComputeThinU");
    }
    return u_out_.clone();
  }

  /// V (n×n). Requires ComputeThinV.
  [[nodiscard]] GpuMatrix<T> matrixV() const {
    ensure_computed();
    if (!want_v_) {
      throw std::logic_error("kpalg::JacobiSVD: matrixV requires ComputeThinV");
    }
    return v_out_.clone();
  }

  /// Numerical rank: number of singular values above the tolerance.
  [[nodiscard]] int rank() const {
    ensure_computed();
    const T tol = threshold();
    int r = 0;
    for (T s : sigma_host_.to_vector()) {
      if (s > tol) ++r;
    }
    return r;
  }

  /// Eigen-style default tolerance: max(m,n) * eps * sigma_max.
  [[nodiscard]] T threshold() const {
    ensure_computed();
    return static_cast<T>(std::max(m_, n_)) *
           std::numeric_limits<T>::epsilon() * sigma_host_[0];
  }

  /// Minimum-norm least-squares solution of A x = b:
  ///   x = V * Sigma^+ * U^T * b   (tiny singular values are discarded)
  [[nodiscard]] GpuVector<T> solve(const GpuVector<T>& b) const {
    ensure_computed();
    if (!want_u_ || !want_v_) {
      throw std::logic_error("kpalg::JacobiSVD::solve requires ComputeThinU | "
                             "ComputeThinV");
    }
    if (b.size() != m_) {
      throw std::invalid_argument("kpalg::JacobiSVD::solve: size mismatch");
    }
    const auto sigma = sigma_host_.to_vector();
    const T tol = threshold();

    // z = U^T b on device, then scale by 1/sigma on host (n values only)
    GpuVector<T> z = u_out_.transpose_times(b);
    z.sync_to_host();
    GpuVector<T> y(n_);
    for (std::size_t i = 0; i < n_; ++i) {
      y.set(i, sigma[i] > tol ? z[i] / sigma[i] : T{0});
    }
    // x = V * y
    return v_out_ * y;
  }

 private:
  /// Runs one round of `n/2` disjoint pairs for the given cyclic shift.
  /// Returns the largest orthogonality metric of the round.
  [[nodiscard]] T run_round(std::uint32_t shift, T) {
    std::vector<std::int32_t> pairs;
    pairs.reserve(2 * (n_ / 2));
    for (std::uint32_t i = 0; i < n_; ++i) {
      const std::uint32_t j = (i + shift) % n_;
      if (i < j) {
        pairs.push_back(static_cast<std::int32_t>(i));
        pairs.push_back(static_cast<std::int32_t>(j));
      }
    }
    if (pairs.empty()) return T{0};
    const std::size_t np = pairs.size() / 2;

    // write the round's pairs straight into the mapped staging buffer
    {
      std::int32_t* pv = pairs_->data();
      std::copy(pairs.begin(), pairs.end(), pv);
    }
    Context::instance().dispatch(
        spv::Kernel::SvdPairs, {w_.dev(), v_.dev(), pairs_, metric_},
        {u32(np), 1, 1}, {i32(m_), i32(n_), i32(np)},
        {pairs_}, {metric_});

    const auto metrics = metric_->vector();
    return *std::max_element(metrics.begin(),
                             metrics.begin() + static_cast<std::ptrdiff_t>(np));
  }

  /// Sort singular values descending and permute U/V columns accordingly.
  void sort_singular_values() {
    Context::instance().download({sigma_});
    const auto s = sigma_->vector();
    std::vector<std::size_t> order(n_);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return s[a] > s[b]; });

    sigma_host_ = GpuVector<T>(n_);
    for (std::size_t i = 0; i < n_; ++i) sigma_host_.set(i, s[order[i]]);

    if (want_u_) {
      u_.sync_to_host();
      u_sorted_ = permute_columns(u_, order);
    }
    if (want_v_) {
      v_.sync_to_host();
      v_sorted_ = permute_columns(v_, order);
    }
  }

  [[nodiscard]] static GpuMatrix<T> permute_columns(const GpuMatrix<T>& m,
                                                    const std::vector<std::size_t>& order) {
    GpuMatrix<T> r(m.rows(), m.cols());
    const T* src = m.dev_->data();
    T* dst = r.dev_->data();
    for (std::size_t j = 0; j < m.cols(); ++j) {
      const T* col = src + order[j] * m.rows();
      std::copy(col, col + m.rows(), dst + j * m.rows());
    }
    r.mark_device_dirty();
    return r;
  }

  void ensure_computed() const {
    if (!computed_) throw std::logic_error("kpalg::JacobiSVD: compute() not called");
  }
  [[nodiscard]] std::int32_t i32(std::size_t v) const {
    if (v > static_cast<std::size_t>(INT32_MAX)) {
      throw std::overflow_error("kpalg: size does not fit int32");
    }
    return static_cast<std::int32_t>(v);
  }

  std::size_t m_ = 0, n_ = 0;
  bool transposed_ = false;
  bool want_u_ = true, want_v_ = true;
  bool computed_ = false;

  GpuMatrix<T> w_, v_, u_;
  GpuMatrix<T> u_sorted_, v_sorted_;
  GpuMatrix<T> u_out_, v_out_;
  GpuVector<T> sigma_host_{};
  std::shared_ptr<kp::TensorT<T>> sigma_, metric_;
  std::shared_ptr<kp::TensorT<std::int32_t>> pairs_;
};

}  // namespace kpalg
