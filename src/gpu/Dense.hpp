// SPDX-License-Identifier: Apache-2.0
// kpalg::GpuVector / kpalg::GpuMatrix — a Kompute-backed mirror of the Eigen
// API subset used by this project (dynamic vectors/matrices, coefficient
// access, vector arithmetic, dot/norm, mat-vec / mat-mat products, col access,
// transpose, Zero/Identity factories, `array().isNaN().any()` chains, ...).
//
// Storage model: dense column-major (like Eigen's default). Each object owns
// exactly one Kompute eDevice tensor. Kompute gives every eDevice tensor a
// permanently-mapped host-visible *staging* buffer, which we use as the host
// image of the data:
//
//   host write  -> staging (tensor->setData / data()), state = HostDirty
//   ensure_device -> OpTensorSyncDevice (staging -> primary), state = Synced
//   kernel writes primary       -> state = DeviceDirty
//   ensure_host   -> OpTensorSyncLocal (primary -> staging), state = Synced
//
// so there is never a second mirror vector and never a redundant copy.
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Context.hpp"

namespace kpalg {

template <std::floating_point T>
class GpuMatrix;

template <std::floating_point T>
struct CwiseAbsProxy;
template <std::floating_point T>
struct ArrayProxy;

// ---------------------------------------------------------------------------
// GpuVector
// ---------------------------------------------------------------------------
template <std::floating_point T>
class GpuVector final {
 public:
  using value_type = T;

  GpuVector() = default;
  GpuVector(const GpuVector&) = delete;             // tensors are not aliasable
  GpuVector& operator=(const GpuVector&) = delete;
  GpuVector(GpuVector&&) noexcept = default;
  GpuVector& operator=(GpuVector&&) noexcept = default;

  explicit GpuVector(std::size_t n, T fill = T{0}) { init(n, fill); }
  explicit GpuVector(std::vector<T> values) { init_from(std::move(values)); }
  GpuVector(std::initializer_list<T> values) { init_from(std::vector<T>(values)); }

  [[nodiscard]] std::size_t size() const noexcept { return n_; }
  [[nodiscard]] bool empty() const noexcept { return n_ == 0; }

  /// Bounds-checked coefficient read.
  [[nodiscard]] T operator[](std::size_t i) const {
    check_host();
    if (i >= n_) throw std::out_of_range("kpalg::GpuVector::operator[]");
    ensure_host();
    return dev_->data()[i];
  }
  /// Bounds-checked coefficient write.
  void set(std::size_t i, T value) {
    check_host();
    if (i >= n_) throw std::out_of_range("kpalg::GpuVector::set");
    dev_->data()[i] = value;
    state_ = SyncState::HostDirty;
  }

  /// Copy of the current values (device is brought up to date first).
  [[nodiscard]] std::vector<T> to_vector() const {
    ensure_host();
    return dev_->vector();
  }

  /// Deep copy (independent buffers). The copy is host-authoritative.
  [[nodiscard]] GpuVector clone() const {
    ensure_host();
    GpuVector r;
    r.n_ = n_;
    r.dev_ = Context::instance().tensor<T>(dev_->vector());
    r.state_ = SyncState::HostDirty;
    return r;
  }

  /// Write the whole vector in one shot.
  void write(std::span<const T> values) {
    if (values.size() != n_) throw std::invalid_argument("kpalg::write: size mismatch");
    if (n_ == 0) return;
    dev_->setData({values.begin(), values.end()});
    state_ = SyncState::HostDirty;
  }

  // -- Eigen-like factories --------------------------------------------------
  [[nodiscard]] static GpuVector Zero(std::size_t n) { return GpuVector(n); }
  [[nodiscard]] static GpuVector Constant(std::size_t n, T v) { return GpuVector(n, v); }
  /// e_i — the i-th canonical unit vector.
  [[nodiscard]] static GpuVector Unit(std::size_t n, std::size_t i) {
    GpuVector r(n);
    r.set(i, T{1});
    return r;
  }

  // -- element-wise arithmetic (axpy / scal kernels) --------------------------
  [[nodiscard]] friend GpuVector operator+(const GpuVector& a, const GpuVector& b) {
    check_same_size(a, b);
    GpuVector r = a.clone();
    axpy(b, r, T{1});
    return r;
  }
  [[nodiscard]] friend GpuVector operator-(const GpuVector& a, const GpuVector& b) {
    check_same_size(a, b);
    GpuVector r = a.clone();
    axpy(b, r, T{-1});
    return r;
  }
  [[nodiscard]] friend GpuVector operator-(const GpuVector& a) { return a * T{-1}; }
  [[nodiscard]] friend GpuVector operator*(const GpuVector& a, T alpha) {
    GpuVector r = a.clone();
    scal(r, alpha);
    return r;
  }
  [[nodiscard]] friend GpuVector operator*(T alpha, const GpuVector& a) { return a * alpha; }
  [[nodiscard]] friend GpuVector operator/(const GpuVector& a, T alpha) {
    if (alpha == T{0}) throw std::domain_error("kpalg: division by zero scalar");
    return a * (T{1} / alpha);
  }

  GpuVector& operator+=(const GpuVector& b) {
    check_same_size(*this, b);
    axpy(b, *this, T{1});
    return *this;
  }
  GpuVector& operator-=(const GpuVector& b) {
    check_same_size(*this, b);
    axpy(b, *this, T{-1});
    return *this;
  }
  GpuVector& operator*=(T alpha) {
    scal(*this, alpha);
    return *this;
  }

  /// y += alpha * x
  static void axpy(const GpuVector& x, GpuVector& y, T alpha) {
    check_same_size(x, y);
    if (x.n_ == 0) return;
    auto alpha_t = Context::instance().tensor<T>({alpha});
    x.ensure_device();
    y.ensure_device();
    Context::instance().dispatch(spv::Kernel::Axpy, {alpha_t, x.dev_, y.dev_},
                                 wg128(x.n_), {i32(x.n_)}, {alpha_t}, {});
    y.state_ = SyncState::DeviceDirty;
  }

  static void scal(GpuVector& x, T alpha) {
    if (x.n_ == 0) return;
    auto alpha_t = Context::instance().tensor<T>({alpha});
    x.ensure_device();
    Context::instance().dispatch(spv::Kernel::Scal, {alpha_t, x.dev_},
                                 wg128(x.n_), {i32(x.n_)}, {alpha_t}, {});
    x.state_ = SyncState::DeviceDirty;
  }

  // -- reductions (GPU kernel + scalar readback) ------------------------------
  [[nodiscard]] friend T dot(const GpuVector& a, const GpuVector& b) {
    check_same_size(a, b);
    if (a.n_ == 0) return T{0};
    auto out = Context::instance().tensor<T>({T{0}});
    a.ensure_device();
    b.ensure_device();
    Context::instance().dispatch(spv::Kernel::DotReduce, {a.dev_, b.dev_, out},
                                 {1, 1, 1}, {i32(a.n_)}, {}, {out});
    return out->vector()[0];
  }

  [[nodiscard]] T squaredNorm() const { return dot(*this, *this); }
  [[nodiscard]] T norm() const { return std::sqrt(squaredNorm()); }

  /// Mimics `v.cwiseAbs().maxCoeff()`.
  [[nodiscard]] T cwiseAbs_maxCoeff() const {
    if (n_ == 0) return T{0};
    auto out = Context::instance().tensor<T>({T{0}});
    ensure_device();
    Context::instance().dispatch(spv::Kernel::AbsMaxReduce, {dev_, out},
                                 {1, 1, 1}, {i32(n_)}, {}, {out});
    return out->vector()[0];
  }

  /// Mimics `v.cwiseAbs().maxCoeff()`.
  [[nodiscard]] CwiseAbsProxy<T> cwiseAbs() const noexcept { return CwiseAbsProxy<T>{this}; }
  /// Mimics `v.array().isNaN().any()`.
  [[nodiscard]] ArrayProxy<T> array() const noexcept { return ArrayProxy<T>{this}; }

  // -- slicing (by value; host-side, like small Eigen expressions) ------------
  [[nodiscard]] GpuVector head(std::size_t n) const { return segment(0, n); }
  [[nodiscard]] GpuVector tail(std::size_t n) const {
    if (n > n_) throw std::out_of_range("kpalg::tail: size out of range");
    return segment(n_ - n, n);
  }
  [[nodiscard]] GpuVector segment(std::size_t pos, std::size_t n) const {
    if (pos > n_ || n > n_ - pos) {
      throw std::out_of_range("kpalg::segment: range out of bounds");
    }
    ensure_host();
    const auto src = dev_->vector();
    return GpuVector(std::vector<T>(src.begin() + static_cast<std::ptrdiff_t>(pos),
                                    src.begin() + static_cast<std::ptrdiff_t>(pos + n)));
  }

  void sync_to_device() const { ensure_device(); }
  void sync_to_host() const { ensure_host(); }

 private:
  template <std::floating_point U>
  friend class PartialPivLU;
  template <std::floating_point U>
  friend class JacobiSVD;
  friend class GpuMatrix<T>;
  template <std::floating_point U>
  friend struct CwiseAbsProxy;
  template <std::floating_point U>
  friend struct ArrayProxy;

  enum class SyncState : std::uint8_t { Synced, HostDirty, DeviceDirty };

  void init(std::size_t n, T fill) {
    n_ = n;
    if (n_ == 0) return;
    dev_ = Context::instance().tensor<T>(std::vector<T>(n_, fill));
    state_ = SyncState::HostDirty;  // primary buffer not yet written
  }
  void init_from(std::vector<T> values) {
    n_ = values.size();
    if (n_ == 0) return;
    dev_ = Context::instance().tensor<T>(std::move(values));
    state_ = SyncState::HostDirty;
  }

  [[nodiscard]] const std::shared_ptr<kp::TensorT<T>>& dev() const {
    ensure_device();
    return dev_;
  }

  static void check_same_size(const GpuVector& a, const GpuVector& b) {
    if (a.n_ != b.n_) throw std::invalid_argument("kpalg: operand size mismatch");
  }

  void check_host() const {
    if (!dev_) throw std::logic_error("kpalg: empty vector");
  }
  void ensure_host() const {
    check_host();
    if (state_ == SyncState::DeviceDirty) {
      Context::instance().download({dev_});
      state_ = SyncState::Synced;
    }
  }
  void ensure_device() const {
    check_host();
    if (state_ == SyncState::HostDirty) {
      Context::instance().upload({dev_});
      state_ = SyncState::Synced;
    }
  }

  [[nodiscard]] static kp::Workgroup wg128(std::size_t n) {
    return {u32((n + 127) / 128), 1, 1};
  }
  [[nodiscard]] static std::int32_t i32(std::size_t v) {
    if (v > static_cast<std::size_t>(INT32_MAX)) {
      throw std::overflow_error("kpalg: size does not fit int32");
    }
    return static_cast<std::int32_t>(v);
  }

  std::size_t n_ = 0;
  std::shared_ptr<kp::TensorT<T>> dev_;
  mutable SyncState state_ = SyncState::Synced;
};

template <std::floating_point T>
struct CwiseAbsProxy {
  const GpuVector<T>* v;
  [[nodiscard]] T maxCoeff() const { return v->cwiseAbs_maxCoeff(); }
};

template <std::floating_point T>
struct ArrayProxy {
  const GpuVector<T>* v;
  struct IsNaN {
    const GpuVector<T>* v;
    [[nodiscard]] bool any() const {
      const auto values = v->to_vector();
      return std::ranges::any_of(values, [](T x) { return std::isnan(x); });
    }
  };
  [[nodiscard]] IsNaN isNaN() const noexcept { return IsNaN{v}; }
};

// ---------------------------------------------------------------------------
// GpuMatrix (column-major, like Eigen's default storage order)
// ---------------------------------------------------------------------------
template <std::floating_point T>
class GpuMatrix final {
 public:
  using value_type = T;

  GpuMatrix() = default;
  GpuMatrix(const GpuMatrix&) = delete;             // tensors are not aliasable
  GpuMatrix& operator=(const GpuMatrix&) = delete;
  GpuMatrix(GpuMatrix&&) noexcept = default;
  GpuMatrix& operator=(GpuMatrix&&) noexcept = default;

  GpuMatrix(std::size_t rows, std::size_t cols, T fill = T{0}) { init(rows, cols, fill); }

  [[nodiscard]] static GpuMatrix Zero(std::size_t rows, std::size_t cols) {
    return GpuMatrix(rows, cols);
  }
  [[nodiscard]] static GpuMatrix Zero(std::size_t n) { return GpuMatrix(n, n); }
  [[nodiscard]] static GpuMatrix Identity(std::size_t n) {
    GpuMatrix r(n, n);
    if (n == 0) return r;  // empty: no backing tensor, nothing to fill
    T* p = r.dev_->data();
    for (std::size_t i = 0; i < n; ++i) p[i * n + i] = T{1};
    r.state_ = SyncState::HostDirty;
    return r;
  }

  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
  [[nodiscard]] std::size_t size() const noexcept { return rows_ * cols_; }

  /// Bounds-checked coefficient read.
  [[nodiscard]] T operator()(std::size_t i, std::size_t j) const {
    check_host();
    if (i >= rows_ || j >= cols_) throw std::out_of_range("kpalg::GpuMatrix::operator()");
    ensure_host();
    return dev_->data()[j * rows_ + i];
  }
  /// Bounds-checked coefficient write.
  void set(std::size_t i, std::size_t j, T value) {
    check_host();
    if (i >= rows_ || j >= cols_) throw std::out_of_range("kpalg::GpuMatrix::set");
    ensure_host();
    dev_->data()[j * rows_ + i] = value;
    state_ = SyncState::HostDirty;
  }

  /// Row-major initialization from a flat list (Eigen's comma-init style):
  ///   M.fill_rows({1, 2, 3,
  ///                4, 5, 6});
  void fill_rows(std::initializer_list<T> row_major) {
    check_host();
    if (row_major.size() != rows_ * cols_) {
      throw std::invalid_argument("kpalg::fill_rows: size mismatch");
    }
    T* p = dev_->data();
    for (std::size_t j = 0; j < cols_; ++j) {
      for (std::size_t i = 0; i < rows_; ++i) {
        p[j * rows_ + i] = row_major.begin()[i * cols_ + j];
      }
    }
    state_ = SyncState::HostDirty;
  }

  /// Copy of the current values in column-major order.
  [[nodiscard]] std::vector<T> to_vector() const {
    ensure_host();
    return dev_->vector();
  }

  /// Column proxy supporting `M.col(j) = vec` and `GpuVector v = M.col(j)`.
  class ColRef final {
   public:
    ColRef(GpuMatrix& m, std::size_t j) noexcept : m_(m), j_(j) {}

    ColRef& operator=(const GpuVector<T>& v) {
      if (v.size() != m_.rows_) {
        throw std::invalid_argument("kpalg: column size mismatch");
      }
      m_.ensure_host();
      const auto vals = v.to_vector();
      T* p = m_.dev_->data();
      const std::size_t rows = m_.rows_;
      for (std::size_t i = 0; i < rows; ++i) p[j_ * rows + i] = vals[i];
      m_.state_ = SyncState::HostDirty;
      return *this;
    }
    [[nodiscard]] operator GpuVector<T>() const {
      m_.ensure_host();
      const T* p = m_.dev_->data();
      const std::size_t rows = m_.rows_;
      return GpuVector<T>(std::vector<T>(p + j_ * rows, p + (j_ + 1) * rows));
    }

   private:
    GpuMatrix& m_;
    std::size_t j_;
  };
  [[nodiscard]] ColRef col(std::size_t j) { return ColRef(*this, j); }

  // -- products (GPU kernels) -------------------------------------------------
  /// y = A * x
  [[nodiscard]] GpuVector<T> operator*(const GpuVector<T>& x) const {
    if (cols_ != x.size()) {
      throw std::invalid_argument("kpalg: mat*vec size mismatch");
    }
    GpuVector<T> y(rows_);
    if (rows_ == 0 || cols_ == 0) return y;
    ensure_device();
    auto& xd = x.dev();
    Context::instance().dispatch(
        spv::Kernel::MatVec, {dev_, xd, y.dev_}, {u32((rows_ + 127) / 128), 1, 1},
        {i32(rows_), i32(cols_)}, {}, {});
    y.state_ = GpuVector<T>::SyncState::DeviceDirty;
    return y;
  }

  /// C = A * B
  [[nodiscard]] GpuMatrix operator*(const GpuMatrix& rhs) const {
    if (cols_ != rhs.rows_) {
      throw std::invalid_argument("kpalg: mat*mat size mismatch");
    }
    GpuMatrix r(rows_, rhs.cols_);
    if (r.size() == 0) return r;
    ensure_device();
    auto& rd = rhs.dev();
    Context::instance().dispatch(
        spv::Kernel::MatMul, {dev_, rd, r.dev_},
        {u32((rhs.cols_ + 15) / 16), u32((rows_ + 15) / 16), 1},
        {i32(rows_), i32(rhs.cols_), i32(cols_)}, {}, {});
    r.state_ = SyncState::DeviceDirty;
    return r;
  }

  /// y = A^T * x (GPU)
  [[nodiscard]] GpuVector<T> transpose_times(const GpuVector<T>& x) const {
    if (rows_ != x.size()) {
      throw std::invalid_argument("kpalg: A^T*vec size mismatch");
    }
    GpuVector<T> y(cols_);
    if (rows_ == 0 || cols_ == 0) return y;
    ensure_device();
    auto& xd = x.dev();
    Context::instance().dispatch(
        spv::Kernel::MatVecT, {dev_, xd, y.dev_}, {u32((cols_ + 127) / 128), 1, 1},
        {i32(rows_), i32(cols_)}, {}, {});
    y.state_ = GpuVector<T>::SyncState::DeviceDirty;
    return y;
  }

  [[nodiscard]] GpuMatrix transpose() const {
    ensure_host();
    GpuMatrix r(cols_, rows_);
    const T* p = dev_->data();
    T* q = r.dev_->data();
    for (std::size_t j = 0; j < cols_; ++j) {
      for (std::size_t i = 0; i < rows_; ++i) {
        q[i * cols_ + j] = p[j * rows_ + i];
      }
    }
    r.state_ = SyncState::HostDirty;
    return r;
  }

  /// Deep copy (independent buffers). The copy is host-authoritative.
  [[nodiscard]] GpuMatrix clone() const {
    GpuMatrix r(rows_, cols_);
    if (size() == 0) return r;
    ensure_host();
    r.dev_->setData(dev_->vector());
    r.state_ = SyncState::HostDirty;
    return r;
  }

  [[nodiscard]] T trace() const {
    ensure_host();
    const T* p = dev_->data();
    T s{0};
    const std::size_t d = std::min(rows_, cols_);
    for (std::size_t i = 0; i < d; ++i) s += p[i * rows_ + i];
    return s;
  }

  /// Determinant via GPU PartialPivLU (defined in PartialPivLU.hpp).
  [[nodiscard]] T determinant() const;

  void sync_to_device() const { ensure_device(); }
  void sync_to_host() const { ensure_host(); }

 private:
  template <std::floating_point U>
  friend class PartialPivLU;
  template <std::floating_point U>
  friend class JacobiSVD;

  enum class SyncState : std::uint8_t { Synced, HostDirty, DeviceDirty };

  void init(std::size_t rows, std::size_t cols, T fill) {
    rows_ = rows;
    cols_ = cols;
    if (size() == 0) return;
    dev_ = Context::instance().tensor<T>(std::vector<T>(size(), fill));
    state_ = SyncState::HostDirty;
  }

  [[nodiscard]] const std::shared_ptr<kp::TensorT<T>>& dev() const {
    ensure_device();
    return dev_;
  }

  void mark_device_dirty() noexcept { state_ = SyncState::DeviceDirty; }

  void check_host() const {
    if (!dev_) throw std::logic_error("kpalg: empty matrix");
  }
  void ensure_host() const {
    check_host();
    if (state_ == SyncState::DeviceDirty) {
      Context::instance().download({dev_});
      state_ = SyncState::Synced;
    }
  }
  void ensure_device() const {
    check_host();
    if (state_ == SyncState::HostDirty) {
      Context::instance().upload({dev_});
      state_ = SyncState::Synced;
    }
  }

  [[nodiscard]] static std::int32_t i32(std::size_t v) {
    if (v > static_cast<std::size_t>(INT32_MAX)) {
      throw std::overflow_error("kpalg: size does not fit int32");
    }
    return static_cast<std::int32_t>(v);
  }

  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::shared_ptr<kp::TensorT<T>> dev_;
  mutable SyncState state_ = SyncState::Synced;
};

// Convenience aliases (Eigen naming style).
using GpuVectorXd = GpuVector<double>;
using GpuVectorXf = GpuVector<float>;
using GpuMatrixXd = GpuMatrix<double>;
using GpuMatrixXf = GpuMatrix<float>;

}  // namespace kpalg
