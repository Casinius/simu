// Self-validating demo of the Kompute-backed linear-algebra library.
//
// Every numerical result is checked against a plain-CPU reference computed
// here (no Eigen), mimicking exactly the usage patterns of the project's
// solvers (Newton iterations with PartialPivLU reuse, least-squares, etc.).
#include <print>
#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "gpu/GpuLinAlg.hpp"

namespace {

int failures = 0;

void check(bool ok, std::string_view what) {
  if (ok) {
    std::println("  [PASS] {}", what);
  } else {
    ++failures;
    std::println("  [FAIL] {}", what);
  }
}

template <class T>
T rel_err(const std::vector<T>& a, const std::vector<T>& b) {
  T num{0}, den{0};
  for (std::size_t i = 0; i < a.size(); ++i) {
    num = std::max(num, std::abs(a[i] - b[i]));
    den = std::max(den, std::abs(b[i]));
  }
  return den > T{0} ? num / den : num;
}

// ---- CPU reference implementations (plain C++, no Eigen) -------------------

/// Flat column-major -> row-of-rows (for the CPU reference routines).
template <class T>
std::vector<std::vector<T>> to_rows(const std::vector<T>& flat, std::size_t m,
                                    std::size_t n) {
  std::vector<std::vector<T>> r(m, std::vector<T>(n));
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < m; ++i) r[i][j] = flat[j * m + i];
  }
  return r;
}

/// Gaussian elimination with partial pivoting.
template <class T>
std::vector<T> cpu_lu_solve(std::vector<std::vector<T>> A, std::vector<T> b) {
  const std::size_t n = b.size();
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t p = k;
    for (std::size_t i = k + 1; i < n; ++i) {
      if (std::abs(A[i][k]) > std::abs(A[p][k])) p = i;
    }
    std::swap(A[k], A[p]);
    std::swap(b[k], b[p]);
    for (std::size_t i = k + 1; i < n; ++i) {
      const T l = A[i][k] / A[k][k];
      for (std::size_t j = k; j < n; ++j) A[i][j] -= l * A[k][j];
      b[i] -= l * b[k];
    }
  }
  for (std::size_t i = n; i-- > 0;) {
    for (std::size_t j = i + 1; j < n; ++j) b[i] -= A[i][j] * b[j];
    b[i] /= A[i][i];
  }
  return b;
}

template <class T>
T cpu_det(std::vector<std::vector<T>> A) {
  const std::size_t n = A.size();
  T det{1};
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t p = k;
    for (std::size_t i = k + 1; i < n; ++i) {
      if (std::abs(A[i][k]) > std::abs(A[p][k])) p = i;
    }
    if (p != k) { std::swap(A[k], A[p]); det = -det; }
    det *= A[k][k];
    if (A[k][k] == T{0}) return T{0};
    for (std::size_t i = k + 1; i < n; ++i) {
      const T l = A[i][k] / A[k][k];
      for (std::size_t j = k; j < n; ++j) A[i][j] -= l * A[k][j];
    }
  }
  return det;
}

/// Least squares via normal equations + Gaussian elimination (full-rank refs).
template <class T>
std::vector<T> cpu_lstsq(const std::vector<std::vector<T>>& A,
                         const std::vector<T>& b) {
  const std::size_t m = A.size(), n = A[0].size();
  std::vector<std::vector<T>> ata(n, std::vector<T>(n, T{0}));
  std::vector<T> atb(n, T{0});
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      atb[j] += A[i][j] * b[i];
      for (std::size_t k = 0; k < n; ++k) ata[j][k] += A[i][j] * A[i][k];
    }
  }
  return cpu_lu_solve(ata, atb);
}

// ---- test drivers -----------------------------------------------------------

/// Exercises the exact pattern of the project's Newton solvers:
/// factor once (lu.compute), then solve repeatedly as the residual changes.
template <class T>
void test_lu_newton_pattern(std::mt19937_64& rng) {
  using Vec = kpalg::GpuVector<T>;
  using Mat = kpalg::GpuMatrix<T>;

  std::println("\n-- PartialPivLU: BDF2-style Newton reuse ({}-precision) --",
               sizeof(T) == 8 ? "double" : "single");
  const std::size_t n = 40;
  std::uniform_real_distribution<T> dist(T{-1}, T{1});

  Mat J(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) J.set(i, j, dist(rng));
    J.set(i, i, J(i, i) + T{8});  // diagonally dominant
  }
  kpalg::PartialPivLU<T> lu;  // like `Eigen::PartialPivLU<...> lu;`
  lu.compute(J);              // like `lu.compute(J)`

  // "Newton iterations": different residuals, same factor
  T worst = 0;
  for (int it = 0; it < 5; ++it) {
    Vec F(n);
    for (std::size_t i = 0; i < n; ++i) F.set(i, dist(rng));
    auto delta = lu.solve(-F * T{1});  // lu.solve(-F), move-in temporary
    worst = std::max(worst,
                     rel_err(delta.to_vector(),
                             cpu_lu_solve(to_rows(J.to_vector(), n, n),
                                          F.to_vector())));
  }
  check(worst < (sizeof(T) == 8 ? 1e-12 : 1e-4),
        "lu.solve matches CPU Gaussian elimination");

  // determinant
  const T det_gpu = lu.determinant();
  const T det_cpu = cpu_det(to_rows(J.to_vector(), n, n));
  check(std::abs(det_gpu - det_cpu) /
            std::max<T>(T{1}, std::abs(det_cpu)) <
        (sizeof(T) == 8 ? 1e-10 : 1e-3),
        "determinant matches CPU");
}

template <class T>
void test_lu_small_and_ops(std::mt19937_64& rng) {
  using Vec = kpalg::GpuVector<T>;
  using Mat = kpalg::GpuMatrix<T>;
  std::println("\n-- Vector/matrix ops & 3x3 solve ({}-precision) --",
               sizeof(T) == 8 ? "double" : "single");
  std::uniform_real_distribution<T> dist(T{-2}, T{2});

  // NeoHookean-style 3x3 usage: determinant/trace/products
  Mat Ds(3, 3);
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) Ds.set(i, j, dist(rng));
  Mat DmInv = Mat::Identity(3);
  for (std::size_t i = 0; i < 3; ++i) DmInv.set(i, i, T{1} / T{2});

  const T detF = (Ds * DmInv).determinant();
  const T detRef = cpu_det(to_rows((Ds * DmInv).to_vector(), 3, 3));
  check(std::abs(detF - detRef) < 1e-10, "det(Ds*DmInv) matches CPU");

  const Mat FtF = Ds.transpose() * Ds;
  T tr{0};
  for (std::size_t i = 0; i < 3; ++i) tr += FtF(i, i);
  check(std::abs(FtF.trace() - tr) < 1e-12, "trace(F^T F) consistent");

  // vector ops: axpy chains, dot, norm, cwiseAbs, NaN probe
  Vec x = Vec::Unit(6, 2) * T{3};         // 3*e2
  Vec y(6);
  for (std::size_t i = 0; i < 6; ++i) y.set(i, dist(rng));
  const T dot_cpu = [&] {
    T s{0};
    for (std::size_t i = 0; i < 6; ++i) s += x[i] * y[i];
    return s;
  }();
  check(std::abs(dot(x, y) - dot_cpu) < 1e-10,  // ADL finds the hidden friend
        "dot matches CPU");
  check(std::abs(x.cwiseAbs().maxCoeff() - T{3}) < 1e-12, "cwiseAbs().maxCoeff");

  Vec z = (x + y * T{2} - y) / T{2};  // == x/2
  bool ok = true;
  for (std::size_t i = 0; i < 6; ++i) ok = ok && std::abs(z[i] - x[i] / T{2}) < 1e-12;
  check(ok, "vector arithmetic chain (a + 2b - b)/2 == a/2");

  z.set(3, std::numeric_limits<T>::quiet_NaN());
  check(z.array().isNaN().any(), "array().isNaN().any() detects NaN");
  check(!x.array().isNaN().any(), "isNaN false on clean vector");

  // segment/head/tail
  auto head2 = y.head(2);
  check(head2[0] == y[0] && head2[1] == y[1], "head(2) elements");

  // 3x3 solve
  Mat A(3, 3);
  std::vector<std::vector<T>> aref(3, std::vector<T>(3));
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) {
      const T v = dist(rng);
      A.set(i, j, v);
      aref[i][j] = v;
    }
  Vec b(3);
  for (std::size_t i = 0; i < 3; ++i) b.set(i, dist(rng));
  kpalg::PartialPivLU<T> lu3(A);
  auto xg = lu3.solve(std::move(b));
  check(rel_err(xg.to_vector(), cpu_lu_solve(aref, b.to_vector())) < 1e-10,
        "3x3 PartialPivLU solve matches CPU");
}

template <class T>
void test_svd(std::mt19937_64& rng) {
  using Vec = kpalg::GpuVector<T>;
  using Mat = kpalg::GpuMatrix<T>;
  std::println("\n-- JacobiSVD: least squares, rank, reconstruction ({}-precision) --",
               sizeof(T) == 8 ? "double" : "single");
  std::uniform_real_distribution<T> dist(T{-1}, T{1});
  const std::size_t m = 30, n = 5;

  Mat A(m, n);
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < n; ++j) A.set(i, j, dist(rng));
  // make column 4 = 2*column 1 -> rank 4
  for (std::size_t i = 0; i < m; ++i) A.set(i, 4, T{2} * A(i, 1));

  const auto svd = kpalg::JacobiSVD<T>(
      A, kpalg::SvdOptions::ComputeThinU | kpalg::SvdOptions::ComputeThinV);

  // descending singular values
  const auto s = svd.singularValues().to_vector();
  bool sorted = std::ranges::is_sorted(s, std::greater<T>{});
  check(sorted && s[0] > T{0}, "singular values sorted descending");

  // rank must be 4 (column 4 is dependent)
  check(svd.rank() == 4, "rank() == 4 for rank-deficient matrix");

  // reconstruction ‖A - U Σ Vᵀ‖_max / ‖A‖_max
  const Mat U = svd.matrixU();
  const Mat V = svd.matrixV();
  Mat R(m, n);
  T amax{0}, rmax{0};
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      T v{0};
      for (std::size_t k = 0; k < n; ++k) v += U(i, k) * s[k] * V(j, k);
      R.set(i, j, v);
      amax = std::max(amax, std::abs(A(i, j)));
      rmax = std::max(rmax, std::abs(R(i, j) - A(i, j)));
    }
  }
  check(rmax / amax < (sizeof(T) == 8 ? 1e-10 : 1e-4), "A ≈ U Σ Vᵀ reconstruction");

  // orthonormality of V: ‖VᵀV - I‖
  T verr{0};
  const Mat VtV = V.transpose() * V;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      verr = std::max(verr, std::abs(VtV(i, j) - (i == j ? T{1} : T{0})));
  check(verr < (sizeof(T) == 8 ? 1e-10 : 1e-4), "V has orthonormal columns");

  // least squares on a full-rank tall matrix vs normal equations
  Mat B(30, 4);
  for (std::size_t i = 0; i < 30; ++i)
    for (std::size_t j = 0; j < 4; ++j) B.set(i, j, dist(rng));
  Vec b(30);
  for (std::size_t i = 0; i < 30; ++i) b.set(i, dist(rng));

  const auto svd2 = kpalg::JacobiSVD<T>(
      B, kpalg::SvdOptions::ComputeThinU | kpalg::SvdOptions::ComputeThinV);
  auto xg = svd2.solve(b);
  check(rel_err(xg.to_vector(),
                cpu_lstsq(to_rows(B.to_vector(), 30, 4), b.to_vector())) <
            (sizeof(T) == 8 ? 1e-11 : 1e-4),
        "SVD least-squares matches CPU normal equations");
}

}  // namespace

int main() {
  auto& ctx = kpalg::Context::instance();
  std::println("== kpalg: Eigen-style GPU linear algebra on Kompute ==");
  std::println("device          : {}", ctx.device_name());
  std::println("shaderFloat64   : {}", ctx.has_f64() ? "yes" : "no (f32 fallback only)");
  std::println();

  std::mt19937_64 rng{42};

  try {
    test_lu_small_and_ops<double>(rng);
    test_lu_newton_pattern<double>(rng);
    test_svd<double>(rng);
    if (ctx.has_f64()) {
      // single precision run on the same device (f32 kernels)
      test_lu_small_and_ops<float>(rng);
      test_svd<float>(rng);
    }
  } catch (const std::exception& e) {
    std::println("EXCEPTION: {}", e.what());
    return 1;
  }

  std::println("\n{}", failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
  return failures == 0 ? 0 : 1;
}
