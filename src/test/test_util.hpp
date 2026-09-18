// 轻量测试工具：断言宏 + 失败计数 + 随机场景构造。
// 无第三方依赖；测试套件通过 register_test 注册，test_main.cpp 统一调度。
#ifndef FPXPBD_TEST_UTIL_H
#define FPXPBD_TEST_UTIL_H

#include "fpxpbd.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdlib>
#include <format>
#include <functional>
#include <print>
#include <random>
#include <source_location>
#include <string>
#include <utility>
#include <vector>

namespace xtest {

// ---- 失败计数与断言 ----
inline int &failure_count() {
  static int count = 0;
  return count;
}

inline bool verbose_enabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("TEST_VERBOSE");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

// 统一失败上报：计数 + 打印（宏体单行转发到这里）
inline void fail(std::source_location loc, std::string &&msg) {
  ++failure_count();
  std::print("  FAIL [{}:{}] {}\n", loc.file_name(), loc.line(), msg);
}

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond))                                                               \
      xtest::fail(std::source_location::current(), std::format("CHECK({})", #cond)); \
  } while (0)

#define CHECK_NEAR(got, want, tol)                                             \
  do {                                                                         \
    double g = static_cast<double>(got);                                       \
    double w = static_cast<double>(want);                                      \
    if (!(std::abs(g - w) <= (tol)))                                           \
      xtest::fail(std::source_location::current(),                              \
                  std::format("CHECK_NEAR({}) got = {:.6e}, want = {:.6e}, "   \
                              "tol = {:.3e}",                                   \
                              #got, g, w, static_cast<double>(tol)));           \
  } while (0)

#define CHECK_MSG(cond, ...)                                                   \
  do {                                                                         \
    if (!(cond))                                                               \
      xtest::fail(std::source_location::current(),                             \
                  std::format(__VA_ARGS__));                                   \
  } while (0)

// ---- 套件注册 ----
struct TestSuite {
  std::string name;
  std::function<void()> fn;
};

inline std::vector<TestSuite> &registry() {
  static std::vector<TestSuite> suites;
  return suites;
}

struct Registrar {
  Registrar(std::string name, std::function<void()> fn) {
    registry().push_back({std::move(name), std::move(fn)});
  }
};

#define XTEST_SUITE(name)                                                      \
  static void xtest_suite_##name();                                            \
  static xtest::Registrar xtest_reg_##name{#name, xtest_suite_##name};         \
  static void xtest_suite_##name()

inline int run_all() {
  int failed_suites = 0;
  for (const auto &suite : registry()) {
    const int before = failure_count();
    std::print("[ RUN  ] {}\n", suite.name);
    suite.fn();
    const int delta = failure_count() - before;
    if (delta == 0)
      std::print("[  OK  ] {}\n", suite.name);
    else {
      std::print("[ FAIL ] {} ({} assertions failed)\n", suite.name, delta);
      ++failed_suites;
    }
  }
  if (failed_suites == 0) {
    std::print("ALL {} SUITES PASSED\n", registry().size());
    return 0;
  }
  std::print("{} of {} SUITES FAILED\n", failed_suites, registry().size());
  return 1;
}

// ---- 随机 / 固定场景构造 ----
using Real = double;

inline std::pair<Real, Real> make_material_constants(Real E = 1e4,
                                                     Real nu = 0.3) {
  return {E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))};
}

// 随机扰动一个顶点数组（逐坐标独立采样）
template <int Dim, int Verts>
inline void perturb(std::array<Eigen::Matrix<Real, Dim, 1>, Verts> &p,
                    Real amplitude, std::mt19937 &rng) {
  std::uniform_real_distribution<Real> dist(-amplitude, amplitude);
  for (auto &vertex : p)
    for (auto &x : vertex)
      x += dist(rng);
}

} // namespace xtest

#endif
