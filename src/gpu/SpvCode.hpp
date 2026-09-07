// SPDX-License-Identifier: Apache-2.0
// Embedded SPIR-V code for the GPU linear-algebra compute kernels.
//
// The *.comp.spv.h files are generated at build time by xmake's
// `utils.glsl2spv` rule (bin2c mode) from src/gpu/shaders/*.comp
// and must be listed BEFORE the C++ translation units in xmake.lua.
//
// Each kernel exists in two specialisations:
//   * f32 (default)          — works on any Vulkan device
//   * f64 (KP_F64 define)    — requires the shaderFloat64 device feature
#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <vector>

namespace kpalg::spv {

/// Kernels provided by src/gpu/shaders. Order must match `table` below.
enum class Kernel : std::uint8_t {
  DotReduce,
  AbsMaxReduce,
  PivotSearch,
  RowSwap,
  LuScale,
  LuUpdate,
  TrsvStep,
  MatVec,
  MatVecT,
  Axpy,
  Scal,
  MatMul,
  SvdPairs,
  SvdFinalize,
  Count
};

namespace detail {

inline const unsigned char dot_reduce[] = {
#include "dot_reduce.comp.spv.h"
};
inline const unsigned char dot_reduce_f64[] = {
#include "dot_reduce_f64.comp.spv.h"
};
inline const unsigned char absmax_reduce[] = {
#include "absmax_reduce.comp.spv.h"
};
inline const unsigned char absmax_reduce_f64[] = {
#include "absmax_reduce_f64.comp.spv.h"
};
inline const unsigned char pivot_search[] = {
#include "pivot_search.comp.spv.h"
};
inline const unsigned char pivot_search_f64[] = {
#include "pivot_search_f64.comp.spv.h"
};
inline const unsigned char row_swap[] = {
#include "row_swap.comp.spv.h"
};
inline const unsigned char row_swap_f64[] = {
#include "row_swap_f64.comp.spv.h"
};
inline const unsigned char lu_scale[] = {
#include "lu_scale.comp.spv.h"
};
inline const unsigned char lu_scale_f64[] = {
#include "lu_scale_f64.comp.spv.h"
};
inline const unsigned char lu_update[] = {
#include "lu_update.comp.spv.h"
};
inline const unsigned char lu_update_f64[] = {
#include "lu_update_f64.comp.spv.h"
};
inline const unsigned char trsv_step[] = {
#include "trsv_step.comp.spv.h"
};
inline const unsigned char trsv_step_f64[] = {
#include "trsv_step_f64.comp.spv.h"
};
inline const unsigned char matvec[] = {
#include "matvec.comp.spv.h"
};
inline const unsigned char matvec_f64[] = {
#include "matvec_f64.comp.spv.h"
};
inline const unsigned char matvec_t[] = {
#include "matvec_t.comp.spv.h"
};
inline const unsigned char matvec_t_f64[] = {
#include "matvec_t_f64.comp.spv.h"
};
inline const unsigned char axpy[] = {
#include "axpy.comp.spv.h"
};
inline const unsigned char axpy_f64[] = {
#include "axpy_f64.comp.spv.h"
};
inline const unsigned char scal[] = {
#include "scal.comp.spv.h"
};
inline const unsigned char scal_f64[] = {
#include "scal_f64.comp.spv.h"
};
inline const unsigned char matmul[] = {
#include "matmul.comp.spv.h"
};
inline const unsigned char matmul_f64[] = {
#include "matmul_f64.comp.spv.h"
};
inline const unsigned char svd_pairs[] = {
#include "svd_pairs.comp.spv.h"
};
inline const unsigned char svd_pairs_f64[] = {
#include "svd_pairs_f64.comp.spv.h"
};
inline const unsigned char svd_finalize[] = {
#include "svd_finalize.comp.spv.h"
};
inline const unsigned char svd_finalize_f64[] = {
#include "svd_finalize_f64.comp.spv.h"
};

struct Entry {
  const unsigned char* f32;
  std::size_t f32_size;
  const unsigned char* f64;
  std::size_t f64_size;
};

inline constexpr Entry table[] = {
    {dot_reduce, sizeof(dot_reduce), dot_reduce_f64, sizeof(dot_reduce_f64)},
    {absmax_reduce, sizeof(absmax_reduce), absmax_reduce_f64, sizeof(absmax_reduce_f64)},
    {pivot_search, sizeof(pivot_search), pivot_search_f64, sizeof(pivot_search_f64)},
    {row_swap, sizeof(row_swap), row_swap_f64, sizeof(row_swap_f64)},
    {lu_scale, sizeof(lu_scale), lu_scale_f64, sizeof(lu_scale_f64)},
    {lu_update, sizeof(lu_update), lu_update_f64, sizeof(lu_update_f64)},
    {trsv_step, sizeof(trsv_step), trsv_step_f64, sizeof(trsv_step_f64)},
    {matvec, sizeof(matvec), matvec_f64, sizeof(matvec_f64)},
    {matvec_t, sizeof(matvec_t), matvec_t_f64, sizeof(matvec_t_f64)},
    {axpy, sizeof(axpy), axpy_f64, sizeof(axpy_f64)},
    {scal, sizeof(scal), scal_f64, sizeof(scal_f64)},
    {matmul, sizeof(matmul), matmul_f64, sizeof(matmul_f64)},
    {svd_pairs, sizeof(svd_pairs), svd_pairs_f64, sizeof(svd_pairs_f64)},
    {svd_finalize, sizeof(svd_finalize), svd_finalize_f64, sizeof(svd_finalize_f64)},
};

static_assert(std::size(table) == static_cast<std::size_t>(Kernel::Count),
              "Kernel enum and SPIR-V table out of sync");

}  // namespace detail

/// Returns the SPIR-V words for `kernel`, f64 or f32 specialisation.
[[nodiscard]] inline std::vector<std::uint32_t> code(Kernel kernel, bool f64) {
  const auto& e = detail::table[static_cast<std::size_t>(kernel)];
  const unsigned char* bytes = f64 ? e.f64 : e.f32;
  const std::size_t size = f64 ? e.f64_size : e.f32_size;
  assert(size % 4 == 0 && "SPIR-V must be a whole number of 32-bit words");
  std::vector<std::uint32_t> words(size / 4);
  std::memcpy(words.data(), bytes, size);
  return words;
}

}  // namespace kpalg::spv
