// One triangular-substitution step for col-major matrix M (n×n):
//   x(j) = (b(j) - sum_{l in [lo,hi)} M(l,j) * x(l)) / (unit ? 1 : M(j,j))
// Run once per j (forward: lo=0, hi=j, unit=1 for packed LU of Eigen-style;
// backward: lo=j+1, hi=n, unit=0).
// bindings: 0=M(SCALAR[n*n]) 1=b(SCALAR[n]) 2=x(SCALAR[n] in/out) 3=params(int32)
// params:   p[0]=n  p[1]=j  p[2]=lo  p[3]=hi  p[4]=unit_diag
// dispatch: single workgroup, local_size 256
#ifdef KP_F64
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#define SCALAR float64_t
#else
#define SCALAR float
#endif

layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer M { SCALAR m[]; };
layout(std430, binding = 1) readonly buffer B { SCALAR b[]; };
layout(std430, binding = 2) buffer X { SCALAR x[]; };
layout(std430, binding = 3) readonly buffer P { int p[]; };

shared SCALAR sh[256];

void main() {
  const uint tid = gl_LocalInvocationID.x;
  const uint n = uint(p[0]);
  const uint j = uint(p[1]);
  const uint lo = uint(p[2]);
  const uint hi = uint(p[3]);
  SCALAR v = SCALAR(0);
  for (uint l = lo + tid; l < hi; l += 256u) {
    v += m[l * n + j] * x[l];  // M(l, j), col-major
  }
  sh[tid] = v;
  barrier();
  memoryBarrierShared();
  for (uint s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) sh[tid] += sh[tid + s];
    barrier();
    memoryBarrierShared();
  }
  if (tid == 0u) {
    const SCALAR diag = (p[4] != 0) ? SCALAR(1) : m[j * n + j];
    x[j] = (b[j] - sh[0]) / diag;
  }
}
