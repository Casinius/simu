// One-sided Jacobi SVD: process a batch of column pairs in parallel.
// Workgroup g handles pair (i, j) = (pairs[2g], pairs[2g+1]):
//   computes dot products of columns i,j of W (m×n, col-major),
//   derives the Jacobi rotation (c, s) and applies it to columns i,j of W and V.
//   metric[g] = |gamma| / (||w_i|| * ||w_j||)  (0 when the pair is orthogonal)
// bindings: 0=W(SCALAR[m*n] in/out) 1=V(SCALAR[n*n] in/out) 2=pairs(int32[2*np])
//           3=metric(SCALAR[np] out) 4=params(int32)
// params:   p[0]=m  p[1]=n  p[2]=np
// dispatch: np workgroups, local_size 256
#ifdef KP_F64
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#define SCALAR float64_t
#define ORTO_TOL SCALAR(1e-14)
#else
#define SCALAR float
#define ORTO_TOL SCALAR(1e-7)
#endif

layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer W { SCALAR w[]; };
layout(std430, binding = 1) buffer V { SCALAR v[]; };
layout(std430, binding = 2) readonly buffer Pairs { int pairs[]; };
layout(std430, binding = 3) writeonly buffer Metric { SCALAR metric[]; };
layout(std430, binding = 4) readonly buffer P { int p[]; };

shared SCALAR sa[256];
shared SCALAR sb[256];
shared SCALAR sg[256];
shared SCALAR cs[2];  // c, s broadcast

void main() {
  const uint tid = gl_LocalInvocationID.x;
  const uint g = gl_WorkGroupID.x;
  const uint m = uint(p[0]);
  const uint n = uint(p[1]);
  const uint i = uint(pairs[2u * g]);
  const uint j = uint(pairs[2u * g + 1u]);

  SCALAR al = SCALAR(0);
  SCALAR be = SCALAR(0);
  SCALAR ga = SCALAR(0);
  for (uint r = tid; r < m; r += 256u) {
    const SCALAR wi = w[i * m + r];
    const SCALAR wj = w[j * m + r];
    al += wi * wi;
    be += wj * wj;
    ga += wi * wj;
  }
  sa[tid] = al; sb[tid] = be; sg[tid] = ga;
  barrier();
  memoryBarrierShared();
  for (uint s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) {
      sa[tid] += sa[tid + s];
      sb[tid] += sb[tid + s];
      sg[tid] += sg[tid + s];
    }
    barrier();
    memoryBarrierShared();
  }

  SCALAR c = SCALAR(1);
  SCALAR s = SCALAR(0);
  if (tid == 0u) {
    const SCALAR gm = abs(sg[0]);
    const SCALAR denom = sqrt(sa[0] * sb[0]);
    if (gm <= ORTO_TOL * denom || sg[0] == SCALAR(0)) {
      metric[g] = SCALAR(0);
    } else {
      const SCALAR zeta = (sb[0] - sa[0]) / (SCALAR(2) * sg[0]);
      const SCALAR t = (zeta >= SCALAR(0) ? SCALAR(1) : SCALAR(-1)) /
                       (abs(zeta) + sqrt(SCALAR(1) + zeta * zeta));
      c = SCALAR(1) / sqrt(SCALAR(1) + t * t);
      s = c * t;
      metric[g] = gm / denom;
    }
    cs[0] = c;
    cs[1] = s;
  }
  barrier();
  memoryBarrierShared();
  c = cs[0];
  s = cs[1];

  for (uint r = tid; r < m; r += 256u) {
    const SCALAR wi = w[i * m + r];
    const SCALAR wj = w[j * m + r];
    w[i * m + r] = c * wi - s * wj;
    w[j * m + r] = s * wi + c * wj;
  }
  const uint vn = uint(p[1]);
  for (uint r = tid; r < vn; r += 256u) {
    const SCALAR vi = v[i * vn + r];
    const SCALAR vj = v[j * vn + r];
    v[i * vn + r] = c * vi - s * vj;
    v[j * vn + r] = s * vi + c * vj;
  }
}
