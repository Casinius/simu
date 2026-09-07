// Partial-pivot search for column-major n×n matrix A:
//   pidx[0] = argmax_{i in [k,n)} |A(i,k)|   (first index on ties)
//   pval[0] = |A(pidx,k)|
// bindings: 0=A(SCALAR[n*n] col-major) 1=pidx(int32[2]) 2=pval(SCALAR[1]) 3=params(int32)
// params:   p[0]=n  p[1]=k
// dispatch: single workgroup, local_size 256
#ifdef KP_F64
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#define SCALAR float64_t
#else
#define SCALAR float
#endif

layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer A { SCALAR mata[]; };
layout(std430, binding = 1) writeonly buffer Pidx { int pidx[]; };
layout(std430, binding = 2) writeonly buffer Pval { SCALAR pval[]; };
layout(std430, binding = 3) readonly buffer P { int p[]; };

shared SCALAR shv[256];
shared int shi[256];

void main() {
  const uint tid = gl_LocalInvocationID.x;
  const uint n = uint(p[0]);
  const uint k = uint(p[1]);
  SCALAR bv = SCALAR(-1);
  int bi = int(n);
  for (uint i = k + tid; i < n; i += 256u) {
    SCALAR x = abs(mata[k * n + i]);
    if (x > bv) { bv = x; bi = int(i); }
  }
  shv[tid] = bv;
  shi[tid] = bi;
  barrier();
  memoryBarrierShared();
  for (uint s = 128u; s > 0u; s >>= 1u) {
    if (tid < s && shv[tid + s] > shv[tid]) {
      shv[tid] = shv[tid + s];
      shi[tid] = shi[tid + s];
    }
    barrier();
    memoryBarrierShared();
  }
  if (tid == 0u) {
    pidx[0] = shi[0];
    pval[0] = shv[0];
  }
}
