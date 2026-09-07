// out[0] = max_i |a[i]|  (0 when n == 0)
// bindings: 0=a(SCALAR[n]) 1=out(SCALAR[1]) 2=params(int32)
// params:   p[0]=n
// dispatch: single workgroup, local_size 256
#ifdef KP_F64
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#define SCALAR float64_t
#else
#define SCALAR float
#endif

layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer A { SCALAR a[]; };
layout(std430, binding = 1) writeonly buffer Out { SCALAR outv[]; };
layout(std430, binding = 2) readonly buffer P { int p[]; };

shared SCALAR sh[256];

void main() {
  const uint tid = gl_LocalInvocationID.x;
  const uint n = uint(p[0]);
  SCALAR v = SCALAR(0);
  for (uint i = tid; i < n; i += 256u) {
    SCALAR x = abs(a[i]);
    if (x > v) v = x;
  }
  sh[tid] = v;
  barrier();
  memoryBarrierShared();
  for (uint s = 128u; s > 0u; s >>= 1u) {
    if (tid < s && sh[tid + s] > sh[tid]) sh[tid] = sh[tid + s];
    barrier();
    memoryBarrierShared();
  }
  if (tid == 0u) outv[0] = sh[0];
}
