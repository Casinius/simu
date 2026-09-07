// Right-looking LU, scale step: A(i,k) /= A(k,k) for i in (k, n)
// bindings: 0=A(SCALAR[n*n] col-major, in/out) 1=params(int32)
// params:   p[0]=n  p[1]=k
// dispatch: ceil(n / 128) workgroups, local_size 128
#ifdef KP_F64
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#define SCALAR float64_t
#else
#define SCALAR float
#endif

layout(local_size_x = 128) in;
layout(std430, binding = 0) buffer A { SCALAR mata[]; };
layout(std430, binding = 1) readonly buffer P { int p[]; };

void main() {
  const uint i = gl_GlobalInvocationID.x;
  const uint n = uint(p[0]);
  const uint k = uint(p[1]);
  if (i > k && i < n) mata[k * n + i] /= mata[k * n + k];
}
