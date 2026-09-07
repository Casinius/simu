// C = A * B, all col-major: A m×k, B k×n, C m×n
// bindings: 0=A 1=B 2=C(out) 3=params(int32)
// params:   p[0]=m  p[1]=n  p[2]=k
// dispatch: (ceil(n/16), ceil(m/16)) workgroups, local_size 16x16
#ifdef KP_F64
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#define SCALAR float64_t
#else
#define SCALAR float
#endif

layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) readonly buffer A { SCALAR ma[]; };
layout(std430, binding = 1) readonly buffer B { SCALAR mb[]; };
layout(std430, binding = 2) writeonly buffer C { SCALAR mc[]; };
layout(std430, binding = 3) readonly buffer P { int p[]; };

void main() {
  const uint j = gl_GlobalInvocationID.x;  // column of C
  const uint i = gl_GlobalInvocationID.y;  // row of C
  const uint m = uint(p[0]);
  const uint n = uint(p[1]);
  const uint k = uint(p[2]);
  if (i < m && j < n) {
    SCALAR s = SCALAR(0);
    for (uint q = 0u; q < k; ++q) s += ma[q * m + i] * mb[j * k + q];
    mc[j * m + i] = s;
  }
}
