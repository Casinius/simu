// y = A * x for col-major m×n A:   y(i) = sum_j A(i,j) x(j)
// bindings: 0=A(SCALAR[m*n]) 1=x(SCALAR[n]) 2=y(SCALAR[m] out) 3=params(int32)
// params:   p[0]=m  p[1]=n
// dispatch: ceil(m / 128) workgroups, local_size 128
#ifdef KP_F64
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#define SCALAR float64_t
#else
#define SCALAR float
#endif

layout(local_size_x = 128) in;
layout(std430, binding = 0) readonly buffer A { SCALAR ma[]; };
layout(std430, binding = 1) readonly buffer X { SCALAR x[]; };
layout(std430, binding = 2) writeonly buffer Y { SCALAR y[]; };
layout(std430, binding = 3) readonly buffer P { int p[]; };

void main() {
  const uint i = gl_GlobalInvocationID.x;
  const uint m = uint(p[0]);
  const uint n = uint(p[1]);
  if (i < m) {
    SCALAR s = SCALAR(0);
    for (uint j = 0u; j < n; ++j) s += ma[j * m + i] * x[j];
    y[i] = s;
  }
}
