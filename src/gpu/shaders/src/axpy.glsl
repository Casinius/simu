// y = alpha * x + y
// bindings: 0=alpha(SCALAR[1]) 1=x(SCALAR[n]) 2=y(SCALAR[n] in/out) 3=params(int32)
// params:   p[0]=n
// dispatch: ceil(n / 128) workgroups, local_size 128
#ifdef KP_F64
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#define SCALAR float64_t
#else
#define SCALAR float
#endif

layout(local_size_x = 128) in;
layout(std430, binding = 0) readonly buffer Alpha { SCALAR alpha[]; };
layout(std430, binding = 1) readonly buffer X { SCALAR x[]; };
layout(std430, binding = 2) buffer Y { SCALAR y[]; };
layout(std430, binding = 3) readonly buffer P { int p[]; };

void main() {
  const uint i = gl_GlobalInvocationID.x;
  if (i < uint(p[0])) y[i] = alpha[0] * x[i] + y[i];
}
