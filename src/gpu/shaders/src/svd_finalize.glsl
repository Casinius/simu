// One-sided Jacobi SVD finish: per column c of W (m×n, col-major):
//   sigma[c] = ||W(:,c)||,  U(:,c) = W(:,c) / sigma[c]  (zero column -> zeros)
// bindings: 0=W(SCALAR[m*n]) 1=sigma(SCALAR[n] out) 2=U(SCALAR[m*n] out) 3=params(int32)
// params:   p[0]=m  p[1]=n
// dispatch: ceil(n / 128) workgroups, local_size 128
#ifdef KP_F64
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#define SCALAR float64_t
#else
#define SCALAR float
#endif

layout(local_size_x = 128) in;
layout(std430, binding = 0) readonly buffer W { SCALAR w[]; };
layout(std430, binding = 1) writeonly buffer Sigma { SCALAR sigma[]; };
layout(std430, binding = 2) writeonly buffer U { SCALAR u[]; };
layout(std430, binding = 3) readonly buffer P { int p[]; };

void main() {
  const uint c = gl_GlobalInvocationID.x;
  const uint m = uint(p[0]);
  const uint n = uint(p[1]);
  if (c < n) {
    SCALAR s2 = SCALAR(0);
    for (uint r = 0u; r < m; ++r) {
      const SCALAR x = w[c * m + r];
      s2 += x * x;
    }
    const SCALAR sg = sqrt(s2);
    sigma[c] = sg;
    const SCALAR inv = (sg > SCALAR(0)) ? SCALAR(1) / sg : SCALAR(0);
    for (uint r = 0u; r < m; ++r) u[c * m + r] = w[c * m + r] * inv;
  }
}
