// Right-looking LU, rank-1 update of the trailing submatrix:
//   A(i,j) -= A(i,k) * A(k,j)   for i,j in (k, n)
// bindings: 0=A(SCALAR[n*n] col-major, in/out) 1=params(int32)
// params:   p[0]=n  p[1]=k
// dispatch: (ceil(n/16), ceil(n/16)) workgroups, local_size 16x16
#ifdef KP_F64
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#define SCALAR float64_t
#else
#define SCALAR float
#endif

layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) buffer A { SCALAR mata[]; };
layout(std430, binding = 1) readonly buffer P { int p[]; };

void main() {
  const uint j = gl_GlobalInvocationID.x;  // column
  const uint i = gl_GlobalInvocationID.y;  // row
  const uint n = uint(p[0]);
  const uint k = uint(p[1]);
  if (i > k && i < n && j > k && j < n) {
    mata[j * n + i] -= mata[k * n + i] * mata[j * n + k];
  }
}
