// Swaps rows k and pidx[0] of a col-major rows×cols matrix A.
// Also flips the permutation-sign flag: pidx[1] ^= 1 (only when a swap happens).
// bindings: 0=A(SCALAR) 1=pidx(int32[2] read-write) 2=params(int32)
// params:   p[0]=rows  p[1]=cols  p[2]=k
// dispatch: ceil(cols / 128) workgroups, local_size 128
#ifdef KP_F64
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#define SCALAR float64_t
#else
#define SCALAR float
#endif

layout(local_size_x = 128) in;
layout(std430, binding = 0) buffer A { SCALAR mata[]; };
layout(std430, binding = 1) buffer Pidx { int pidx[]; };
layout(std430, binding = 2) readonly buffer P { int p[]; };

void main() {
  const uint c = gl_GlobalInvocationID.x;
  const uint rows = uint(p[0]);
  const uint cols = uint(p[1]);
  const uint k = uint(p[2]);
  if (c < cols) {
    const int piv = pidx[0];
    if (piv != int(k)) {
      const SCALAR t = mata[k * rows + c];
      mata[k * rows + c] = mata[uint(piv) * rows + c];
      mata[uint(piv) * rows + c] = t;
      if (c == 0u) atomicXor(pidx[1], 1);  // odd permutation
    }
  }
}
