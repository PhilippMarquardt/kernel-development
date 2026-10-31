// kdev P5 fixture: PDL hoist, a hot loop with repeated scale decode, loop-invariant math, spills.
extern "C" __global__ void __launch_bounds__(128) pdl_dot(const float* __restrict__ x, const unsigned* __restrict__ w,
                                    const unsigned* __restrict__ sc, float* __restrict__ y, int n, int k) {
  float x0 = __ldg(x + blockIdx.x);              // producer data loaded before the wait (hoist bug)
  asm volatile("griddepcontrol.wait;" ::: "memory");
  float acc = 0.f;
  #pragma unroll 1
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    unsigned s = sc[i >> 3];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {                  // scale decode repeated per j (redundant)
      float lo = __uint_as_float(((s >> (j & 1)) & 0xff) << 23) * (float)(k * 3 + 7);
      float hi = __uint_as_float(((s >> 8) & 0xff) << 23);
      unsigned q = w[i * 4 + j];
      acc += lo * (float)(q & 0xf) + hi * (float)(q >> 4) * x0;
    }
  }
  y[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
extern "C" __global__ void spill_k(float* out, const float* in, int n) {
  float r[96];
  #pragma unroll
  for (int i = 0; i < 96; ++i) r[i] = in[i * n + threadIdx.x];
  float s = 0.f;
  for (int t = 0; t < n; ++t) {
    #pragma unroll
    for (int i = 0; i < 96; ++i) s += r[(i + t) % 96] * r[i];
  }
  out[threadIdx.x] = s;
}
