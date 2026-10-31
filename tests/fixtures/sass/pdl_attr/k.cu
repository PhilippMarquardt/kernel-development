// kdev P5 fixture: pre-wait loads through a by-value Params struct (attention-style prologue).
struct Params {
  const float* q;        // 0x0
  const int* seq_lens;   // 0x8
  const int* page_table; // 0x10
  int n, page;           // 0x18, 0x1c
};
__global__ void attn(Params p) {
  int len = __ldg(&p.seq_lens[threadIdx.x]);
  int pg = p.page_table[len];
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
