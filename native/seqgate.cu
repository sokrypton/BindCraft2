// Gate for soft_seq (seqxform.cuh) against the library's own implementation.
#include "seqxform.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: seqgate <oracle-dir>\n"); return 1; }
  Oracle o(argv[1]);
  Tensor& x = o["in/seq"];
  int N = x.shape[0];
  printf("soft_seq  N=%d  alphabet=%d\n", N, seqx::NAA);
  for (const char* k : {"in/seq", "in/seq_bias", "ct/pseudo"}) o[k].to_device();

  float *soft, *pseudo, *dx;
  CU(cudaMalloc(&soft, (size_t)N * seqx::NAA * 4));
  CU(cudaMalloc(&pseudo, (size_t)N * seqx::NAA * 4));
  CU(cudaMalloc(&dx, (size_t)N * seqx::NAA * 4));
  const float alpha = 2.0f, temp = 0.7f, sw = 0.6f, hw = 0.4f;
  seqx::softSeqK<<<(N + 255) / 256, 256>>>(x.dev, o["in/seq_bias"].dev, alpha,
                                           temp, sw, hw, soft, pseudo, N);
  seqx::softSeqBwdK<<<(N + 255) / 256, 256>>>(o["ct/pseudo"].dev, soft, alpha,
                                              temp, sw, hw, dx, N);
  CU(cudaDeviceSynchronize());
  printf("forward\n");
  report("out/pseudo", compare(fetch(pseudo, (size_t)N * seqx::NAA),
                               o["out/pseudo"].host));
  report("out/soft", compare(fetch(soft, (size_t)N * seqx::NAA),
                             o["out/soft"].host));
  printf("backward\n");
  report("grad/seq", compare(fetch(dx, (size_t)N * seqx::NAA),
                             o["grad/seq"].host));
  return 0;
}
