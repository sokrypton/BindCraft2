// Gate for one template pair-stack iteration (tplblock.cuh), against the
// oracle's --part pair dump under ORACLE_TEMPLATE -- which is the same five
// modules in the same order that modules.TemplateEmbeddingIteration runs.
#include "tplblock.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: tplblock <oracle-dir> [reps]\n"); return 1; }
  Oracle o(argv[1]);
  int reps = argc > 2 ? atoi(argv[2]) : 10;
  Tensor& pair = o["in/pair"];
  int N = pair.shape[0], C = pair.shape.back();
  if (C != tplblock::CZ) {
    fprintf(stderr, "this is a %d-channel dump; the template stack is %d\n",
            C, tplblock::CZ);
    return 1;
  }
  size_t np = (size_t)N * N * C;
  printf("template pair-stack iteration  N=%d  C=%d\n", N, C);

  pair.to_device();
  Tensor& ct = o["ct/pair"]; ct.to_device();
  Tensor& pm = o["in/pair_mask"]; pm.to_device();

  tplblock::TplBlock b;
  b.alloc(N);
  b.setWeights(o);

  __nv_bfloat16* pairB;
  CU(cudaMalloc(&pairB, np * 2));
  trans::f32ToBf16K<<<(np + 255) / 256, 256>>>(pair.dev, pairB, np);
  __nv_bfloat16* ctB;
  CU(cudaMalloc(&ctB, np * 2));
  trans::f32ToBf16K<<<(np + 255) / 256, 256>>>(ct.dev, ctB, np);

  b.forward(pairB, pm.dev);
  b.backward(ctB, pm.dev);
  CU(cudaDeviceSynchronize());
  {
    // the block's output is the ABSOLUTE activation (the last residual already
    // added p5), unlike a module's, which is a delta.
    std::vector<__nv_bfloat16> h(np);
    CU(cudaMemcpy(h.data(), b.p6b, np * 2, cudaMemcpyDeviceToHost));
    std::vector<float> y(np);
    for (size_t i = 0; i < np; ++i) y[i] = __bfloat162float(h[i]);
    printf("forward\n");
    report("out/pair", compare(y, o["out/pair"].host));
    CU(cudaMemcpy(h.data(), b.dp, np * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < np; ++i) y[i] = __bfloat162float(h[i]);
    printf("backward\n");
    report("grad/pair", compare(y, o["grad/pair"].host));
  }
  double tf = time_ms([&] { b.forward(pairB, pm.dev); }, reps);
  double tb = time_ms([&] { b.backward(ctB, pm.dev); }, reps);
  printf("\n  fwd %7.3f ms   bwd %7.3f ms   fwd+bwd %7.3f ms\n", tf, tb, tf + tb);
  return 0;
}
