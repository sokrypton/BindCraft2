// Gate for the fold iteration MINUS attention and the rigid update: both
// LayerNorms and the three-layer transition.  It exists because the assembled
// FoldIteration's backward is 25x looser than IPA's and QuatRigid's are on
// their own, and a scale ladder did not close the gap -- so the error is in a
// piece that had no gate, or it is the conditioning of the composition.  This
// is what tells the two apart.
#include "foldit.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: trans3 <oracle-dir> [reps]\n"); return 1; }
  Oracle o(argv[1]);
  int reps = argc > 2 ? atoi(argv[2]) : 20;
  Tensor& x = o["in/single"];
  int N = x.shape[0];
  const size_t n = (size_t)N * foldit::C;
  printf("layernorm + 3-layer transition  N=%d  C=%d\n", N, foldit::C);
  x.to_device(); o["ct/single"].to_device();

  foldit::FoldIteration m;   // only the non-attention half is exercised
  m.alloc(N);
  // the attention and rigid weights are absent from this dump, so load by hand
  auto get = [&](const char* name, float** dst, size_t cnt) {
    Tensor& t = o[std::string("param/") + name];
    CU(cudaMalloc(dst, cnt * 4));
    CU(cudaMemcpy(*dst, t.host.data(), cnt * 4, cudaMemcpyHostToDevice));
  };
  const int C = foldit::C;
  get("attention_layer_norm/scale", &m.ln1s, C);
  get("attention_layer_norm/offset", &m.ln1o, C);
  get("transition_layer_norm/scale", &m.ln2s, C);
  get("transition_layer_norm/offset", &m.ln2o, C);
  get("transition/weights", &m.w1, (size_t)C * C);
  get("transition/bias", &m.b1, C);
  get("transition_1/weights", &m.w2, (size_t)C * C);
  get("transition_1/bias", &m.b2, C);
  get("transition_2/weights", &m.w3, (size_t)C * C);
  get("transition_2/bias", &m.b3, C);

  float* dx;
  CU(cudaMalloc(&dx, n * 4));
  auto fwd = [&] { m.midForward(x.dev); };
  auto bwd = [&] { m.midBackward(o["ct/single"].dev, dx); };
  fwd(); bwd();
  CU(cudaDeviceSynchronize());
  printf("forward\n");
  report("out/single", compare(fetch(m.a4, n), o["out/single"].host));
  printf("backward\n");
  report("grad/single", compare(fetch(dx, n), o["grad/single"].host));
  double tf = time_ms(fwd, reps), tb = time_ms(bwd, reps);
  printf("\n  fwd %7.3f ms   bwd %7.3f ms\n", tf, tb);
  return 0;
}
