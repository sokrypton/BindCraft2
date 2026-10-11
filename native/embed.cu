// Gate for the input embedder (embed.cuh).
//
// The oracle seeds TWO CHAINS, because every relative-position branch worth
// testing is the cross-chain one: a single chain exercises only the clipped
// offset and would leave the "different chain" and "different entity" bins
// unvisited.
#include "embed.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: embed <oracle-dir> [reps]\n"); return 1; }
  Oracle o(argv[1]);
  int reps = argc > 2 ? atoi(argv[2]) : 20;
  Tensor& tf0 = o["in/target_feat"];
  int N = tf0.shape[0];
  Tensor& mf = o["in/msa_feat"];
  int S = mf.shape[0];
  printf("embedder  N=%d  S=%d  msa=%d  pair=%d  single=%d\n", N, S, embed::CM,
         embed::CZ, embed::CS);
  for (const char* k : {"in/target_feat", "in/msa_feat", "in/residue_index",
                        "in/asym_id", "in/entity_id", "in/sym_id", "ct/msa",
                        "ct/pair"})
    o[k].to_device();

  // target_feat arrives 20-wide and the model pads it to 21
  std::vector<float> tfp((size_t)N * embed::NAA, 0.f);
  for (int i = 0; i < N; ++i)
    for (int c = 0; c < 20; ++c)
      tfp[(size_t)i * embed::NAA + c] = tf0.host[(size_t)i * 20 + c];
  float* tf;
  CU(cudaMalloc(&tf, tfp.size() * 4));
  CU(cudaMemcpy(tf, tfp.data(), tfp.size() * 4, cudaMemcpyHostToDevice));

  embed::Embed m;
  m.alloc(N, S);
  m.setWeights(o);
  float *dtf, *dmf;
  CU(cudaMalloc(&dtf, (size_t)N * embed::NAA * 4));
  CU(cudaMalloc(&dmf, (size_t)S * N * embed::NMSA_F * 4));

  auto fwd = [&] {
    m.forward(tf, mf.dev, o["in/residue_index"].dev, o["in/asym_id"].dev,
              o["in/entity_id"].dev, o["in/sym_id"].dev);
  };
  auto bwd = [&] {
    m.backward(o["ct/msa"].dev, o["ct/pair"].dev, nullptr, tf, dtf, dmf);
  };
  fwd(); bwd();
  CU(cudaDeviceSynchronize());
  printf("forward\n");
  report("out/msa", compare(fetch(m.msa, (size_t)S * N * embed::CM),
                            o["out/msa"].host));
  report("out/pair", compare(fetch(m.pair, (size_t)N * N * embed::CZ),
                             o["out/pair"].host));
  printf("backward\n");
  // the oracle differentiates the UNPADDED target_feat, so drop the pad column
  {
    auto g = fetch(dtf, (size_t)N * embed::NAA);
    std::vector<float> g20((size_t)N * 20);
    for (int i = 0; i < N; ++i)
      for (int c = 0; c < 20; ++c)
        g20[(size_t)i * 20 + c] = g[(size_t)i * embed::NAA + c];
    report("grad/target_feat", compare(g20, o["grad/target_feat"].host));
  }
  report("grad/msa_feat", compare(fetch(dmf, (size_t)S * N * embed::NMSA_F),
                                  o["grad/msa_feat"].host));
  double tf_ms = time_ms(fwd, reps), tb = time_ms(bwd, reps);
  printf("\n  fwd %7.3f ms   bwd %7.3f ms\n", tf_ms, tb);
  return 0;
}
