// Gate for the three-layer transition alone (foldit.cuh): three Linears, two
// relus, nothing else.
#include "foldit.cuh"

int main(int argc, char** argv) {
  Oracle o(argv[1]);
  Tensor& x = o["in/single"];
  int N = x.shape[0];
  const int C = foldit::C;
  size_t n = (size_t)N * C;
  printf("transition (3 linear, 2 relu)  N=%d  C=%d\n", N, C);
  x.to_device(); o["ct/single"].to_device();

  foldit::FoldIteration m;
  m.alloc(N);
  auto get = [&](const char* nm, float** dst, size_t cnt) {
    Tensor& t = o[std::string("param/") + nm];
    CU(cudaMalloc(dst, cnt * 4));
    CU(cudaMemcpy(*dst, t.host.data(), cnt * 4, cudaMemcpyHostToDevice));
  };
  get("transition/weights", &m.w1, (size_t)C * C);
  get("transition/bias", &m.b1, C);
  get("transition_1/weights", &m.w2, (size_t)C * C);
  get("transition_1/bias", &m.b2, C);
  get("transition_2/weights", &m.w3, (size_t)C * C);
  get("transition_2/bias", &m.b3, C);

  int g = (int)((n + 255) / 256);
  m.gemm(N, C, C, x.dev, m.w1, m.h1);
  addRowBiasK<<<g, 256>>>(m.h1, m.b1, C, n);
  foldit::reluK<<<g, 256>>>(m.h1, n);
  m.gemm(N, C, C, m.h1, m.w2, m.h2);
  addRowBiasK<<<g, 256>>>(m.h2, m.b2, C, n);
  foldit::reluK<<<g, 256>>>(m.h2, n);
  m.gemm(N, C, C, m.h2, m.w3, m.tr);
  addRowBiasK<<<g, 256>>>(m.tr, m.b3, C, n);
  CU(cudaDeviceSynchronize());
  report("out/single", compare(fetch(m.tr, n), o["out/single"].host));

  m.gemmNT(N, C, C, o["ct/single"].dev, m.w3, m.dh);
  foldit::reluBwdK<<<g, 256>>>(m.dh, m.h2, n);
  m.gemmNT(N, C, C, m.dh, m.w2, m.d3);
  foldit::reluBwdK<<<g, 256>>>(m.d3, m.h1, n);
  m.gemmNT(N, C, C, m.d3, m.w1, m.dh);
  CU(cudaDeviceSynchronize());
  report("grad/single", compare(fetch(m.dh, n), o["grad/single"].host));
  return 0;
}
