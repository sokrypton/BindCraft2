// Gate for the f32 LayerNorm alone (foldit.cuh), forward and backward.
#include "foldit.cuh"

int main(int argc, char** argv) {
  Oracle o(argv[1]);
  Tensor& x = o["in/single"];
  int N = x.shape[0];
  const int C = foldit::C;
  size_t n = (size_t)N * C;
  printf("layer norm  N=%d  C=%d\n", N, C);
  x.to_device(); o["ct/single"].to_device();
  float *sc, *of, *y, *mu, *rs, *dx;
  auto get = [&](const char* nm, float** dst) {
    Tensor& t = o[std::string("param/attention_layer_norm/") + nm];
    CU(cudaMalloc(dst, (size_t)C * 4));
    CU(cudaMemcpy(*dst, t.host.data(), (size_t)C * 4, cudaMemcpyHostToDevice));
  };
  get("scale", &sc); get("offset", &of);
  CU(cudaMalloc(&y, n * 4)); CU(cudaMalloc(&dx, n * 4));
  CU(cudaMalloc(&mu, (size_t)N * 4)); CU(cudaMalloc(&rs, (size_t)N * 4));
  int gw = (N + 7) / 8;
  foldit::lnFwdK<<<gw, 256>>>(x.dev, sc, of, y, mu, rs, N);
  foldit::lnBwdK<<<gw, 256>>>(o["ct/single"].dev, x.dev, sc, mu, rs, dx, N);
  CU(cudaDeviceSynchronize());
  report("out/single", compare(fetch(y, n), o["out/single"].host));
  report("grad/single", compare(fetch(dx, n), o["grad/single"].host));
  return 0;
}
