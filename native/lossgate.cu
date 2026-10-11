// Gate for the design losses (losses.cuh) against colabdesign2's own
// implementations.  The configuration is i_con -- rows on chain A, columns on
// chain B, num=2, num_pos=9 -- because that is the binder objective, not the
// intra-chain con.
#include "losses.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: lossgate <oracle-dir>\n"); return 1; }
  Oracle o(argv[1]);
  Tensor& dg = o["in/dgram"];
  int N = dg.shape[0];
  const size_t NN = (size_t)N * N;
  printf("design losses  N=%d\n", N);
  for (const char* k : {"in/dgram", "in/plddt", "in/pae"}) o[k].to_device();

  // the same two-chain split the oracle uses
  std::vector<float> rows(N), cols(N), mpair(NN);
  for (int i = 0; i < N; ++i) { rows[i] = i < N / 2; cols[i] = i >= N / 2; }
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) mpair[(size_t)i * N + j] = rows[i] * cols[j];
  float *dRows, *dMp;
  CU(cudaMalloc(&dRows, N * 4)); CU(cudaMemcpy(dRows, rows.data(), N * 4, cudaMemcpyHostToDevice));
  CU(cudaMalloc(&dMp, NN * 4)); CU(cudaMemcpy(dMp, mpair.data(), NN * 4, cudaMemcpyHostToDevice));

  float *p, *pxs, *pxms, *pk, *icon, *dp, *dpk, *ddg;
  int *pick1, *cnt1, *pick2, *cnt2;
  CU(cudaMalloc(&p, NN * 4)); CU(cudaMalloc(&pxs, NN * 64 * 4));
  CU(cudaMalloc(&pxms, NN * 64 * 4)); CU(cudaMalloc(&pk, N * 4));
  CU(cudaMalloc(&icon, 4)); CU(cudaMalloc(&dp, NN * 4));
  CU(cudaMalloc(&dpk, N * 4)); CU(cudaMalloc(&ddg, NN * 64 * 4));
  CU(cudaMalloc(&pick1, (size_t)N * losses::MAXK * 4));
  CU(cudaMalloc(&cnt1, N * 4));
  CU(cudaMalloc(&pick2, (size_t)losses::MAXK * 4));
  CU(cudaMalloc(&cnt2, 4));

  losses::conPerPairK<<<(int)((NN + 255) / 256), 256>>>(dg.dev, 14.0f, 0, p,
                                                        pxs, pxms, N);
  losses::minKK<true><<<N, 32, N>>>(p, dMp, 2, pk, pick1, cnt1, N, N);
  losses::minKK<false><<<1, 32, N>>>(pk, dRows, 9, icon, pick2, cnt2, N, 0);
  CU(cudaDeviceSynchronize());
  report("out/icon", compare(fetch(icon, 1), o["out/icon"].host));

  // backward: cotangent 1 on the scalar
  {
    std::vector<float> one(1, 1.f);
    float* d1; CU(cudaMalloc(&d1, 4));
    CU(cudaMemcpy(d1, one.data(), 4, cudaMemcpyHostToDevice));
    CU(cudaMemset(dpk, 0, N * 4));
    losses::minKBwdK<<<1, 32>>>(pick2, cnt2, d1, dpk, 1, N, 0);
    CU(cudaMemset(dp, 0, NN * 4));
    losses::minKBwdK<<<N, 32>>>(pick1, cnt1, dpk, dp, N, N, 1);
    losses::conPerPairBwdK<<<(int)((NN + 255) / 256), 256>>>(dp, pxs, pxms, p,
                                                             14.0f, 0, ddg, N);
    CU(cudaDeviceSynchronize());
    report("grad/dgram", compare(fetch(ddg, NN * 64), o["grad/dgram"].host));
  }

  // plddt: centres 0.01, 0.03, ... and the loss is 1 - masked mean
  float *plv, *plp, *dpl, *dplog;
  CU(cudaMalloc(&plv, N * 4)); CU(cudaMalloc(&plp, (size_t)N * 50 * 4));
  CU(cudaMalloc(&dpl, N * 4)); CU(cudaMalloc(&dplog, (size_t)N * 50 * 4));
  losses::binExpectK<50><<<(N + 255) / 256, 256>>>(o["in/plddt"].dev, 0.01f,
                                                   0.02f, plv, plp, N);
  CU(cudaDeviceSynchronize());
  {
    auto v = fetch(plv, N);
    double num = 0, den = 0;
    for (int i = 0; i < N; ++i) { num += v[i] * rows[i]; den += rows[i]; }
    std::vector<float> got(1, (float)(1.0 - num / (1e-8 + den)));
    report("out/plddt", compare(got, o["out/plddt"].host));
    std::vector<float> dv(N);
    for (int i = 0; i < N; ++i) dv[i] = (float)(-rows[i] / (1e-8 + den));
    CU(cudaMemcpy(dpl, dv.data(), N * 4, cudaMemcpyHostToDevice));
    losses::binExpectBwdK<50><<<(N + 255) / 256, 256>>>(plp, plv, dpl, 0.01f,
                                                        0.02f, dplog, N);
    CU(cudaDeviceSynchronize());
    report("grad/plddt", compare(fetch(dplog, (size_t)N * 50),
                                 o["grad/plddt"].host));
  }

  // pae: breaks linspace(0, 31, 63), centres shifted by half a step
  float *pv, *pp, *dpv, *dpaelog;
  CU(cudaMalloc(&pv, NN * 4)); CU(cudaMalloc(&pp, NN * 64 * 4));
  CU(cudaMalloc(&dpv, NN * 4)); CU(cudaMalloc(&dpaelog, NN * 64 * 4));
  const float step = 31.0f / 62.0f;
  losses::binExpectK<64><<<(int)((NN + 255) / 256), 256>>>(
      o["in/pae"].dev, step / 2.f, step, pv, pp, NN);
  CU(cudaDeviceSynchronize());
  {
    auto v = fetch(pv, NN);
    double num = 0, den = 0;
    for (size_t i = 0; i < NN; ++i) { num += v[i] * mpair[i]; den += mpair[i]; }
    std::vector<float> got(1, (float)(num / (1e-8 + den)));
    report("out/pae", compare(got, o["out/pae"].host));
    std::vector<float> dv(NN);
    for (size_t i = 0; i < NN; ++i) dv[i] = (float)(mpair[i] / (1e-8 + den));
    CU(cudaMemcpy(dpv, dv.data(), NN * 4, cudaMemcpyHostToDevice));
    losses::binExpectBwdK<64><<<(int)((NN + 255) / 256), 256>>>(
        pp, pv, dpv, step / 2.f, step, dpaelog, NN);
    CU(cudaDeviceSynchronize());
    report("grad/pae", compare(fetch(dpaelog, NN * 64), o["grad/pae"].host));
  }
  return 0;
}
