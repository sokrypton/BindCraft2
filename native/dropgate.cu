// Does the dropout BACKWARD use the same mask the forward did?
//
// Nothing already in the directory can answer this.  The rate-0 gates compare
// against a JAX oracle, and with dropout on there IS no oracle: our masks come
// from a hash of (seed, depth, module, index) and JAX's from threefry, so the
// two never agree sample by sample.  A mismatched mask -- backward dropping a
// different set of channels than the forward kept -- would leave every number
// in the previous gate looking exactly as it does now, and would quietly bias
// every design step.
//
// So this is a SELF-consistency test instead, which needs no oracle.  Define
// L = <W, forward(x)> for a fixed random W.  Then backward(W) is dL/dx, and
// for any direction v the analytic directional derivative <dL/dx, v> must
// equal the numerical one, (L(x + ev) - L(x - ev)) / 2e.  The mask is a
// deterministic function of the seed, so it is held fixed across all three
// forwards -- exactly the condition under which the identity holds.
//
// The chain is bf16, so the numerical side is noisy.  That is why rate 0 runs
// FIRST as a control: it says what agreement this estimator can reach at all,
// and the dropout rows are only meaningful against that ceiling.
#include "block.cuh"

#include <cstdio>

__global__ void axpyBf16K(__nv_bfloat16* dst, const __nv_bfloat16* src,
                          const float* v, float eps, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(__bfloat162float(src[i]) + eps * v[i]);
}
__global__ void dotBf16K(const __nv_bfloat16* a, const float* b, double* out,
                         size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(out, (double)__bfloat162float(a[i]) * (double)b[i]);
}
__global__ void dotF32K(const float* a, const float* b, double* out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(out, (double)a[i] * (double)b[i]);
}

int main(int argc, char** argv) {
  Oracle o(argv[1]);
  auto& pair = o["in/pair"]; auto& msa = o["in/msa"];
  auto& pmask = o["in/pair_mask"]; auto& mmask = o["in/msa_mask"];
  auto& ctp = o["ct/pair"]; auto& ctm = o["ct/msa"];
  int N = pair.shape[0], S = msa.shape[0];
  size_t np = (size_t)N * N * 128, nm = (size_t)S * N * 256;
  const std::string P =
      o.has("param/evoformer_iteration/pair_transition/transition1/weights") ||
      o.has("param/evoformer_iteration/outer_product_mean/left_projection/weights")
          ? "param/evoformer_iteration/" : "param/";

  // Tensor::dev is null until to_device(); block.cu uploads through a path of
  // its own, so this gate has to ask for each array it touches.
  for (auto* x : {&pair, &msa, &pmask, &mmask, &ctp, &ctm}) x->to_device();

  int gp = (np + 255) / 256, gm = (nm + 255) / 256;
  __nv_bfloat16 *pairIn, *pairPert, *ctpB;
  float *msaB, *v, *dL;
  double* acc;
  CU(cudaMalloc(&pairIn, np * 2)); CU(cudaMalloc(&pairPert, np * 2));
  CU(cudaMalloc(&ctpB, np * 2));   CU(cudaMalloc(&msaB, nm * 4));
  CU(cudaMalloc(&v, np * 4));      CU(cudaMalloc(&dL, np * 4));
  CU(cudaMalloc(&acc, 8));
  {  // round the inputs once, as block.cu does
    __nv_bfloat16* msaH; CU(cudaMalloc(&msaH, nm * 2));
    trans::f32ToBf16K<<<gm, 256>>>(msa.dev, msaH, nm);
    opm::bf16ToF32K<<<gm, 256>>>(msaH, msaB, nm);
    toBf16K<<<gp, 256>>>(pairIn, pair.dev, np);
    toBf16K<<<gp, 256>>>(ctpB, ctp.dev, np);
  }
  // the probe direction: reuse the oracle's cotangent, scaled to unit-ish rms,
  // so v has the same spectral character as the activations it perturbs
  CU(cudaMemcpy(v, ctp.dev, np * 4, cudaMemcpyDeviceToDevice));

  auto loss = [&](block::Block& b, const __nv_bfloat16* x) {
    b.forward(const_cast<__nv_bfloat16*>(x), msaB, pmask.dev, mmask.dev);
    CU(cudaMemset(acc, 0, 8));
    if (BLK_BF16CHAIN) dotBf16K<<<gp, 256>>>(b.p6b, ctp.dev, acc, np);
    else               dotF32K<<<gp, 256>>>(b.p6, ctp.dev, acc, np);
    CU(cudaDeviceSynchronize());
    double h; CU(cudaMemcpy(&h, acc, 8, cudaMemcpyDeviceToHost));
    return h;
  };

  printf("  %-22s %14s %14s %9s\n", "configuration", "analytic", "numerical",
         "ratio");
  auto run = [&](const char* label, bool drop, uint32_t seed, float eps) {
    block::Block b;
    // the same prefix detection block.cu does: a pair-only dump has no
    // outer_product_mean, and asking for one exits on a missing key
    b.do_msa = o.has(P + "outer_product_mean/left_projection/weights") ||
               o.has("param/outer_product_mean/left_projection/weights");
    b.do_pair = o.has(P + "triangle_multiplication_outgoing/projection/weights") ||
                o.has("param/triangle_multiplication_outgoing/projection/weights");
    b.use_dropout = drop;
    b.alloc(N, S);
    b.setWeights(o, P);
    b.setSeed(seed);

    // analytic: <dL/dx, v>
    b.forward(pairIn, msaB, pmask.dev, mmask.dev);
    b.backward(ctpB, ctm.dev, pmask.dev, mmask.dev);
    CU(cudaDeviceSynchronize());
    widenBf16K<<<gp, 256>>>(dL, (const __nv_bfloat16*)b.dp, np);
    CU(cudaMemset(acc, 0, 8));
    dotF32K<<<gp, 256>>>(dL, v, acc, np);
    CU(cudaDeviceSynchronize());
    double analytic; CU(cudaMemcpy(&analytic, acc, 8, cudaMemcpyDeviceToHost));

    // numerical: central difference on the same fixed mask
    axpyBf16K<<<gp, 256>>>(pairPert, pairIn, v, eps, np);
    double lp = loss(b, pairPert);
    axpyBf16K<<<gp, 256>>>(pairPert, pairIn, v, -eps, np);
    double lm = loss(b, pairPert);
    double numeric = (lp - lm) / (2 * eps);
    printf("  %-22s %14.4e %14.4e %9.4f\n", label, analytic, numeric,
           analytic ? numeric / analytic : 0.0);
  };
  const float eps = argc > 2 ? atof(argv[2]) : 1e-2f;
  printf("  (eps %.0e, central difference on a bf16 chain)\n\n", eps);
  run("CONTROL rate 0", false, 0, eps);
  run("dropout seed 1", true, 1, eps);
  run("dropout seed 2", true, 2, eps);
  run("dropout seed 99", true, 99, eps);
  return 0;
}
