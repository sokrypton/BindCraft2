// Gate for the template pair STACK (tplstack.cuh) on top of the block gate.
//
// tplblock.cu already gates one iteration against the oracle.  What a stack
// adds is two things, and this asks about exactly those:
//
//   1. the wrapper is transparent -- a depth-1 stack must reproduce a bare
//      TplBlock bit for bit, forward and backward, or the chaining is wrong
//      before any depth question arises.
//
//   2. THE SAVE-SET SURVIVES AN INTERVENING FORWARD.  This is the whole
//      reason the save-set crosses as a value: JAX dispatches asynchronously,
//      so a second forward can overtake the first backward.  If the blocks
//      simply kept their own buffers, that second forward would overwrite
//      them and the backward would differentiate the wrong input -- silently,
//      as a wrong gradient rather than an error.  So: run forward(A), then
//      forward(B) on different data, then backward(A), and require the same
//      answer as a backward taken immediately.
#include "tplstack.cuh"
#include <cstdio>

__global__ void scaleBf16K(__nv_bfloat16* dst, const __nv_bfloat16* src,
                           float k, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(__bfloat162float(src[i]) * k);
}

static double cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double d = 0, na = 0, nb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
  }
  return d / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

static std::vector<float> widen(const __nv_bfloat16* d, size_t n) {
  std::vector<__nv_bfloat16> h(n);
  CU(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost));
  std::vector<float> y(n);
  for (size_t i = 0; i < n; ++i) y[i] = __bfloat162float(h[i]);
  return y;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: tplstack <oracle-dir>\n"); return 1; }
  Oracle o(argv[1]);
  Tensor& pair = o["in/pair"];
  int N = pair.shape[0], C = pair.shape.back();
  if (C != tplstack::CZ) {
    fprintf(stderr, "this is a %d-channel dump; the template stack is %d\n",
            C, tplstack::CZ);
    return 1;
  }
  size_t np = (size_t)N * N * C;
  pair.to_device();
  Tensor& ct = o["ct/pair"]; ct.to_device();
  Tensor& pm = o["in/pair_mask"]; pm.to_device();
  int gp = (int)((np + 255) / 256);

  __nv_bfloat16 *pairB, *ctB, *otherB;
  CU(cudaMalloc(&pairB, np * 2)); CU(cudaMalloc(&ctB, np * 2));
  CU(cudaMalloc(&otherB, np * 2));
  trans::f32ToBf16K<<<gp, 256>>>(pair.dev, pairB, np);
  trans::f32ToBf16K<<<gp, 256>>>(ct.dev, ctB, np);
  // something materially different for the intervening forward
  scaleBf16K<<<gp, 256>>>(otherB, pairB, -2.5f, np);
  CU(cudaDeviceSynchronize());

  printf("template pair stack  N=%d  C=%d\n", N, C);

  // ---- 1. a depth-1 stack against a bare block
  tplblock::TplBlock one;
  one.alloc(N);
  one.setWeights(o, "param/");
  one.forward(pairB, pm.dev);
  one.backward(ctB, pm.dev);
  CU(cudaDeviceSynchronize());
  auto bo = widen(one.p6b, np), bg = widen(one.dp, np);

  tplstack::TplStack s1;
  s1.alloc(N, 1);
  s1.setWeightsFlat(o);
  s1.setDropout(false, nullptr);
  __nv_bfloat16* save1;
  CU(cudaMalloc(&save1, s1.saveElems() * 2));
  s1.forward(pairB, pm.dev, save1);
  s1.backward(ctB, pm.dev, save1);
  CU(cudaDeviceSynchronize());
  auto so = widen(s1.out(), np), sg = widen(s1.dpair(), np);
  printf("  depth-1 vs bare block   out cos %.9f   grad cos %.9f\n",
         cosine(bo, so), cosine(bg, sg));

  // ---- 2. the save-set survives an intervening forward, at the real depth 2
  tplstack::TplStack s2;
  s2.alloc(N, 2);
  s2.setWeightsFlat(o);
  s2.setDropout(false, nullptr);   // the gates measure the rate-0 model
  __nv_bfloat16* saveA;
  CU(cudaMalloc(&saveA, s2.saveElems() * 2));

  s2.forward(pairB, pm.dev, saveA);
  s2.backward(ctB, pm.dev, saveA);           // immediately
  CU(cudaDeviceSynchronize());
  auto clean_out = widen(s2.out(), np), clean_grad = widen(s2.dpair(), np);

  // THE FLOOR FIRST.  This backward accumulates with float atomics, so two
  // identical runs do not agree bit for bit either -- without knowing by how
  // much, a race and ordinary nondeterminism are indistinguishable, and the
  // first version of this gate called 0.99999 a race on a threshold I had
  // picked out of the air.
  s2.forward(pairB, pm.dev, saveA);
  s2.backward(ctB, pm.dev, saveA);
  CU(cudaDeviceSynchronize());
  auto repeat_grad = widen(s2.dpair(), np);
  double floor_cos = cosine(clean_grad, repeat_grad);

  s2.forward(pairB, pm.dev, saveA);          // A
  s2.forward(otherB, pm.dev, nullptr);       // B overtakes it
  s2.backward(ctB, pm.dev, saveA);           // A's backward, from A's saves
  CU(cudaDeviceSynchronize());
  auto raced_grad = widen(s2.dpair(), np);
  double c = cosine(clean_grad, raced_grad);
  printf("  CONTROL  identical run, no interference   cos %.9f\n", floor_cos);
  printf("  grad after an intervening forward         cos %.9f\n", c);
  // the race is real only if it moves the gradient MORE than repeating the
  // same computation does
  double slack = (1.0 - floor_cos) * 4 + 1e-9;
  printf("  -> %s\n", (1.0 - c) <= slack
         ? "the save-set holds: no worse than repeating the same run"
         : "RACE: the saves did not survive");
  return 0;
}
