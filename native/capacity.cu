// How large a target fits, and what it costs -- the memory FLOOR of the
// 48-block stack as a function of length.
//
// No oracle and no gate: this allocates the stack on random data and reports
// what it takes and how long it runs.  The question it answers is "can we do
// a bigger target", which is about the floor (scratch + save-set), not about
// the keep tiers -- those are an explicit memory/time dial that degrades to
// nothing under pressure.
#include "stack.cuh"

int main(int argc, char** argv) {
  int N = argc > 1 ? atoi(argv[1]) : 256;
  int depth = argc > 2 ? atoi(argv[2]) : 48;
  // THE WEIGHTS ARE N-INDEPENDENT, so one directory serves every length --
  // which is the only reason this probe can sweep lengths without an oracle
  // dump per length.  Leaving them out segfaults rather than erroring: the
  // forward dereferences null weight pointers.
  const char* wdir = argc > 3 ? argv[3] : "stack_weights";
  int S = 2;
  size_t fb0, tot;
  CU(cudaMemGetInfo(&fb0, &tot));

  Oracle w(wdir);
  stack::Stack s;
  s.alloc(N, S, depth);
  s.setWeights(w);
  size_t fb1;
  CU(cudaMemGetInfo(&fb1, &tot));

  size_t np = (size_t)N * N * block::CZ, nm = (size_t)S * N * block::CM;
  float *pair, *msa, *pmask, *mmask, *ctp, *ctm;
  CU(cudaMalloc(&pair, np * 4)); CU(cudaMalloc(&msa, nm * 4));
  CU(cudaMalloc(&pmask, (size_t)N * N * 4)); CU(cudaMalloc(&mmask, nm / block::CM * 4));
  CU(cudaMalloc(&ctp, np * 4)); CU(cudaMalloc(&ctm, nm * 4));
  CU(cudaMemset(pair, 0, np * 4)); CU(cudaMemset(msa, 0, nm * 4));
  {
    std::vector<float> ones((size_t)N * N, 1.f);
    CU(cudaMemcpy(pmask, ones.data(), (size_t)N * N * 4, cudaMemcpyHostToDevice));
    std::vector<float> om((size_t)S * N, 1.f);
    CU(cudaMemcpy(mmask, om.data(), (size_t)S * N * 4, cudaMemcpyHostToDevice));
  }
  CU(cudaMemset(ctp, 0, np * 4)); CU(cudaMemset(ctm, 0, nm * 4));

  s.forward(pair, msa, pmask, mmask);
  s.backward(ctp, ctm, pmask, mmask);
  CU(cudaDeviceSynchronize());
  size_t fb2;
  CU(cudaMemGetInfo(&fb2, &tot));
  double tf = time_ms([&] { s.forward(pair, msa, pmask, mmask); }, 2);
  double tb = time_ms([&] { s.backward(ctp, ctm, pmask, mmask); }, 2);
  printf("N=%-4d depth=%-3d  alloc %6.2f GB  peak %6.2f GB of %.1f  "
         "fwd %7.1f ms  bwd %7.1f ms\n",
         N, depth, (fb0 - fb1) / 1e9, (tot - fb2) / 1e9, tot / 1e9, tf, tb);
  return 0;
}
