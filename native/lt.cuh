// A thin cuBLASLt wrapper for the row-major GEMMs this prototype needs, with
// the epilogues that keep the wide intermediate out of HBM.
//
// WHY EPILOGUES AND NOT A HAND-WRITTEN MMA KERNEL (yet).  The pair transition is
// memory-bound, not compute-bound: unfused and in f32 it moves ~740 MB forward
// where the arithmetic is 17.2 GFLOP.  Most of that traffic is the 512-wide
// intermediate being written, read back to have a bias and a relu applied,
// written again, and read by the second GEMM.  cuBLASLt will do the bias and the
// relu inside the first GEMM's epilogue and hand back the relu's sign as a
// bitmask, and apply that same mask inside the backward GEMM (DRELU).  That
// removes the round trips without writing a single MMA instruction.
//
// ROW-MAJOR THROUGH A COLUMN-MAJOR LIBRARY.  The column-major view of a
// row-major [M, N] array with leading dimension N is the [N, M] transpose.  So
// computing D[N,M] = op(W)[N,K] * op(X)[K,M] in column major writes exactly the
// row-major [M, N] result, with no transposes anywhere.  Every call below is
// that identity; `Nn` is always the output width and is D's leading dimension,
// so the epilogue's bias vector is indexed per output channel, as wanted.
//
// TWO THINGS THE HEURISTIC REJECTS, FOUND WITH probe.cu:
//   * an epilogue in the NN form -- bias/relu epilogues are served in the TN
//     form (op(A)=T), which is why every call here transposes A;
//   * a bias whose data type differs from the matmul's.  A bf16 matmul with an
//     f32 bias is NOT_SUPPORTED for every bias epilogue.  The bias must be bf16.
//
// PLANS ARE CACHED.  cublasLtMatmulAlgoGetHeuristic is a host-side search, and
// building the descriptors is not free either.  Doing that per call put enough
// host time inside the measurement that a timed forward plus a timed backward
// came to less than a timed forward-and-backward (1.68 + 1.55 against 5.79 ms).
// Everything that does not depend on the data is built once, keyed by shape.
#pragma once
#include <cublasLt.h>
#include <cuda_bf16.h>

#include <map>
#include <tuple>

#include "common.cuh"

#define CUBLT(x)                                                               \
  do {                                                                         \
    cublasStatus_t s_ = (x);                                                   \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "%s:%d cublasLt: %d\n", __FILE__, __LINE__, (int)s_);    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

struct Plan {
  bool tout = false;   // the product is written transposed; see plan()
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t la, lb, ld;
  cublasLtMatmulAlgo_t algo;
  cublasLtMatmulAlgo_t cand[16];
  int ncand = 0;
  bool tuned = false;
  int M = 0, Nn = 0, K = 0, epi = 0;   // kept only so a failure can name itself
};

struct Lt {
  cublasLtHandle_t h = nullptr;
  void* ws = nullptr;
  size_t ws_bytes = 64u << 20;
  // (M, Nn, K, epilogue, dtype) -> plan
  std::map<std::tuple<int, int, int, int, int>, Plan> cache;

  // ONE WORKSPACE A PROCESS, not one an Lt.  Every module owns an Lt, and at
  // 64 MB each that is 604 MB of cuBLASLt scratch across a block's nine --
  // measured with BLK_MEM, and the second largest line in the stack's memory
  // after the remat save-set.  Sharing is safe exactly because everything runs
  // on one stream (Block::alloc ends with setStream): two matmuls on DIFFERENT
  // streams must not share a workspace, so if a module is ever given a stream
  // of its own again, this has to go back to one allocation an Lt.
  bool inited = false;
  void init() {
    // Idempotent: re-alloc at a smaller N calls alloc() again, and a second
    // cublasLtCreate would leak a handle a pass.  See nbrealloc in common.cuh.
    if (inited) return;
    inited = true;
    CUBLT(cublasLtCreate(&h));
    static void* shared = nullptr;
    if (!shared) CU(cudaMalloc(&shared, ws_bytes));
    ws = shared;
  }

  // `tout` asks for the product TRANSPOSED -- D as [Nn][M] row-major
  // (channel-major) instead of [M][Nn].  D^T = op(B)^T op(A)^T, so it is the
  // same two operand layouts with their roles swapped and the output's
  // leading dimension changed; cuBLAS does the transpose for free as part of
  // the write.  It exists because `gemm dcn` fed a 33.6 MB round trip through
  // HBM (transposeRowToChanK) purely to match `t`'s channel-major layout.
  Plan& plan(int M, int Nn, int K, cudaDataType dtype, cublasLtEpilogue_t epi,
             const void* bias, void* aux, bool tout = false) {
    auto key = std::make_tuple(M, Nn, K, (int)epi + (tout ? 1 << 16 : 0),
                               (int)dtype);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    Plan p;
    CUBLT(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tn = CUBLAS_OP_N;
    CUBLT(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta,
                                         sizeof(ta)));
    CUBLT(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tn,
                                         sizeof(tn)));
    p.tout = tout;
    CUBLT(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                         &epi, sizeof(epi)));
    if (bias) {
      CUBLT(cublasLtMatmulDescSetAttribute(
          p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
      cudaDataType bt = dtype;  // must match the matmul's type -- see the header
      CUBLT(cublasLtMatmulDescSetAttribute(
          p.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)));
    }
    if (aux) {
      CUBLT(cublasLtMatmulDescSetAttribute(
          p.op, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &aux, sizeof(aux)));
      int64_t ld = Nn;  // the relu bitmask follows the OUTPUT's width
      CUBLT(cublasLtMatmulDescSetAttribute(
          p.op, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ld, sizeof(ld)));
    }

    // A is the weight, column-major [K, Nn] and transposed by op(A)=T to the
    // [Nn, K] the product needs.  A column-major [K, Nn] buffer is a row-major
    // [Nn, K] one, so every call site passes the weight in its [Nn, K] form --
    // for the forward's Linears that is a transposed copy made once at setup,
    // for the backward's dX = dY W^T it is the checkpoint's own layout.
    if (!tout) {
      CUBLT(cublasLtMatrixLayoutCreate(&p.la, dtype, K, Nn, K));
      CUBLT(cublasLtMatrixLayoutCreate(&p.lb, dtype, K, M, K));
      CUBLT(cublasLtMatrixLayoutCreate(&p.ld, dtype, Nn, M, Nn));
    } else {
      // the ACTIVATION becomes op(A)=T and the weight op(B)=N, so m=M, n=Nn,
      // and D is M x Nn column-major -- which IS [Nn][M] row-major
      CUBLT(cublasLtMatrixLayoutCreate(&p.la, dtype, K, M, K));
      CUBLT(cublasLtMatrixLayoutCreate(&p.lb, dtype, K, Nn, K));
      CUBLT(cublasLtMatrixLayoutCreate(&p.ld, dtype, M, Nn, M));
    }

    cublasLtMatmulPreference_t pref;
    CUBLT(cublasLtMatmulPreferenceCreate(&pref));
    CUBLT(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
        sizeof(ws_bytes)));
    // ASK FOR MANY CANDIDATES AND TIME THEM, rather than trusting the first.
    // The heuristic is tuned for ordinary GEMM shapes; ours are not ordinary.
    // Freezing the weights means every backward GEMM is dX = dY W^T and
    // nothing else, so the shapes are extremely tall and thin -- M = N^2 rows
    // against Nn = 128 columns -- and they are the SAME few shapes tens of
    // thousands of times in a design run.  That is exactly the case where a
    // one-off search beats a heuristic: the cost is paid once a shape, at the
    // first call, and a design run then replays it for hours.
    const int kCand = 16;
    cublasLtMatmulHeuristicResult_t heur[kCand];
    int found = 0;
    CUBLT(cublasLtMatmulAlgoGetHeuristic(h, p.op, p.la, p.lb, p.ld, p.ld, pref,
                                         kCand, heur, &found));
    cublasLtMatmulPreferenceDestroy(pref);
    if (!found) {
      fprintf(stderr,
              "cublasLt: no algo for M=%d Nn=%d K=%d epi=%d dtype=%d\n", M, Nn,
              K, (int)epi, (int)dtype);
      exit(1);
    }
    p.algo = heur[0].algo;
    p.ncand = found;
    for (int i = 0; i < found && i < kCand; ++i) p.cand[i] = heur[i].algo;
    p.M = M; p.Nn = Nn; p.K = K; p.epi = (int)epi;
    return cache.emplace(key, p).first->second;
  }

  // THE BIAS AND AUX POINTERS ARE RE-SET ON EVERY CALL, and they have to be.
  //
  // They live on the matmul DESCRIPTOR, which the plan cache keys only on
  // (M, Nn, K, epilogue, dtype) -- so once a shape has been planned, every
  // later call reuses the descriptor built the first time, with whatever bias
  // pointer it was handed then.  A single module is unaffected: its bias never
  // moves.  The 48-BLOCK STACK IS NOT: it swaps 48 sets of weight pointers
  // through one Block, every block has the same shapes, and so blocks 1..47
  // silently used BLOCK 0's biases.  Depth 1 was correct by accident, which is
  // why the error appeared only from depth 2 on (7.5e-04 -> 5.2e-02) while
  // every single-module and single-block gate stayed green.
  //
  // Keying the cache on the pointers instead would build 48 plans a shape and
  // run the heuristic 48 times; setting an attribute is host-side and free.
  // Time every candidate once, on the real pointers, and keep the fastest.
  // Runs at a shape's FIRST call and never again, so a design run pays it once
  // and replays the winner for hours.  Each candidate computes the correct
  // answer, so the tuning pass is not wasted work.
  //
  // Worth asking because the shapes are not ordinary: freezing the weights
  // makes every backward GEMM dX = dY W^T and nothing else, so they are all
  // M = N^2 tall and 128 thin.  Worth 0.9% on the stack (999.97 -> 991.32 ms).
  // It costs a little accuracy -- forward 6.94e-03 -> 8.73e-03, grad/pair
  // 9.94e-02 -> 1.12e-01 -- because a different algorithm accumulates in a
  // different order; that is well inside the band the design cares about.
  // OFF BY DEFAULT, AND THAT IS A REPRODUCIBILITY DECISION.
  //
  // This is an autotuner, and it has the same failure XLA's has: it times
  // candidates with CUDA events and keeps the fastest, so WHICH ONE WINS
  // DEPENDS ON WHAT ELSE THE CARD WAS DOING.  A second process picks
  // differently, accumulates in a different order, and returns a different
  // gradient -- measured at rel 1.5e-02, which was enough to send two
  // identical design trajectories to different sequences (27 of 64
  // positions agreeing after ten steps).
  //
  // It is not a trade worth making: a native fwd+bwd measured 604.4 ms
  // against 598.7 with it on at 192 residues (+0.95%), and 84.9 against
  // 85.6 at 64.  About one per cent, for two runs that agree bit for bit.
  //
  // LT_TUNE=1 turns it back on.  NO_LT_TUNE still works and is now a no-op.
  void tune(Plan& p, const void* A, const void* B, void* D, cudaStream_t stream) {
    p.tuned = true;
    if (p.ncand < 2 || !getenv("LT_TUNE") || getenv("NO_LT_TUNE")) return;
    cudaEvent_t t0, t1;
    CU(cudaEventCreate(&t0)); CU(cudaEventCreate(&t1));
    float alpha = 1.f, beta = 0.f, best = 1e30f;
    int bi = 0;
    for (int i = 0; i < p.ncand; ++i) {
      // one warm-up, then three timed: the first call on an algo pays for its
      // workspace setup and would rank it unfairly
      bool ok = true;
      for (int w = 0; w < 1 && ok; ++w)
        ok = cublasLtMatmul(h, p.op, &alpha, B, p.la, A, p.lb, &beta, D, p.ld, D,
                            p.ld, &p.cand[i], ws, ws_bytes, stream)
             == CUBLAS_STATUS_SUCCESS;
      if (!ok) continue;                      // a candidate may not run here
      CU(cudaEventRecord(t0, stream));
      for (int r = 0; r < 3; ++r)
        cublasLtMatmul(h, p.op, &alpha, B, p.la, A, p.lb, &beta, D, p.ld, D,
                       p.ld, &p.cand[i], ws, ws_bytes, stream);
      CU(cudaEventRecord(t1, stream));
      CU(cudaEventSynchronize(t1));
      float ms = 0; CU(cudaEventElapsedTime(&ms, t0, t1));
      if (ms < best) { best = ms; bi = i; }
    }
    p.algo = p.cand[bi];
    if (getenv("LT_TUNE_LOG"))
      fprintf(stderr, "  lt tune M=%6d Nn=%4d K=%4d epi=%d -> cand %d of %d"
              " (%.3f ms)\n", p.M, p.Nn, p.K, p.epi, bi, p.ncand, best / 3);
    CU(cudaEventDestroy(t0)); CU(cudaEventDestroy(t1));
  }

  void run(Plan& p, const void* A, const void* B, void* D,
           const void* bias, void* aux, cudaStream_t stream = 0) {
    if (bias) {
      CUBLT(cublasLtMatmulDescSetAttribute(
          p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }
    if (aux) {
      CUBLT(cublasLtMatmulDescSetAttribute(
          p.op, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &aux, sizeof(aux)));
    }
    float alpha = 1.f, beta = 0.f;
    // tout swaps which operand is op(A): see plan().  The product is the
    // transpose of the untransposed call's, written straight out that way.
    const void* o1 = p.tout ? A : B;
    const void* o2 = p.tout ? B : A;
    cublasStatus_t s_ = cublasLtMatmul(h, p.op, &alpha, o1, p.la, o2, p.lb,
                                       &beta, D, p.ld, D, p.ld, &p.algo, ws,
                                       ws_bytes, stream);
    if (s_ != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cublasLtMatmul %d on M=%d Nn=%d K=%d epi=%d "
              "A=%p B=%p D=%p\n", (int)s_, p.M, p.Nn, p.K, p.epi, A, B, D);
      exit(1);
    }
  }

  // D[M,Nn] = A[M,K] @ Wt^T, with Wt the weight in its [Nn, K] row-major form.
  // Both directions are this one call: the forward's Linears pass a transposed
  // copy made once at setup, and the backward's dX = dY W^T passes the
  // checkpoint's own [Nn, K] array, read transposed in place.  There is never a
  // second GEMM for a weight gradient, because the weights are frozen.
  void matmul(int M, int Nn, int K, const void* A, const void* Wt, void* D,
              cudaDataType dtype, cublasLtEpilogue_t epi, const void* bias,
              void* aux, cudaStream_t stream = 0, bool tout = false) {
    Plan& p = plan(M, Nn, K, dtype, epi, bias, aux, tout);
    if (!p.tuned) {
      // the bias/aux pointers must be live on the descriptor before tuning,
      // because an epilogue candidate will read them
      if (bias) CUBLT(cublasLtMatmulDescSetAttribute(
          p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
      if (aux) CUBLT(cublasLtMatmulDescSetAttribute(
          p.op, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &aux, sizeof(aux)));
      tune(p, A, Wt, D, stream);
    }
    run(p, A, Wt, D, bias, aux, stream);
  }
};
