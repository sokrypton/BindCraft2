// The backward of AF2's triangle attention, against a forward that stored
// nothing but its row log-sum-exp.
//
// WHY THIS IS A HYBRID AND NOT A THREE-PASS FLASH.  The four gradients this owes
// reduce along three different axes:
//
//   dQ[b,h,i,:] = sum over KEYS       dS[b,h,i,j] K[b,h,j,:]
//   dK[b,h,j,:] = sum over QUERIES    dS[b,h,i,j] Q[b,h,i,:]
//   dV[b,h,j,:] = sum over QUERIES    P [b,h,i,j] dO[b,h,i,:]
//   dnbb[h,i,j] = sum over the BATCH  dS[b,h,i,j]
//
// The first three are the ordinary flash backward and a block that owns
// (b, h, query tile) can serve dQ outright and dK/dV by atomic accumulation --
// there are only N/BM query tiles, so the contention is 2-4 deep, not N.  dnbb
// is the one AF2 adds, because its attention bias is computed from the pair
// itself, and it reduces over the axis the grid is parallel in.  Serving it
// inside this loop order would mean N_b-deep atomics on [H,N,N]; serving it in
// its own pass would mean recomputing S and dP a third time.  Writing dS once
// and reducing it coalesced is cheaper than either, so dS -- and only dS -- is
// materialised.
//
// P is recomputed from Q, K, the bias and lse rather than stored.  That the
// reconstruction is exact is gated in triattn.cu ("P from lse vs w").
// NO `#pragma once`: included once per HEAD DIM by flash_bwd.cuh.  FB_NS /
// FB_FD select the pass; see the note there.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "common.cuh"

#ifndef PROBE_KV
#define PROBE_KV 0
#endif
// PROBE_NOSTORE: WHAT THE ROUND TRIPS COST, AS A SET.
//
// This file says in three places that the kernel IS its five store -> shared
// -> reload round trips, and that the mma.sync rewrite is the way to retire
// them -- and then prices that rewrite at 1.7% on an OCCUPANCY-and-bank-
// conflict basis, which is a different axis.  The round trips themselves have
// never been priced together.  One of them was (FB_REGSM, the softmax, -9.5%
// and now the default), which is the only reason to think the set is worth
// more than 1.7%.
//
// This drops the four remaining `store_matrix_sync` calls.  The consumers
// then read whatever was in shared, so the arithmetic is wrong; the loads,
// the barriers and every mma stay exactly where they were.  The delta is a
// LOWER bound on the set -- an mma.sync rewrite retires the matching loads
// too.  If this is small the rewrite is dead for the right reason at last.
#ifndef PROBE_NOSTORE
#define PROBE_NOSTORE 0
#endif
// PROBE_NOPS: the OTHER half of what chaining retires.  Three of the matmuls
// take their A operand by loading P or dS back out of shared -- and under
// mma.sync that operand IS the previous accumulator, so the load does not
// happen at all.  This skips those three loads and leaves the fragment
// holding whatever it had: wrong arithmetic, every mma and barrier still
// there, one load fewer each.  PROBE_NOSTORE + PROBE_NOPS together bound
// what an mma.sync rewrite can take.
#ifndef PROBE_NOPS
#define PROBE_NOPS 0
#endif
// FILLED, NOT LEFT UNDEFINED.  The first version of this probe simply skipped
// the load, and the compiler then kept the fragment live across the loop:
// 96 registers -> 128 with PROBE_NOPS and -> 202 with both probes, which is
// 1 block an SM instead of 3.  "Both" measured SLOWER than the baseline --
// removing instructions cannot do that, and the register dump said why.
// fill_fragment keeps `a` defined and not loop-carried, so the register count
// holds and the delta is the load.
#if PROBE_NOPS
#define FB_LOADPS(f, p, l) fill_fragment(f, __float2half(1.f))
#else
#define FB_LOADPS(f, p, l) load_matrix_sync(f, p, l)
#endif
#if PROBE_NOSTORE
#define FB_STORE(...) do { } while (0)
#else
#define FB_STORE(...) store_matrix_sync(__VA_ARGS__)
#endif
// PROBE_NOSTORE_DP: only the dP store, which is the ONLY one an mma.sync
// rewrite can actually retire.  The other three stage dV, dQ and dK for
// COALESCED GLOBAL ATOMICS, and doing those from the accumulator's own lane
// layout is what FB_REGDS and FB_REGDQ already measured as much worse.  And
// the two transposed operands (dV = P^T dO, dK = dS^T Q) go through shared
// under mma.sync too -- lf_flash.cuh's forward does exactly that with
// `ldsm4t`, ldmatrix.trans out of shared.  So the chaining prize is this one
// store, not the set of four.
#if PROBE_NOSTORE_DP
#define FB_STORE_DP(...) do { } while (0)
#else
#define FB_STORE_DP(...) FB_STORE(__VA_ARGS__)
#endif
#ifndef PROBE_NOSTORE_DP
#define PROBE_NOSTORE_DP 0
#endif
// THE SOFTMAX HERE IS ELEMENTWISE, so it does not need the "lane == key"
// layout it currently round-trips through shared to get.  The forward already
// stored the row log-sum-exp, so there is no reduction: every element of
// P is exp2(S*log2e + bias - lse) of the element of S in the same place.  S
// arrives in a wmma ACCUMULATOR, whose element -> (row, col) map is fixed and
// was measured rather than assumed (dev scratch fraglayout.cu, 16x16x16 f32):
//
//     row = (lane / 4) + 8 * ((e / 2) % 2)
//     col = (lane % 4) * 2 + (e % 2) + 8 * (e / 4)
//
// so the softmax can be applied in registers and P written straight out,
// retiring one of the five store -> shared -> reload round trips the note
// above names as what this kernel actually is.  e and e+1 are adjacent
// columns, so P goes out as half2.
//
// THE CATCH, and it is why this is a flag and not a rewrite: the `bias` read
// stops being 32 consecutive keys of one row and becomes 8 rows x 16 B, which
// is half the sector efficiency on a stream the same size as K/V's 128 MB.
#ifndef FB_REGSM
#define FB_REGSM 1
#endif
// The SECOND accumulator is the same shape of problem: dS = P * (dP - D) is
// elementwise too, dP arrives in an accumulator, and P is already in shared
// where a scattered read of it is free (LDH = 40 makes the 32 lanes of the
// fragment layout land on 32 distinct banks).  Its own flag because, unlike
// the softmax, its OUTPUTS go to global -- the dnbb atomics and the optional
// dS write stop being 32 contiguous keys.
//
// **AND IT LOSES, BADLY: 3.802 ms against 2.822.**  The same change that is
// worth -13.2% on the softmax is worth +34.7% here, and the difference is
// exactly where the outputs go.  The softmax writes P to SHARED, where the
// fragment layout is free; dS writes the dnbb ATOMICS to global, and 32
// contiguous keys becoming eight rows of scattered pairs costs far more than
// a round trip saves.  Left at 0, with the measurement, because "do the other
// accumulator too" is the obvious next thought.
#ifndef FB_REGDS
#define FB_REGDS 0
#endif
// dQ is the THIRD accumulator and the one the rule most clearly favours: its
// round trip exists only to reach `dq`, which is REGISTERS, and it runs
// sixteen times a block (eight key tiles x two channel tiles) while the one
// global write it feeds happens once, in the epilogue.  Holding dQ in the
// accumulator's own layout costs exactly the same registers -- dq[16] becomes
// dqa[FD/16][8] -- and also retires the half-warp idle in the old accumulate
// (`cc = lane - dt*16` left 16 of 32 lanes with nothing to do).
//
// **AND IT LOSES TOO, on REGISTERS: 2.697 ms against 2.456, with one template
// instantiation going from 96 registers to 222 and no spills.**  At 128
// threads that is 2 blocks an SM where there were 3.  dq[16] and dqa[2][8]
// hold the same sixteen floats, so the cost is not the array -- it is that
// the accumulator `c` must stay live until the adds, where the old code
// stored it to shared and freed it at once.  Hoisting the three-way
// destination branch out of the epilogue moved it 2.780 -> 2.697 and taking
// the unroll off the epilogue moved it not at all; 222 both times.
//
// So of the five round trips exactly ONE was removable, and the rule from the
// softmax holds in all four of the others: the conversion is free only when
// the consumer is shared memory.  dV and dK feed global ATOMICS, dS feeds the
// dnbb atomics, and dQ feeds registers the accumulator cannot reach without
// keeping itself alive.
#ifndef FB_REGDQ
#define FB_REGDQ 0
#endif
// FB_REGDS failed on its OUTPUTS, not on the conversion -- so split them.
// Compute dS in the accumulator's layout (retiring the dP round trip) and
// write it to Ps, which is shared and therefore free; then make the dnbb
// atomics and the optional dS store a SECOND pass that reads Ps back in the
// coalesced lane == key layout they want.  That buys the round trip without
// paying the scattered atomics.
//
// **AND IT IS A WASH, slightly negative: 2.526 ms against 2.475 over four
// reps each.**  The second pass over Ps costs what the removed round trip
// saves.  So all five round trips are now measured and exactly one -- the
// softmax, whose only consumer is shared -- was worth retiring.
#ifndef FB_REGDS2
#define FB_REGDS2 0
#endif
// The matrix_a fragments for S (from Qs) and dP (from dOs) depend only on kk,
// not on nt, so the nt loop reloads each of them BN/16 times.  Hoisting costs
// (FD/16) fragments of 4 registers.
//
// **AND IT IS 5.4% WORSE: 2.596 ms against 2.462, three reps each.**  The
// fragments cost more held than reloaded, and the compiler was almost
// certainly already common-subexpressing them out of the nt loop.
// **BOTH MEASURED AND BOTH NEGATIVE.**  The barriers cost NOTHING: replacing
// all 40 of them with warp-wide ones measures 2.512 ms against 2.500 over
// three reps, i.e. the block-wide synchronisation is already hidden behind
// other latency.  Unrolling the key loop by two is 2.527, +1%.  Recorded
// because "too many barriers" and "not enough ILP" are the two things one
// guesses about a kernel like this, and neither is true here.
//
// PROBE: turn the five BLOCK-wide barriers in the j0 loop into warp-wide ones.
// Wrong across warps (dV and dK each read every warp's rows), identical
// instruction count and traffic, so the delta is what block-wide
// synchronisation costs -- 5 a tile x 8 tiles = 40 a block, the one axis of
// this kernel nobody had priced.
#ifndef PROBE_BAR
#define PROBE_BAR 0
#endif
// PROBE: this kernel issues FORTY load_matrix_sync a warp a key tile against
// TWENTY mma -- a 2:1 ratio, worse than combineDx's 1.25:1 -- and a WMMA
// fragment load from shared is several LDS ops where `ldmatrix` is one.  So
// the kernel may be LDS-INSTRUCTION bound, which is the axis the declined
// mma.sync price never covered (that was occupancy and bank conflicts).
// Pinning the u index in dV's and dK's loads makes them loop-invariant, so
// the compiler hoists them: 12 of the 40 loads go, for 2 extra fragments.
// Wrong math, same mma count.  If this is worth >10%, ldmatrix is worth
// writing; if not, the rewrite is dead for a measured reason.
//
// **IT IS WORTH NOTHING: 2.484 ms against 2.471 over three reps.**  Removing
// thirty per cent of the fragment loads does not move the kernel, so it is
// not LDS-instruction bound, and `ldmatrix` -- whose whole benefit is fewer
// LDS instructions -- has nothing to collect.  combineDx said the same more
// weakly (`PROBE_LDS`, 8.5%).  **So the mma.sync rewrite is dead for the
// right reason now.**  It was declined earlier at 1.7%, but that number came
// from an occupancy-and-bank-conflict experiment, which is a different axis;
// this is the one that mattered and it is zero.
#ifndef PROBE_FBLDS
#define PROBE_FBLDS 0
#endif
#if PROBE_BAR
#define FB_BAR() __syncwarp()
#else
#define FB_BAR() __syncthreads()
#endif
// Unrolling the key loop by two, for instruction-level parallelism across
// iterations.  Costs nothing but code size if the scheduler cannot use it.
#ifndef FB_UNROLLJ
#define FB_UNROLLJ 1
#endif

#ifndef FB_HOISTA
#define FB_HOISTA 0
#endif
// These two sit OUTSIDE the namespace and describe the mma fragment's lane
// layout, which does not depend on the head dim -- so unlike everything below
// they are defined once however many widths this body is included at.
#ifndef FB_FRAG_LAYOUT_ONCE
#define FB_FRAG_LAYOUT_ONCE
__device__ __forceinline__ int fragRow(int lane, int e) {
  return (lane >> 2) + 8 * ((e >> 1) & 1);
}
__device__ __forceinline__ int fragCol(int lane, int e) {
  return (lane & 3) * 2 + (e & 1) + 8 * (e >> 2);
}
#endif

namespace FB_NS {

// THIS KERNEL IS 34% OF THE WHOLE STACK'S BACKWARD -- 199.6 ms of 592 at N=256
// depth 48, four times the next kernel -- so it is the only place in the
// backward where a rewrite could matter.  What it is NOT is slow for its
// shape.  Counting the five matmuls it runs (QK^T recomputed, P^T dO, dO V^T,
// dS K, dS^T Q) at N_b=256, H=4, N=256, FD=32 that is 21.5 GFLOP a call, so
// 2.079 ms a call is **10.3 TFLOP/s**.  For scale, on the same shape:
//
//   this backward            10.3 TFLOP/s
//   localfold's tuned FORWARD 6.4   (8.59 GFLOP in 1.338 ms)
//   XLA's Triton, fwd+bwd     4.2   (30.1 GFLOP in 7.156 ms)
//
// 10.3 is 8% of the A10's 125 TFLOP/s fp16 peak and it is still the fastest
// thing measured on this shape, including a mature autotuned Triton kernel and
// localfold's own hand-tuned forward.  **FD = 32 is why.**  A 16x16x16 wmma
// tile gets two k-steps out of a 32-wide head before it has to store the
// accumulator and reload it in the "lane == key" layout the softmax needs; the
// MMA pipeline never fills, and the shared round-trip -- five of them a tile --
// is the kernel.  Nobody beats this shape by tuning; it would take a different
// instruction (mma.sync with register-resident fragments, cp.async double
// buffering) and even then the ceiling is set by FD, not by the code.
//
// WHERE ITS TIME GOES, by ablation at N=256 depth 48 (`NO_DKV`, `NO_DNBB`):
//
//   total                      199.6 ms
//   dK/dV atomics               44.4   22%
//   dnbb atomics                20.3   10%
//   the MMA + softmax core     134.9   68%
//
// so even retiring EVERY atomic -- which is all the persistent-block rewrite
// below would buy -- leaves two thirds of it.
//
// OCCUPANCY CANNOT BE RAISED, and this is now measured rather than estimated.
// -Xptxas -v says 127-128 registers a thread and no spills, so at 128 threads
// the register file would allow 4 blocks an SM; shared memory allows 3
// (31.2 KB of the 100 KB an SM) and is the binding constraint, giving 12 warps
// of 48 -- 25%.  An earlier note here said "getting under 25 KB a block would
// buy the fourth block".  IT WOULD NOT.  `cudaOccupancyMaxActiveBlocks` on the
// real kernel returns 3 at LDH=32 LDF=36 (26112 B) AND at the floor LDH=LDF=32
// (25088 B), with and without the 100 KB carveout -- there is a per-block
// reserve the arithmetic missed.  LDH cannot go below FD=32 and LDF cannot go
// below BN=32, so 25088 B IS the floor for this buffer set and the 4th block is
// unreachable without deleting a buffer.  Ss is the only candidate (10.24 KB,
// float [BM][LDF]) and it exists because a wmma accumulator fragment's lane
// layout does not match "lane == key"; making it half would need half-precision
// MMA accumulation, which is a numerics change for ~8 points of occupancy.
//
// **AND THE mma.sync REWRITE IS NOT WORTH WRITING.  MEASURED.**  The obvious
// reading of the table below is "the 4th block loses only because the padding
// floor it needs costs bank conflicts -- so swizzle the layout, which is
// conflict-free with no padding, and collect the occupancy for free".  That
// needs `ldmatrix` + `mma.sync` instead of load_matrix_sync, i.e. a rewrite of
// this kernel, so it is worth pricing the prize first.
//
// FB_PAD_SMEM asks the launch for bytes the kernel does not use, which holds
// the block count down and separates the two effects the LDH/LDF sweep
// conflates.  Module backward at N=256:
//
//   LDH LDF  pad    blocks   bwd ms
//    40  40    0       3     2.980     conflict-free baseline
//    32  32 6144       3     3.564     conflicts alone:  +19.6%
//    32  32    0       4     3.376     the 4th block:     -5.3%
//
// **The extra block is worth 5.3%.  The conflicts that buying it costs are
// worth 19.6%.**  A swizzled four-block kernel would therefore land at
// 2.980 - 0.188 = 2.792 ms, -6.3% -- which on flashBwd's 194.5 ms of the
// stack's backward is 12 ms, 1.7%.  Days of work on the most delicate kernel
// here, for 1.7%, against a history of two wrong-gradient incidents in it.
// NO.
//
// (An earlier estimate of "up to 86 ms" came from comparing this kernel's
// 10.6 TFLOP/s with our own flash FORWARD's 19.2 on the same shape and
// assuming occupancy was the gap.  It is not: occupancy is 5% of it.  The
// rest is the serial dependency chain inside a warp -- five
// store_matrix_sync -> shared -> reload round trips a tile -- which more
// resident warps only partly hide.)
//
// **CORRECTION, MEASURED: THE 4TH BLOCK *IS* REACHABLE -- AND IT LOSES.**
// The note above says cudaOccupancyMaxActiveBlocks returns 3 "at the floor
// LDH=LDF=32 (25088 B)" and concludes the 4th block is unreachable without
// deleting a buffer.  Asking the driver at 128-byte granularity instead of
// at the two points it happened to try, the step is at **24576 B**:
//
//     25088 -> 3 blocks        24576 -> 4 blocks (16 warps, not 12)
//
// and 25088 - 24576 = 512 is exactly `lseS[BM] + DS[BM]`, the two per-query
// scalars staged in shared.  Deleting them (reading lse and D from global
// where they are used -- 64 rows x 8 key steps a block, which L1 absorbs)
// does reach four blocks.  Module backward at N=256:
//
//     LDH LDF   smem    blocks   bwd ms
//      40  40   30720      3     3.324      <- 2.973 WITH the staging
//      32  40   27136      3     3.541
//      40  32   29184      3     3.429
//      32  32   24576      4     3.701      <- the 4th block, and the worst
//
// Two separate findings.  Staging lse/D is worth 12% on its own (2.973 ->
// 3.324) -- a per-element global read inside the softmax loop is not free
// even out of L1.  And the padding floor the 4th block requires costs more
// in bank conflicts than +33% occupancy returns.  **Occupancy is not what
// this kernel is short of**, which is the same answer the BM sweep gave and
// the same answer projGateK gave to four different occupancy experiments.
//
// THE PERSISTENT-BLOCK REWRITE IS NOT WORTH IT, and the register count says so
// without writing it.  Giving one block a whole (b, h) would hold dK/dV for
// every key in registers and retire the atomics outright -- 256 keys x 32 dims
// x 2 tensors over 256 threads is 64 more registers a thread, so ~192 total,
// 49152 a block, and ONE block an SM: 8 warps against today's 12.  Losing a
// third of the occupancy on a latency-bound kernel to remove 22% of its time
// is a wash before counting the risk of spilling.
//
// BM IS GENERIC NOW, AND 64 IS STILL THE BEST VALUE.  The dV and dK sections
// map warp -> (key tile, dim tile), and that tile count is (BN/16)*(FD/16) = 4
// -- fixed by the KEY tile, not the query tile -- so they used to require
// exactly WARPS == 4 and at BM=128 warps 4..7 indexed past the end: grad/pair
// cos 0.186 and NaN out of MSA row attention.  TILES/SPLIT below fixes that
// properly, and BM=128 is now bit-for-bit as correct as BM=64 (grad/pair cos
// 0.999970716 at N=256, identical digits).
//
// **RETRACTED: "BM=128 measures 2.45x faster (3.44 -> 1.40 ms)".**  That was a
// timing of the BROKEN kernel and it is not reproducible once the kernel is
// right.  Module backward at N=256, swept over the paddings:
//
//   BM   LDH  LDF   smem      bwd ms
//   64    40   40   30.5 KB   2.891   <- shipped
//   64    32   36   25.5      3.243
//  128    32   36   47.0      3.075   <- best BM=128
//  128    32   40   49.0      3.239
//  128    40   40   56.0      3.283
//  128    40   36   54.0      3.555
//  128    32   32   45.0      4.105
//
// Every BM=128 row loses, because the wider tile costs more occupancy (56 KB is
// 1 block an SM, 47 KB is 2) than the halved K/V loads and halved block count
// return.  The generalisation is kept anyway: it removes a landmine that
// returned a plausible-looking wrong gradient, and it makes BM a constant
// anyone can A/B with -DFB_BM instead of a kernel rewrite.
#ifndef FB_BM
#define FB_BM 64
#endif
static const int BM = FB_BM; // queries a block -- -DFB_BM=128 to A/B it
static const int BN = 32;    // keys a step -- see the static_assert
static const int FD = FB_FD;  // head dim
#ifndef FB_LDH
// THE FLOOR IS max(FD, BN), NOT FD.  Qs/dOs/Ks/Vs are FD wide but Ps shares
// the same leading dimension and holds the BN scores of a row (indexed
// `Ps[row * LDH + lane]`, lane over the keys).  At FD == BN == 32 the two
// floors coincide and the distinction never showed.
#define FB_LDH ((FB_FD > 32 ? FB_FD : 32) + 8)
#endif
#ifndef FB_LDF
#define FB_LDF 40
#endif
static const int LDH = FB_LDH; // f16 ldm, >= FD and a multiple of 8.  32 saves
static const int LDF = FB_LDF; // 5 KB, does NOT change the block count (see the
                               // occupancy note above) and costs 9-26% in bank
                               // conflicts.  -DFB_LDH / -DFB_LDF to sweep.
static const int WARPS = BM / 16;
static const float LOG2E_ = 1.4426950408889634f;

// THE dV AND dK SECTIONS DO NOT SCALE WITH BM, which is what pinned BM at 64.
// Their output is [BN keys][FD], i.e. (BN/16)*(FD/16) = TILES tiles, and that
// count is fixed by the KEY tile, not the query tile -- so at WARPS > TILES the
// extra warps have no tile of their own and used to index past the end.
// SPLIT warps now share a tile and each reduces its own slice of the BM
// queries, so every warp works and the only cost is that the partials are
// accumulated separately: the atomic count per tile goes back up by SPLIT,
// which is exactly the count BM=64 already paid.  Everything else -- the K/V
// shared loads, the block count, the S/P/dS/dQ passes -- halves.
static const int TILES = (BN / 16) * (FD / 16);
static const int SPLIT = WARPS / TILES;
// SOFTWARE PIPELINE.  The j0 loop opened by loading the K/V tile from global
// straight into shared and then waiting on it at a barrier, so every one of
// the N/BN iterations paid the full global latency with nothing to overlap.
// PROBE_KV (pin the tile to j0 = 0 so it comes out of L1 -- wrong arithmetic,
// identical instruction count) measured that exposure at 2.852 -> 2.506 ms,
// **12% of the module backward**.
//
// cp.async double buffering is the textbook fix and it does not fit: a second
// K/V pair is 5 KB, 30720 -> 35840, and 3 x 35840 > the 100 KB an SM, so it
// would buy latency hiding with a third of the occupancy.  Registers are free
// here instead -- each thread already owns exactly RPT elements of each tile,
// and holding them across the compute body costs 8 registers against a 127
// baseline that has 43 to spare before the 3-block limit bites.
static const int RSW = (32 * WARPS) / FD;       // tile rows a sweep
static const int RPT = (BN + RSW - 1) / RSW;    // sweeps, hence registers
//
// **AND IT IS A WASH -- SO THE 12% WAS NEVER LATENCY.**  Bit-identical
// gradient, 2.739 ms against 2.723.  The arithmetic that should have come
// before the code: 8 tiles x BN x FD x 2 tensors x 2 B = 32 KB a block over
// 4096 blocks is 128 MB a call, and 128 MB at this card's measured 489 GB/s
// is 0.26 ms -- essentially the whole 0.346 ms the probe returned.  PROBE_KV
// removed BYTES, not latency, and no amount of prefetching moves bytes.
// Kept at 0: it is correct and costs nothing to re-try on a card with a
// different L2, but it buys nothing here.
#ifndef FB_PREFETCH
#define FB_PREFETCH 0
#endif
static_assert(WARPS >= TILES && WARPS % TILES == 0,
              "a dV/dK tile is owned by a whole number of warps");
static_assert((BM / 16) % SPLIT == 0,
              "the BM reduction must divide evenly among the warps sharing a tile");

// The softmax and dS passes give one key to one lane, so a key tile IS a warp.
// BN = 64 compiles, runs, and returns NaN.
static_assert(BN == 32, "a key tile is one lane per key across a warp");
static_assert(BM % 16 == 0 && BN % 16 == 0, "whole wmma tiles");

// D_i = sum_d dO[i,d] * O[i,d], the term the softmax backward subtracts.
//
// `B` is the BATCH count and is not N.  It is N for triangle attention, where
// the batch axis is the pair's first index, and the alignment depth (1-2) for
// MSA row attention.  Deriving it as N here read 128x past the end of a
// 2-row MSA's arrays and still produced a plausible-looking answer -- forward
// cos 0.991.
__global__ void rowDotK(half* __restrict__ dO, const half* __restrict__ O,
                        float* __restrict__ Dv, int N, int H, int Dh, int B,
                        const float* __restrict__ lscale = nullptr) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t rows = (size_t)B * N * H;
  if (t >= rows) return;
  int h = (int)(t % H);
  size_t bi = t / H;              // b*N + i
  const int W = H * Dh;
  const float sc = lscale ? lscale[0] : 1.f;
  float s = 0.f;
  for (int d = 0; d < Dh; ++d) {
    size_t o = bi * W + h * Dh + d;
    half dv = dO[o];
    if (lscale) {
      dv = __float2half(__half2float(dv) * sc);
      dO[o] = dv;
    }
    s += __half2float(dv) * __half2float(O[o]);
  }
  int b = (int)(bi / N), i = (int)(bi % N);
  Dv[((size_t)b * H + h) * N + i] = s;
}

// NBB_ATOMIC: accumulate dnbb straight from the tile instead of materialising
// dS for a separate reduction.  dS is read by nothing else, so if the atomics
// are cheaper than a 134 MB write plus a 134 MB read, dS need not exist at all.
// The contention is N_b deep (every batch row hits the same [h][i][j]) against
// the 2-4 of dK/dV, which is why it has to be measured rather than assumed.
// 2^40.  Resolution 9.1e-13 against dnbb values that live around 1e-3 to
// 1e-6, so the fixed point carries far more than the bf16 the gradient is
// stored in; and the int64 range saturates only at a SUM of 8.4e6, which a
// finite gradient does not reach.  `nbbSat` counts any term that would have
// overflowed, so the choice cannot fail silently -- it is read and reported
// rather than assumed.
__device__ __forceinline__ long long nbbFixed(float v, unsigned* sat = nullptr) {
  const float scaled = v * 1099511627776.0f;   // 2^40
  if (sat && !(scaled > -9.0e18f && scaled < 9.0e18f)) {
    atomicAdd(sat, 1u);
    return scaled > 0.f ? 9000000000000000000LL : -9000000000000000000LL;
  }
  return (long long)scaled;
}

__global__ void nbbFromFixedK(const long long* __restrict__ src,
                              float* __restrict__ dst, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (float)src[i] * (1.0f / 1099511627776.0f);
}

template <bool DO_DV = true, bool DO_DK = true, bool DO_DS = true,
          bool NBB_ATOMIC = false>
__global__ void __launch_bounds__(32 * WARPS) flashBwdK(
    const half* __restrict__ qkvg, int projLD, const half* __restrict__ bias,
    const half* __restrict__ dO, const float* __restrict__ lse,
    const float* __restrict__ Dv, const float* __restrict__ mask,
    __nv_bfloat16* __restrict__ dS, float* __restrict__ dQ, float* __restrict__ dK,
    float* __restrict__ dV, int N, int H, float* __restrict__ dnbb = nullptr,
    // BF16 OUT: dQ written and dK/dV accumulated straight into the combined
    // gradient at stride `tot`, instead of three f32 buffers that a later pass
    // has to read (100 MB) and narrow (50 MB).  dK/dV use the bf16x2 atomic,
    // so the accumulation rounds per add -- gated, not assumed.
    // DETERMINISTIC dnbb.  `dnbbI` is the same tensor in int64 fixed point:
    // every one of the N pair rows atomicAdds into the same [h][i][j], so
    // with floats the ORDER decides the low bits and two identical runs
    // return different gradients (measured: a design trajectory diverges by
    // step 1).  INTEGER addition IS associative, so the fixed-point sum is
    // the same whatever order the blocks arrive in.  See NBB_SCALE.
    long long* __restrict__ dnbbI = nullptr,
    __nv_bfloat16* __restrict__ dcomb = nullptr, int tot = 0,
    // dQ STRAIGHT OUT AS BF16, separately from dK/dV.  `dcomb` above is an
    // all-or-nothing switch and the measurement that rejected it was about
    // dK/dV: those need ATOMICS, and a bf16x2 atomic into a 516-wide buffer
    // cost the kernel 1.946 -> 3.527 ms.  dQ is different in kind -- it is
    // written ONCE, from registers, with no atomic at all -- so it can land in
    // the combined gradient directly and skip an f32 buffer that a later pass
    // would have to read back and narrow.  51 MB a call.
    __nv_bfloat16* __restrict__ dQb = nullptr, int qtot = 0) {
  extern __shared__ char smem[];
  half* Qs = (half*)smem;                      // [BM][LDH]
  half* dOs = Qs + BM * LDH;                   // [BM][LDH]
  half* Ks = dOs + BM * LDH;                   // [BN][LDH]
  half* Vs = Ks + BN * LDH;                    // [BN][LDH]
  half* Ps = Vs + BN * LDH;                    // [BM][LDH]   (reused as dS tile)
  float* Ss = (float*)(Ps + BM * LDH);         // [BM][LDF]
  float* lseS = Ss + BM * LDF;                 // [BM]
  float* DS = lseS + BM;                       // [BM]

  int it = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  int i0 = it * BM;
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int W = H * FD, W4 = 4 * W;
  size_t M = (size_t)N * N;

  const int FD8 = FD / 8;
  for (int u = tid; u < BM * FD8; u += 32 * WARPS) {
    int r = u / FD8, d = (u % FD8) * 8, i = i0 + r;
    bool ok = i < N;
    *(uint4*)(Qs + r * LDH + d) =
        ok ? *(const uint4*)(qkvg + ((size_t)b * N + i) * projLD + h * FD + d)
           : make_uint4(0, 0, 0, 0);
    *(uint4*)(dOs + r * LDH + d) =
        ok ? *(const uint4*)(dO + ((size_t)b * N + i) * W + h * FD + d)
           : make_uint4(0, 0, 0, 0);
  }
  for (int r = tid; r < BM; r += 32 * WARPS) {
    int i = i0 + r;
    if (!FB_REGSM) lseS[r] = i < N ? lse[((size_t)b * H + h) * N + i] : 0.f;
    DS[r] = i < N ? Dv[((size_t)b * H + h) * N + i] : 0.f;
  }
  const int row_e0 = warp * 16 + (lane >> 2), row_e2 = row_e0 + 8;
  const int i_e0 = i0 + row_e0, i_e2 = i0 + row_e2;
  const float lse_e0 = (FB_REGSM && i_e0 < N) ? lse[((size_t)b * H + h) * N + i_e0] : 0.f;
  const float lse_e2 = (FB_REGSM && i_e2 < N) ? lse[((size_t)b * H + h) * N + i_e2] : 0.f;
  const half* bias_e0 = bias + ((size_t)h * N + i_e0) * N;
  const half* bias_e2 = bias + ((size_t)h * N + i_e2) * N;
  // dQ stays in registers: lane == channel, 16 rows a warp (the forward's lesson
  // -- shared memory is what caps occupancy in these kernels)
#if FB_REGDQ
  float dqa[FD / 16][8];
#pragma unroll
  for (int t = 0; t < FD / 16; ++t)
#pragma unroll
    for (int e = 0; e < 8; ++e) dqa[t][e] = 0.f;
#else
  float dq[16];
#pragma unroll
  for (int r = 0; r < 16; ++r) dq[r] = 0.f;
#endif
  __syncthreads();

  using namespace nvcuda::wmma;
  // the tile this thread owns, held across the compute body of the PREVIOUS
  // iteration so its global latency lands under real work
  half rk[RPT], rv[RPT];
#define FB_LOAD_KV(J0)                                                         \
  _Pragma("unroll") for (int u = 0; u < RPT; ++u) {                            \
    int r = tid / FD + u * RSW, d = tid % FD, j = (J0) + r;                    \
    if (PROBE_KV) j = r;                                                       \
    bool ok = r < BN && j < N;                                                 \
    rk[u] = ok ? qkvg[((size_t)b * N + j) * projLD + W + h * FD + d]           \
               : __float2half(0.f);                                            \
    rv[u] = ok ? qkvg[((size_t)b * N + j) * projLD + 2 * W + h * FD + d]       \
               : __float2half(0.f);                                            \
  }
  if (FB_PREFETCH) FB_LOAD_KV(0);

#if FB_UNROLLJ > 1
#pragma unroll 2
#endif
  for (int j0 = 0; j0 < N; j0 += BN) {
    if (FB_PREFETCH) {
#pragma unroll
      for (int u = 0; u < RPT; ++u) {
        int r = tid / FD + u * RSW, d = tid % FD;
        if (r < BN) { Ks[r * LDH + d] = rk[u]; Vs[r * LDH + d] = rv[u]; }
      }
    } else {
#pragma unroll
      for (int u = tid; u < BN * FD8; u += 32 * WARPS) {
        int r = u / FD8, d = (u % FD8) * 8, j = j0 + r;
        if (PROBE_KV) j = r;
        bool ok = j < N;
        *(uint4*)(Ks + r * LDH + d) =
            ok ? *(const uint4*)(qkvg + ((size_t)b * N + j) * projLD + W + h * FD + d)
               : make_uint4(0, 0, 0, 0);
        *(uint4*)(Vs + r * LDH + d) =
            ok ? *(const uint4*)(qkvg + ((size_t)b * N + j) * projLD + 2 * W + h * FD + d)
               : make_uint4(0, 0, 0, 0);
      }
    }
    if (FB_REGSM && tid < BN) {
      int j = j0 + tid;
      lseS[tid] = j < N ? 1e9f * (mask[(size_t)b * N + j] - 1.f) : 0.f;
    }
    FB_BAR();
    // issued here, consumed at the top of the next iteration: everything
    // between is cover
    if (FB_PREFETCH && j0 + BN < N) FB_LOAD_KV(j0 + BN);

    // S = Q K^T, in the log2 domain the forward used
    fragment<matrix_a, 16, 16, 16, half, row_major> qa[FB_HOISTA ? FD / 16 : 1];
    if (FB_HOISTA)
#pragma unroll
      for (int kk = 0; kk < FD / 16; ++kk)
        load_matrix_sync(qa[kk], Qs + (size_t)warp * 16 * LDH + kk * 16, LDH);
    for (int nt = 0; nt < BN / 16; ++nt) {
      fragment<accumulator, 16, 16, 16, float> c;
      fill_fragment(c, 0.f);
#pragma unroll
      for (int kk = 0; kk < FD / 16; ++kk) {
        fragment<matrix_a, 16, 16, 16, half, row_major> a;
        fragment<matrix_b, 16, 16, 16, half, col_major> bk;
        if (!FB_HOISTA) load_matrix_sync(a, Qs + (size_t)warp * 16 * LDH + kk * 16, LDH);
        load_matrix_sync(bk, Ks + (size_t)nt * 16 * LDH + kk * 16, LDH);
        mma_sync(c, FB_HOISTA ? qa[kk] : a, bk, c);
      }
      if (FB_REGSM) {
        // P = exp2(S*log2e + bias - lse), in the accumulator's own layout.
        // Two elements a step: e and e+1 are adjacent columns, so this is a
        // half2 store and four of them cover the lane's eight values.
        // e=0 and e=2 share cc=(lane&3)*2; e=4 and e=6 share cc=(lane&3)*2+8.
        const bool evenN = (N & 1) == 0;
#pragma unroll
        for (int ebase = 0; ebase < 8; ebase += 4) {
          int cc = fragCol(lane, ebase);
          int j = j0 + nt * 16 + cc;
          float2 m2 = *(const float2*)(lseS + nt * 16 + cc);
          float mk0 = m2.x, mk1 = m2.y;
#pragma unroll
          for (int ed = 0; ed < 4; ed += 2) {
            int e = ebase + ed;
            int row = ed ? row_e2 : row_e0;
            int i = ed ? i_e2 : i_e0;
            float lse_r = ed ? lse_e2 : lse_e0;
            const half* bias_r = ed ? bias_e2 : bias_e0;
            float p0 = 0.f, p1 = 0.f;
            if (i < N && j < N) {
              if (evenN) {
                half2 b2 = *(const half2*)(bias_r + j);
                float sv0 = c.x[e] * LOG2E_ + __low2float(b2) + mk0 * LOG2E_;
                p0 = exp2f(sv0 - lse_r);
                float sv1 = c.x[e + 1] * LOG2E_ + __high2float(b2) + mk1 * LOG2E_;
                p1 = exp2f(sv1 - lse_r);
              } else {
                float sv0 = c.x[e] * LOG2E_ + __half2float(bias_r[j]) + mk0 * LOG2E_;
                p0 = exp2f(sv0 - lse_r);
                if (j + 1 < N) {
                  float sv1 = c.x[e + 1] * LOG2E_ + __half2float(bias_r[j + 1]) + mk1 * LOG2E_;
                  p1 = exp2f(sv1 - lse_r);
                }
              }
            }
            *(half2*)(Ps + (size_t)row * LDH + nt * 16 + cc) =
                __floats2half2_rn(p0, p1);
          }
        }
      } else {
        FB_STORE(Ss + (size_t)warp * 16 * LDF + nt * 16, c, LDF, mem_row_major);
      }
    }
    __syncwarp();

    if (!FB_REGSM) {
      // P = exp2(S*log2e + bias - lse); the bias is already in the log2 domain
#pragma unroll
      for (int r = 0; r < 16; ++r) {
        int row = warp * 16 + r, i = i0 + row, j = j0 + lane;
        float p = 0.f;
        if (i < N && j < N) {
          float s = Ss[(size_t)row * LDF + lane] * LOG2E_ +
                    __half2float(bias[((size_t)h * N + i) * N + j]) +
                    1e9f * (mask[(size_t)b * N + j] - 1.f) * LOG2E_;
          p = exp2f(s - lseS[row]);
        }
        Ps[(size_t)row * LDH + lane] = __float2half(p);
      }
    }
    FB_BAR();   // dV reads every warp's rows of P, not just its own

    // dV += P^T dO, while P is still live.  The output is [BN keys][FD], which
    // is (BN/16) x (FD/16) tiles -- one per warp -- and each reduces over every
    // query in the block, NOT over a slice of them.  Mapping a warp to a query
    // tile here (as the S product does) would walk off the end of a 32-key tile.
    {
      const int tile = warp % TILES, part = warp / TILES;
      const int kt = tile / (FD / 16), dt = tile % (FD / 16);
      fragment<accumulator, 16, 16, 16, float> c;
      fill_fragment(c, 0.f);
#pragma unroll
      for (int u = 0; u < BM / 16 / SPLIT; ++u) {
        const int kk = part * (BM / 16 / SPLIT) + u;
        fragment<matrix_a, 16, 16, 16, half, col_major> a;   // P^T
        fragment<matrix_b, 16, 16, 16, half, row_major> bb;
        const int lk = PROBE_FBLDS ? part * (BM / 16 / SPLIT) : kk;
        FB_LOADPS(a, Ps + (size_t)lk * 16 * LDH + kt * 16, LDH);
        load_matrix_sync(bb, dOs + (size_t)lk * 16 * LDH + dt * 16, LDH);
        mma_sync(c, a, bb, c);
      }
      FB_BAR();
      FB_STORE(Ss + (size_t)warp * 16 * LDF, c, LDF, mem_row_major);
      __syncwarp();
      // only N/BM query tiles contribute, so this is 2-4 deep, not N deep
      if (DO_DV) {
        const int cc = lane & 15, r_base = lane >> 4;
        const size_t col_off = (size_t)h * FD + dt * 16 + cc;
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          int rr = r_base + u * 2, j = j0 + kt * 16 + rr;
          if (j < N) {
            float g = Ss[(size_t)(warp * 16 + rr) * LDF + cc];
            if (dcomb) {
              if ((cc & 1) == 0) {
                __nv_bfloat162 v = __floats2bfloat162_rn(
                    g, Ss[(size_t)(warp * 16 + rr) * LDF + cc + 1]);
                atomicAdd((__nv_bfloat162*)&dcomb[((size_t)b * N + j) * tot +
                                                  2 * W + col_off], v);
              }
            } else if (g != 0.f) {
              atomicAdd(&dV[((size_t)b * N + j) * W + col_off], g);
            }
          }
        }
      }
      __syncwarp();
    }

    // dP = dO V^T
    fragment<matrix_a, 16, 16, 16, half, row_major> da[FB_HOISTA ? FD / 16 : 1];
    if (FB_HOISTA)
#pragma unroll
      for (int kk = 0; kk < FD / 16; ++kk)
        load_matrix_sync(da[kk], dOs + (size_t)warp * 16 * LDH + kk * 16, LDH);
    for (int nt = 0; nt < BN / 16; ++nt) {
      fragment<accumulator, 16, 16, 16, float> c;
      fill_fragment(c, 0.f);
#pragma unroll
      for (int kk = 0; kk < FD / 16; ++kk) {
        fragment<matrix_a, 16, 16, 16, half, row_major> a;
        fragment<matrix_b, 16, 16, 16, half, col_major> bv;
        if (!FB_HOISTA) load_matrix_sync(a, dOs + (size_t)warp * 16 * LDH + kk * 16, LDH);
        load_matrix_sync(bv, Vs + (size_t)nt * 16 * LDH + kk * 16, LDH);
        mma_sync(c, FB_HOISTA ? da[kk] : a, bv, c);
      }
      if (FB_REGDS2) {
        // dS into Ps only; the global outputs are a coalesced pass below
#pragma unroll
        for (int e = 0; e < 8; e += 2) {
          int rr = fragRow(lane, e), cc = fragCol(lane, e);
          int row = warp * 16 + rr, i = i0 + row, j = j0 + nt * 16 + cc;
          half2* slot = (half2*)(Ps + (size_t)row * LDH + nt * 16 + cc);
          half2 pv = *slot;
          float v0 = 0.f, v1 = 0.f;
          if (i < N) {
            if (j < N) v0 = __low2float(pv) * (c.x[e] - DS[row]);
            if (j + 1 < N) v1 = __high2float(pv) * (c.x[e + 1] - DS[row]);
          }
          *slot = __floats2half2_rn(v0, v1);
        }
      } else if (FB_REGDS) {
        // dS = P * (dP - D) in the accumulator's layout; P comes back out of
        // its own shared slot, which this then overwrites with dS.  Each lane
        // owns its elements, so the read-then-write needs no barrier.
#pragma unroll
        for (int e = 0; e < 8; e += 2) {
          int rr = fragRow(lane, e), cc = fragCol(lane, e);
          int row = warp * 16 + rr, i = i0 + row, j = j0 + nt * 16 + cc;
          half2* slot = (half2*)(Ps + (size_t)row * LDH + nt * 16 + cc);
          half2 pv = *slot;
          float v0 = 0.f, v1 = 0.f;
          if (i < N) {
            if (j < N) v0 = __low2float(pv) * (c.x[e] - DS[row]);
            if (j + 1 < N) v1 = __high2float(pv) * (c.x[e + 1] - DS[row]);
            if (DO_DS) {
              size_t o = (((size_t)b * H + h) * N + i) * N + j;
              if (j < N) dS[o] = __float2bfloat16(v0);
              if (j + 1 < N) dS[o + 1] = __float2bfloat16(v1);
            }
            if (NBB_ATOMIC) {
              size_t o = ((size_t)h * N + i) * N + j;
              if (dnbbI) {
                if (j < N) {
                  unsigned long long fx0 = (unsigned long long)nbbFixed(v0, nullptr);
                  if (fx0) atomicAdd((unsigned long long*)&dnbbI[o], fx0);
                }
                if (j + 1 < N) {
                  unsigned long long fx1 = (unsigned long long)nbbFixed(v1, nullptr);
                  if (fx1) atomicAdd((unsigned long long*)&dnbbI[o + 1], fx1);
                }
              } else {
                if (j < N && v0 != 0.f) atomicAdd(&dnbb[o], v0);
                if (j + 1 < N && v1 != 0.f) atomicAdd(&dnbb[o + 1], v1);
              }
            }
          }
          *slot = __floats2half2_rn(v0, v1);
        }
      } else {
        FB_STORE(Ss + (size_t)warp * 16 * LDF + nt * 16, c, LDF, mem_row_major);
      }
    }
    __syncwarp();

    if (FB_REGDS2) {
      // the global half of dS, in the layout it wants: 32 contiguous keys
      if (DO_DS || NBB_ATOMIC) {
#pragma unroll
        for (int r = 0; r < 16; ++r) {
          int row = warp * 16 + r, i = i0 + row, j = j0 + lane;
          if (i < N && j < N) {
            float v = __half2float(Ps[(size_t)row * LDH + lane]);
            if (DO_DS) dS[(((size_t)b * H + h) * N + i) * N + j] = __float2bfloat16(v);
            if (NBB_ATOMIC) {
              const size_t o = ((size_t)h * N + i) * N + j;
              if (dnbbI) {
                unsigned long long fx = (unsigned long long)nbbFixed(v, nullptr);
                if (fx) atomicAdd((unsigned long long*)&dnbbI[o], fx);
              } else if (v != 0.f) atomicAdd(&dnbb[o], v);
            }
          }
        }
      }
    } else if (!FB_REGDS) {
      // dS = P * (dP - D).  P's slot is reused: it is not read again.
      const int j = j0 + lane;
      if (j < N) {
        size_t o_nbb = ((size_t)h * N + i0 + warp * 16) * N + j;
        size_t o_ds = (((size_t)b * H + h) * N + i0 + warp * 16) * N + j;
#pragma unroll
        for (int r = 0; r < 16; ++r, o_nbb += N, o_ds += N) {
          int row = warp * 16 + r, i = i0 + row;
          float v = 0.f;
          if (i < N) {
            float p = __half2float(Ps[(size_t)row * LDH + lane]);
            v = p * (Ss[(size_t)row * LDF + lane] - DS[row]);
            if (DO_DS) dS[o_ds] = __float2bfloat16(v);
            if (NBB_ATOMIC) {
              if (dnbbI) {
                unsigned long long fx = (unsigned long long)nbbFixed(v, nullptr);
                if (fx) atomicAdd((unsigned long long*)&dnbbI[o_nbb], fx);
              } else if (v != 0.f) atomicAdd(&dnbb[o_nbb], v);
            }
          }
          Ps[(size_t)row * LDH + lane] = __float2half(v);
        }
      } else {
#pragma unroll
        for (int r = 0; r < 16; ++r)
          Ps[(size_t)(warp * 16 + r) * LDH + lane] = __float2half(0.f);
      }
    }
    FB_BAR();   // dK reads every warp's rows of dS

    // dQ += dS K   (registers, lane == channel)
    for (int dt = 0; dt < FD / 16; ++dt) {
      fragment<accumulator, 16, 16, 16, float> c;
      fill_fragment(c, 0.f);
#pragma unroll
      for (int kk = 0; kk < BN / 16; ++kk) {
        fragment<matrix_a, 16, 16, 16, half, row_major> a;
        fragment<matrix_b, 16, 16, 16, half, row_major> bk;
        FB_LOADPS(a, Ps + (size_t)warp * 16 * LDH + kk * 16, LDH);
        load_matrix_sync(bk, Ks + (size_t)kk * 16 * LDH + dt * 16, LDH);
        mma_sync(c, a, bk, c);
      }
#if FB_REGDQ
#pragma unroll
      for (int e = 0; e < 8; ++e) dqa[dt][e] += c.x[e];
#else
      FB_STORE(Ss + (size_t)warp * 16 * LDF + dt * 16, c, LDF, mem_row_major);
#endif
    }
#if !FB_REGDQ
    __syncwarp();
    if (lane < FD) {
#pragma unroll
      for (int r = 0; r < 16; ++r)
        dq[r] += Ss[(size_t)(warp * 16 + r) * LDF + lane];
    }
    __syncwarp();
#endif

    // dK += dS^T Q -- the same [BN][FD] shape and the same warp mapping
    {
      const int tile = warp % TILES, part = warp / TILES;
      const int kt = tile / (FD / 16), dt = tile % (FD / 16);
      fragment<accumulator, 16, 16, 16, float> c;
      fill_fragment(c, 0.f);
#pragma unroll
      for (int u = 0; u < BM / 16 / SPLIT; ++u) {
        const int kk = part * (BM / 16 / SPLIT) + u;
        fragment<matrix_a, 16, 16, 16, half, col_major> a;   // dS^T
        fragment<matrix_b, 16, 16, 16, half, row_major> bq;
        const int lk = PROBE_FBLDS ? part * (BM / 16 / SPLIT) : kk;
        FB_LOADPS(a, Ps + (size_t)lk * 16 * LDH + kt * 16, LDH);
        load_matrix_sync(bq, Qs + (size_t)lk * 16 * LDH + dt * 16, LDH);
        mma_sync(c, a, bq, c);
      }
      FB_BAR();
      FB_STORE(Ss + (size_t)warp * 16 * LDF, c, LDF, mem_row_major);
      __syncwarp();
      if (DO_DK) {
        const int cc = lane & 15, r_base = lane >> 4;
        const size_t col_off = (size_t)h * FD + dt * 16 + cc;
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          int rr = r_base + u * 2, j = j0 + kt * 16 + rr;
          if (j < N) {
            float g = Ss[(size_t)(warp * 16 + rr) * LDF + cc];
            if (dcomb) {
              if ((cc & 1) == 0) {
                __nv_bfloat162 v = __floats2bfloat162_rn(
                    g, Ss[(size_t)(warp * 16 + rr) * LDF + cc + 1]);
                atomicAdd((__nv_bfloat162*)&dcomb[((size_t)b * N + j) * tot +
                                                  W + col_off], v);
              }
            } else if (g != 0.f) {
              atomicAdd(&dK[((size_t)b * N + j) * W + col_off], g);
            }
          }
        }
      }
      __syncwarp();
    }
  }

#if FB_REGDQ
  // one write a block, in the accumulator's layout: e and e+1 are adjacent
  // channels, so each pair is one 4-byte store.  The destination is chosen
  // ONCE -- leaving the three-way branch inside the unrolled loops cost 222
  // registers against 96, which is 2 blocks an SM instead of 3.
  {
    __nv_bfloat16* dst16 = dQb ? dQb : dcomb;
    const size_t stride = dQb ? (size_t)qtot : (size_t)tot;
    for (int t = 0; t < FD / 16; ++t)
      for (int e = 0; e < 8; e += 2) {
        int rr = fragRow(lane, e), col = t * 16 + fragCol(lane, e);
        int i = i0 + warp * 16 + rr;
        if (i >= N) continue;
        __nv_bfloat162 v = __floats2bfloat162_rn(dqa[t][e], dqa[t][e + 1]);
        if (dst16)
          *(__nv_bfloat162*)&dst16[((size_t)b * N + i) * stride + h * FD + col] = v;
        else
          *(float2*)&dQ[((size_t)b * N + i) * W + h * FD + col] =
              make_float2(dqa[t][e], dqa[t][e + 1]);
      }
  }
#else
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    int i = i0 + warp * 16 + r;
    // lane == channel, so a head narrower than a warp leaves the upper lanes
    // owning nothing -- unguarded they wrote their zeros into the next head.
    if (i >= N || lane >= FD) continue;
    if (dQb) dQb[((size_t)b * N + i) * qtot + h * FD + lane] = __float2bfloat16(dq[r]);
    else if (dcomb) dcomb[((size_t)b * N + i) * tot + h * FD + lane] = __float2bfloat16(dq[r]);
    else dQ[((size_t)b * N + i) * W + h * FD + lane] = dq[r];
  }
#endif
}

// FB_PAD_SMEM asks the launch for bytes the kernel does not use, purely to
// hold the block count down.  It exists to SEPARATE two things the LDH/LDF
// sweep conflates: the bank-conflict cost of a tighter layout, and the gain
// from the extra block that layout's smaller footprint buys.  Run 32/32 at
// three blocks and the difference from 40/40 is conflicts alone; run it at
// four and the difference from that is occupancy alone.  Which says whether
// a swizzled layout -- conflict-free with no padding, but needing mma.sync
// instead of load_matrix_sync -- is worth writing.
#ifndef FB_PAD_SMEM
#define FB_PAD_SMEM 0
#endif
inline size_t bwdSmem() {
  return (size_t)(BM * LDH * 3 + BN * LDH * 2) * sizeof(half) +
         (size_t)(BM * LDF + 2 * BM) * sizeof(float) + FB_PAD_SMEM;
}

}  // namespace FB_NS
