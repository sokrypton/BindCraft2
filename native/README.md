# A native CUDA forward+backward for AF2's Evoformer, at design shapes

Step 1 of the question "can hand-written CUDA beat XLA for ColabDesign2's
gradient?".  The premise comes from sokrypton/localfold's `native/` tree, which
is a forward-only CUDA AlphaFold 2/3 running 2-5x the JAX it was measured
against.  Design needs a backward, and localfold has none anywhere -- so the
question is whether the same headroom exists on the gradient.

This directory answers that one module at a time.  It is a prototype: nothing
here is wired into ColabDesign2, and the JAX path remains the reference.

## What is being measured, and why it is this

One AF2 Evoformer block, at the shapes the binder preset actually designs at
(`alphafold2_multimer_v3`: msa_channel 256, pair_channel 128, 48 blocks, MSA
depth 2 -- the designed row plus the template row).

Measured on an A10, `--part full` against `--part pair`, bf16, Triton attention:

| N | full block fwd+bwd | pair track alone | pair track's share |
|---|---|---|---|
| 128 | 30.43 ms | 26.70 ms | 88% |
| 192 | 79.30 ms | 72.89 ms | 92% |

So at design depth the MSA track, the outer product mean and all of their
attention are under 12% of the block.  **The pair track is the target**:
triangle multiplication x2, triangle attention x2, pair transition.  That is
five modules to write a backward for, not forty.

## What "optimise the input, not the weights" is actually worth

Design freezes the weights, so the gradient wanted is d(loss)/d(activations)
and nothing else.  That is a stronger statement than it first looks, and it
cashes out in six distinct places.  Three are banked here, three are not.

**Banked.**

1. *Every backward GEMM is `dX = dY W^T` alone*, with no `X^T dY`.  For the
   triangle multiplication that is 21.4 GFLOP of backward against 34.2 with
   weight gradients -- 37% less.  It does not beat XLA by itself (JAX never
   traces the weight-gradient VJP either) but it halves the implementation.
2. *The input of every Linear is dead the moment its forward GEMM retires*,
   because only `dW` would have needed it.  The sharp case is the transition's
   512-wide relu output: a trainer keeps it (134 MB at N=256 in f32) to form
   `dW2`; we only need to know where it was positive, which is one bit each.
3. *Weights are constant, so any amount of setup-time preprocessing is free* --
   the transposed copies, the concatenation of the five projections into one
   `[C, 640]` matrix, the bf16 conversion.  All done once per run, never per
   step.

**Not banked, in descending order of what they are worth.**

4. **The weights are tiny AND constant, so they live in L2 forever.**  The whole
   pair track of one block is about 1 MB in bf16 -- 393 KB for the two triangle
   multiplications, 330 KB for the two attentions, 262 KB for the transition --
   against an A10's 6 MB L2.  The usual reason a fused GEMM+epilogue MMA kernel
   is hard is streaming weights through shared memory fast enough to keep the
   tensor cores fed.  **That cost is zero here**, and it is the one thing cuBLAS
   cannot exploit, because it treats every call as a fresh problem.  This is the
   strongest argument for writing the fused kernels, and it is what the two
   remaining bottlenecks both need: never materialising `pre` (83.9 MB, 31% of
   the triangle multiplication) and never materialising the attention logits
   (134 MB at N=256).
5. *LayerNorm scale and offset fold into the following Linear.*
   `((p-mu)*rstd*g + o) @ W + b` is `((p-mu)*rstd) @ (diag(g) W) + (o @ W + b)`,
   so the norm kernel stops doing a per-element multiply-add and the backward
   stops multiplying by `scale`.  Only legal because both tensors are frozen.
6. *The backward can run at lower precision than the forward.*  A gradient that
   steers a 20-way sequence logit needs less than a structure prediction does.
   An int8 weight copy for the backward GEMMs alone is 2x bf16 on this card, and
   quantising once is only possible because the weights never move.  Changes
   numerics, so it needs its own gate.

**What it does NOT help.**  The triangle multiplication's contraction and every
matmul inside attention are activation x activation, so both operands need a
gradient and nothing is saved.  That is exactly the part that dominates
triangle attention -- and triangle attention is exactly where this port does not
beat XLA.

## Using it from Python

```python
import sys; sys.path.insert(0, '<repo>/dev/native_block')
import backend

m = af2.model('A:109', pdb=target)
backend.enable(m)                  # every AF2 forward and grad now runs native
m.design(seeds=range(8), iters=125)
```

`backend.ready()` returns `(ok, reason)` so a campaign can fall back to XLA
instead of dying, and `enable(m, strict=False)` does that for you. `disable()`
puts the XLA scan back.

The two things that were easy to get wrong, and are now done for you: the
runner is built LAZILY so its parameters do not exist until forced, and
`install()` must be handed the MODEL'S OWN parameters -- a checkpoint loaded
separately makes its safety check compare the file on disk against the file
the weights were written from, which checks nothing.

### Building the library

A fat binary over the cards we actually run on, so one file serves an A10, an
A100 and an H100. 32 s to build, 4.8 MB against 2.8 for one arch:

```
nvcc -O3 -std=c++17 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_90,code=sm_90 \
  --compiler-options -fPIC -shared -I. \
  -I$(python -c 'import jax.ffi;print(jax.ffi.include_dir())') \
  ffi.cu -lcublas -lcublasLt -o libnative_evoformer.so
```

It was previously built `-arch=sm_86` only, which is the A10 -- it would not
have run on the A100s at all. `ready()` checks the device's compute capability
against what the library carries and says which `-gencode` to add.

## Status

**The acceptance test passes, and ONE RUN OF IT IS NOT A MEASUREMENT.**
`design_outcomes.py` runs real PD1 binder design on both backends, 3 seeds x
125 iterations each. XLA is deterministic -- two runs gave bit-identical
tables. Native is not, so here are four draws of arithmetically identical
code:

| | i_con mean | i_con best | wall clock |
|---|---|---|---|
| XLA (twice, identical) | 2.1178 | 1.8502 | 10.8 min |
| native, draw 1 | 2.1129 | 1.7531 | 8.5 min |
| native, draw 2 | 2.5061 | 2.3327 | 8.3 min |
| native, draw 3 | 1.7629 | 1.6586 | 8.4 min |
| native, draw 4 | 2.1823 | 1.8373 | 8.4 min |
| **native mean of 4** | **2.136** | | **1.28x** |

So native and XLA agree in the mean to under 1% on i_con -- the objective
binder design is actually driven by -- while a SINGLE 3-seed run of native
has a spread of about 0.7. Draws 2-4 are the same binary. The timing, by
contrast, repeats to a tenth of a minute.

Why: the backward is not bit-reproducible (`flashBwd` accumulates dK, dV and
`dnbb` with float atomics) and the cuBLASLt autotuner picks plans per process,
so two native processes are two different numerical paths. 125 steps of
gradient descent amplify that into different designs. This is not a defect to
fix -- it is what a non-deterministic backward does to an optimisation -- but
it means **no conclusion may be drawn from one design run against another.**
Draw 2 arriving right after a kernel fusion looked exactly like a regression
and was not; the repeats are what settled it.

### How it scales

One design step (two forwards, one backward), A10, both backends on the same
monomer hallucination, one process per point and peak memory read from NVML
rather than from JAX -- the stack allocates with `cudaMalloc`, which
`jax.memory_stats()` cannot see. `scale_sweep.py`.

| L | xla ms | native ms | speedup | xla GB | native GB |
|---|---|---|---|---|---|
| 64 | 128 | 132 | **0.97x** | 1.99 | 2.34 |
| 96 | 281 | 243 | 1.16x | 2.00 | 2.91 |
| 128 | 480 | 408 | 1.18x | 1.98 | 3.85 |
| 160 | 806 | 646 | 1.25x | 3.06 | 4.81 |
| 192 | 1121 | 889 | 1.26x | 5.04 | 6.67 |
| 224 | 1576 | 1219 | 1.29x | 4.12 | 7.93 |
| 256 | 2077 | 1576 | 1.32x | 5.59 | 11.55 |
| 320 | 3399 | 2541 | 1.34x | 10.13 | 14.69 |
| 384 | 5296 | 3898 | **1.36x** | 17.14 | 18.57* |

**The speedup GROWS with length** -- 0.97x at 64 to 1.36x at 384, monotonic.
At L=64 native is slightly SLOWER: the stack issues many small kernels and a
64-residue block does not fill the card, so launch overhead dominates where
XLA has fused. The crossover is just under L=96. This is the opposite of the
af3 package's shape, where the speedup was a small-size effect that decayed.

**CUDA graphs: a real win standalone, and WRONG in-model.** The 0.97x at
L=64 looked like launch overhead -- the stack issues ~36k kernels a step at
1.9 us each -- so the whole 48-deep stack was captured as one graph. It
works and it is worth having, in the harness:

| | stream | graph |
|---|---|---|
| standalone stack, N=64 | 84.23 ms | **76.18** |
| standalone stack, N=224 | 845.4 | **834.2** |

It is also not correct through the FFI, and it took two failures to learn
how. Capturing JAX's own stream sweeps whatever XLA touches during the
window into the graph: NaN gradient under a perfect forward. Capturing on a
private stream stitched with events fixes that, and a single
forward+backward then agrees to 1.697e-01 against 1.690e-01 -- but a
40-iteration design still diverges, i_con 4.4977 against 2.3456 and loss
7.2404 against 4.2824. Finite, repeatable, far outside noise. So the flag is
`STACK_GRAPH_UNSAFE`, it is off, and it must stay off for design until that
is explained. First suspect: a module enqueueing onto a stream that is not
`st`, which capture would silently omit from the graph.

**Memory is the weak side.** Native runs 1.2-2.1x XLA's peak, because the f32
residual chain doubles the remat save-set. The ratio is worst in the middle
(2.07x at L=256) and narrows at the top only because the keep gets dropped.
*At L=384 the flash-output keep does not fit and is abandoned, so that row is
the no-flash-keep path -- which costs 3.3% and still gives the best speedup in
the table. **Since the keep became graded** (see "the replay's keep space"
below) that length still holds the two cheap tiers, and the row re-measures at
**3881 ms at the same 18.57 GB** -- 17 ms for nothing. Both backends are near the 23.68 GB card at that point;
L=384 is roughly where an A10 stops either way.

Compile is consistently about 0.6x XLA's at every length.

| module | forward | backward | gated | share of the pair track |
|---|---|---|---|---|
| triangle multiplication (out/in) | yes | yes | yes (**2.24x** XLA) | 39% |
| triangle attention (start/end) | yes | yes | yes (**1.15x** XLA) | 44% |
| pair transition | yes | yes | yes (1.71x XLA) | 8% |

The module budget, XLA forward+backward at N=256, uncontended, real design
settings: triangle multiplication 6.295 ms each, triangle attention 7.156 each,
pair transition 2.583, pair track 32.209 in total.

### pair transition: ours against XLA

A10, uncontended, both at the real design settings (bf16, JAX's default matmul
precision).  Forward+backward, ms:

| N | XLA | ours (bf16 cuBLASLt) | ratio |
|---|---|---|---|
| 128 | 0.725 | **0.399** | 1.82x |
| 256 | 2.583 | **1.514** | 1.71x |
| 384 | 5.650 | **3.301** | 1.71x |

At N=256, per step: LN fwd 0.135, gemm1 0.205, gemm2 0.213 | gemm3 0.198,
relu mask 0.409, gemm4 0.211, LN bwd 0.169.  The f32 unfused path is 5.00 ms,
so the bf16 + epilogue work is 3.3x over our own starting point and 1.7x over
XLA.

The remaining floor is ~0.7 ms (compute-bound at ~50 TFLOP/s on 34.4 GFLOP),
so a fully-fused MMA kernel keeping `dr` in registers between gemm3 and gemm4
would roughly halve it again.  The relu mask is already at 491 GB/s on a
600 GB/s card, so it is honest bandwidth, not slack.

### ...and why 1.7x on this module is worth about 3%

The same uncontended measurement, forward+backward at N=256:

| | ms | share of the pair track |
|---|---|---|
| pair track (the five modules) | 32.21 | 100% |
| pair transition | 2.58 | **8.0%** |
| triangle multiplication x2 + triangle attention x2 | ~29.6 | **92%** |

**The module this milestone optimised is a twelfth of the target.**  Taking it
from 2.58 to 1.51 moves the pair track by 3.4%.  The transition was the right
thing to build first -- it is the simplest module, it proved the method and it
found every trap below -- but it is also the shape XLA fuses best, so 1.7x is
plausibly a floor rather than an average.  **Nothing is decided until triangle
multiplication is measured**, and that is where localfold's forward won most.

### triangle multiplication: ours against XLA

Forward+backward, ms.  Both orientations measure the same, as they should --
they are the same work with one operand's indices swapped.

| N | XLA | ours | ratio |
|---|---|---|---|
| 128 | 1.882 | **0.719** | 2.62x |
| 256 | 6.295 | **2.816** | 2.24x |
| 384 | 15.199 | **6.349** | 2.39x |

Against the fp32 JAX reference, bf16: forward cos 0.99998 (relL2 5.3-6.1e-03),
backward cos 0.99997 (relL2 7.3-7.4e-03), stable across all six
size-x-orientation combinations.  No float64 referee is needed at this
tolerance -- JAX's own f32 error here is ~1e-06, three orders below ours.

Per step at N=256 (ms), after the fixes below:

| forward | | backward | |
|---|---|---|---|
| LayerNorm | 0.131 | output gate | (in `gemm dcn`) |
| projection GEMM (128->640) | 0.250 | `gemm dcn` | 0.080 |
| gate + transpose | 0.340 | dcn transpose | 0.080 |
| **contraction** (128 x NxN) | 0.122 | centre norm bwd | 0.181 |
| centre norm (stats + write) | 0.123 | **contraction x2** | 0.229 |
| output GEMM | 0.085 | combine + transpose | 0.522 |
| output gate | 0.103 | `gemm dx` | 0.250 |
| | | LayerNorm bwd | 0.168 |

**The GEMMs are not the problem.**  They run at 35-43 TFLOP/s and are 1.02 of
the 2.82 ms; the contraction itself -- the N^3 term everyone worries about -- is
0.122 ms forward.  Everything else is bandwidth on the 83.9 MB `pre` tensor and
the layout changes around the contraction.  `combine + transpose` at 0.522 ms is
already running at ~385 GB/s of a 600 GB/s card, so it is traffic, not slack:
cutting it means never materialising `pre`, which needs a fused GEMM+gate MMA
kernel.  That is the remaining factor, and it is not written.

### triangle attention: ours against XLA -- and the negative result

Forward+backward, ms.  Both orientations measure the same; the ending node is
the same computation on the transposed pair.

| N | XLA | ours | ratio | what XLA is running |
|---|---|---|---|---|
| 128 | 3.080 | **1.230** | 2.50x | its own einsum |
| 256 | 7.156 | **6.209** | **1.15x** | Triton flash |
| 384 | 20.882 | **18.468** | **1.13x** | Triton flash |

Correct everywhere (forward cos 0.99994, backward cos 0.99987, relL2 1.0-1.6e-02
-- about sqrt(n) * 2^-8 for a chain of n bf16 operations, which is what 8
mantissa bits predict; `bf16_gap.py` compares that against XLA's own bf16).

**This is where the approach runs out.**  Above `flash_attention_min_len` XLA
dispatches to Triton's flash attention, which never materialises the N x N
scores.  We do: 134 MB at N=256, 453 MB at N=384.  Beating that needs a flash
forward AND backward of our own -- the hardest kernel in the model, which
already exists and is well tuned in Triton, and whose backward must also produce
dBias because `nbb` is computed from the pair.  Against XLA's *einsum* path at
N=128 the materialised version wins 2.5x; against flash it does not.

Per step at N=256 after the fixes: ln 0.141 | projection GEMM 0.417 | logits
0.496 | softmax 0.573 | av 0.447 | gate+out 0.208 || dw 0.487 | softmax bwd
0.830 | dq,dk 1.054 | dnbb 0.382 | dx 0.302.  The batched attention GEMMs run at
8-9 TFLOP/s against the 35-43 the other modules' GEMMs reach, because they are
32-wide slices of a 516-wide array -- another thing flash fixes by construction.

### What the three modules add up to

N=256, forward+backward, ms:

| | XLA | ours | |
|---|---|---|---|
| triangle multiplication x2 | 12.59 | 5.63 | 2.24x |
| triangle attention x2 | 14.31 | 12.42 | 1.15x |
| pair transition | 2.58 | 1.51 | 1.71x |
| residual adds etc. (not written) | 2.72 | 2.72 | — |
| **pair track** | **32.21** | **22.29** | **1.45x** |

1.45x, or about 1.65x if the residual adds were absorbed too.  Triangle
attention is 44% of the track and contributes almost nothing, which caps the
whole thing.

### flash_attention_min_len (now 64, was 256)

Applied in the fork.  The guard exists because AF2's MSA COLUMN attention runs
over the alignment depth, 1-2 in design, and the flash kernels reject an axis
that short -- 64 still excludes it (2 < 64) while letting triangle attention in
at 64-255 residues.  Same answer, dumped both ways at N=192: out/pair relL2
3.4e-07, grad/pair 1.9e-07.

### The measurement behind it

`make_config` sets `flash_attention_min_len = 256`, so below 256 residues every
attention falls back to AF2's einsum.  The comment says "below this the einsum
wins anyway".  It does not:

| | min_len=256 | min_len=64 | |
|---|---|---|---|
| pair track, N=128 | 10.849 | **7.617** | 1.42x |
| pair track, N=192 | 29.637 | **17.488** | 1.69x |

The guard exists because AF2's MSA *column* attention runs over the alignment
depth, which is 1-2 in design, and the flash kernels reject an axis that short.
64 still excludes it (2 < 64) while letting triangle attention in, so the guard
keeps doing its job.

**At N=192 that free change is worth 1.69x -- the same as everything this
directory does at N=128.**  Much binder work sits in the 128-255 band.

### Borrowing localfold's flash kernel (the right move, made late)

localfold is Beer-ware and the author is the same person this repository's
design work is for, so there was never a licensing question -- only one I failed
to ask.  `native/af3/src/flash.cuh` is a tuned FlashAttention-2 for EXACTLY this
shape: a fused `[rows][position][4W]` q/k/v/gate projection, a non-batched bias
with a row stride, a pair mask, a `tr` flag that serves the ending node without
transposing the input, and an output that is already gated.
`native/af2/src/evoformer.cuh:844` calls it for AF2's triangle attention
verbatim.  It is vendored here as `lf_flash.cuh` with the notice retained; the
only edit is its include, and `lf_shim.cuh` is the 40 lines of
`common.cuh` it actually uses (STREAM, sigm, CK, the shared-memory opt-in).

Module forward, ms:

| N | materialised | this directory's WMMA kernel | **localfold** | vs XLA's forward |
|---|---|---|---|---|
| 128 | 0.455 | 0.398 | **0.319** | **3.02x** |
| 256 | 2.276 | 1.966 | **1.338** | **2.10x** |
| 384 | 6.776 | 5.691 | **3.132** | **2.55x** |

cos 0.99997 against the materialised path at every size, and it scales the right
way with N where the hand-rolled one got worse.  **One wiring bug cost an hour
and is worth remembering: the kernel keeps its scores in the log2 domain, so the
caller must pre-multiply the bias by log2(e)** (`biasFromProjRowsK`:
`fmaxf(v * LOG2E, -6e4f)`).  Converting only the dtype left the attention exactly
right wherever the bias was zero and wrong wherever it was not -- cos 0.9687
with the bias, 0.99996 with it zeroed in both paths, which is what localised it.

Two LOCAL ADDITIONS to the vendored kernel make a backward possible: it now
optionally writes the attention output BEFORE the gate (recovering it as
`out / sigm(gate)` would divide by a sigmoid that reaches 1e-05) and the row
log-sum-exp in the log2 domain it works in.  Both are gated in `triattn.cu`:
`av (pre-gate)` at cos 0.99997, and `P from lse vs w` -- the softmax
reconstructed from nothing but q, k, the bias and lse -- at cos 0.99997.

### The triangle attention backward (`flash_bwd.cuh`)

Written against a forward that stored only its log-sum-exp, so P is recomputed
rather than kept.  End to end against the fp32 oracle, all six
size-x-orientation cases:

| N | materialised fwd+bwd | flash fwd+bwd | | backward cos |
|---|---|---|---|---|
| 128 | 1.203 | **0.988** | 1.22x | 0.999932 |
| 256 | 6.151 | **4.929** | 1.25x | 0.999930 |
| 384 | 18.511 | **13.487** | 1.37x | 0.999929 |

Against XLA that is **3.12x / 1.45x / 1.55x**.

The gradient is BETTER than the materialised path's (relL2 1.18e-02 against
1.55e-02): f16 carries 10 mantissa bits where bf16 carries 8.

**RETRACTED: dnbb does not force the score gradient into memory.**  I argued
several times, from the reduction axes alone, that because dQ reduces over keys,
dK/dV over queries and dnbb over the BATCH -- the axis the grid is parallel in --
dS had to be written once and reduced coalesced, since N_b-deep atomics would be
ruinous.  Measured at N=256, they are not:

| | ms |
|---|---|
| write dS (134 MB) + reduce it in a second pass | 1.982 + 0.374 = **2.356** |
| accumulate dnbb by atomics, never materialise dS | **1.946** |

So nothing is materialised: dS is not written at all, the second pass is gone,
and the module went 5.42 -> 4.93 ms.  The reasoning was sound and the
conclusion was wrong, which is the whole argument for measuring it.

What IS true is that dK/dV atomics are only 2-4 deep (there are only N/BM query
tiles), and that the output-side costs had to be measured rather than guessed --
the breakdown that started this was: all outputs 2.174, no dS 1.707, no dK
1.969, no outputs 1.010.  The dS write was the expensive one; the atomics were
not.

Tuning, measured.  Writing dQ and accumulating dK/dV straight into the combined
gradient as bf16x2 -- which would retire three f32 buffers and a 150 MB scatter
-- costs the kernel 1.79 -> 3.53 ms and the module 4.93 -> 6.22, slower than the
path it replaces, with the gradient unchanged.  An f32 atomic into a packed
[M][128] buffer beats a bf16x2 atomic into a 516-wide one, and only half the
lanes carry a pair.  The kernel sits at 25% occupancy, capped by shared memory
(31 KB) AND registers (168) at 3 blocks an SM.  BN = 64 **returns NaN** -- the
softmax and dS passes give one key to one lane, so a key tile is a warp; that is
a `static_assert` now rather than a trap.  LDH = 32 saves 5 KB, does not change
the block count and costs 26%; LDF = 36 costs 9%.  So the first configuration is
the best one, which is worth knowing but is not where this ends.

### FlashAttention: a hand-rolled forward first (superseded)

`flash.cuh` holds a WMMA flash forward -- the [N_b, H, N, N] score tensor never
reaches memory.  It agrees with the materialised path to bf16 (cos 0.99997,
relL2 6.6-7.6e-03 on w@v at every size).

| N | materialised fwd | flash fwd | vs ours | vs XLA's forward |
|---|---|---|---|---|
| 128 | 0.455 | **0.386** | 1.18x | 2.49x |
| 256 | 2.217 | **1.978** | 1.12x | 1.42x |
| 384 | 6.486 | **5.678** | 1.14x | 1.41x |

**Occupancy is the whole story, and it is shared memory that sets it.**  The
first working version ran at 17% occupancy (2 blocks an SM, 34.3 KB of shared)
and was 1.6x SLOWER than the materialised path -- 2.5 TFLOP/s, about 15x off
both the compute and the bandwidth limit, i.e. latency-bound.  Two changes:

* the running output moved from shared memory into registers (lane = channel,
  16 rows a warp): 34.3 -> 26.1 KB, 2 -> 3 blocks, **3.42 -> 2.22 ms**;
* the query tile went 64 -> 128 (8 warps), trading blocks an SM for threads a
  block: 2 blocks of 256 beats 3 of 128, **2.22 -> 1.98 ms**.

Tightening the shared paddings from 40 to 32 saves 9 KB, does NOT change the
block count, and costs 7% in bank conflicts (1.99 -> 2.13).  Measured, not
assumed.

Still at 33% of the card's threads, so there is headroom -- but closing on
Triton means raw `mma.sync` with register-resident accumulators, cp.async double
buffering and swizzled shared layouts, against a mature autotuned kernel.

**The backward is not written.**  It is where the win would be: the materialised
backward is 3.93 ms of tri-attn's 6.21 and carries 938 MB of traffic against the
forward's 536.  It needs THREE passes, not the usual two, because dQ reduces
over keys, dK/dV over queries, and `dnbb` over the batch -- no single loop order
serves all three, and `dnbb` is not optional since AF2's bias is computed from
the pair.

### Things tried that did NOT work

**L2 row-blocking of the projection (falsified).**  The idea followed directly
from lever 4: compute `pre` in row blocks small enough that the GEMM's output
stays in L2, so the gate kernel reads it back from cache instead of HBM, and
keep the sigmoid gates instead of `pre` (the backward needs `dp = dpg*g` and
`dg_pre = dpg*p*g*(1-g)`, and since `a = p*g` is already saved for the
contraction, `p` is never needed -- 33.5 MB instead of 83.9).  Measured at
N=256, forward+backward as one graph:

| rows per block | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 (one call) |
|---|---|---|---|---|---|---|---|
| ms | 4.39 | 3.37 | 3.16 | 3.02 | 3.00 | 2.94 | **2.91** |

Monotonic, and every blocked variant is worse than the single call it replaced
(2.76).  A GEMM over a few thousand rows is far enough off peak to swamp the
traffic saved, and the gate kernel's own streaming writes evict the block before
it is read back.  Reverted; the code carries the measurement instead.

**CUDA graphs (kept, but small).**  Capturing forward+backward as one graph is
free and legal here -- a design run replays one shape for thousands of steps --
but it is worth 1.08-1.12x at N=128 and only 1.03x at N=256 and above.  The
launch overhead it removes is a small share of the work at these sizes.

**The attention GEMMs' leading dimension (real, capped).**  q, k and v are
32-wide slices of the one [M, 4C+H] projection, so a batched GEMM walks rows
1032 bytes apart to use 64 of them.  `ldbench.cu` times the same arithmetic at
three strides, N=256: ld=516 **0.504 ms** (8.5 TFLOP/s), ld=128 0.390 (11.0),
ld=32 0.325 (13.2).  So splitting the projection into separate arrays is worth
~1.3x -- but at ld=32 the kernel is writing 134 MB at 412 GB/s, i.e. bound by
materialising the logits at all.  Layout is not the fix; not materialising is.

## Measure the step, not the block -- and measure it properly

Everything else here times one Evoformer block.  `step_bench.py` times a real
design step through the facade; `step_profile.py` splits one into its phases.
Getting them to run turned up a bug: `af2.features.featurize` writes
`inputs['batch']` only when there IS a reference structure, and `_featurise`
dereferenced it unconditionally, so `af2.model('100').design()` -- AF2 de novo
monomer design, the facade's own entry point -- raised `KeyError: 'batch'`
before the first step.  `tests/test_app.py` builds all four
(af2|af3) x (monomer|binder) models and never calls `design()`.

**A design step at L=100 is 333 ms**, confirmed two ways:

    ctx 0.0 | sample_params 0.5 | dispatch 132.0 | device 202.1 | floats 1.1

and by solving the two-point fit `design(iters=5)` 19.205 s,
`design(iters=20)` 24.197 s -> 333 ms an iter with **17.5 s of fixed cost per
design() call**.

**I first reported 3.9 s a step and concluded the Evoformer was under 20% of it.
That was wrong, and it was my harness.**  `DesignModel.design()` builds a NEW
`Design` for every seed, so `jax.jit(jax.value_and_grad(self.objective))` is
handed a fresh bound method, misses its cache, and recompiles; with `iters=5`
that 17.5 s was charged to every step.  The flat-with-length curve
(3983 / 3862 / 4572 ms at L=50/100/150) and the idle GPU were both that
compile.  The end-to-end min_len figure quoted from the same harness
(4288 -> 3879 ms) is contaminated too; the block measurement is the sound one.

Two things that ARE real, and neither is a kernel:

* **17.5 s of recompilation per seed -- FIXED.**  The `Design` was constructed
  inside `for seed in seeds:` and the jit lives on the `Design`, so every seed
  handed `jax.jit` a fresh bound method.  It is now reused for a seed whose spec
  is structurally identical (`_design_sig`), with the cache on the model so a
  repeat `design()` hits it too, and each result holding a shallow copy so
  `validate.py` and `plot_traj` still see per-seed state.  3 seeds x 4
  iterations at L=100: **56.891 -> 4.106 s, 13.86x**, trajectories identical
  across all three seeds.  The 342 ms an iteration that leaves agrees with the
  profile (336) and the two-point fit (333).
* **RETRACTED: "133 ms of a 333 ms step is host-side dispatch, GPU idle."**  The
  step profile splits a call at `block_until_ready`, and under async dispatch
  that boundary is meaningless -- whatever the host does not absorb surfaces in
  the sync, and the reverse.  Turning on XLA command buffers moved the split
  from 133.9/203.2 to 206.6/135.2 and left the TOTAL at 338 ms.  Queuing twelve
  steps and syncing once gives 338.8 ms a step against 338.0 syncing each, so
  there is no run-ahead to be had (PJRT bounds the queue, so "enqueue only"
  at 321.7 ms does not separate them either), and nvidia-smi reads 97% through
  the benchmark.  The "GPU idle" sample that started this was taken during a
  compile.

**A design step is ~338 ms at L=100 and it is DEVICE-BOUND.**  So the device
time is the right target after all, and these kernels aim at it -- the argument
against finishing them is effort per millisecond saved, not host-versus-device.
The outer product mean's reassociation cut device work too, and cost twenty
lines.

## The biggest win so far was not a kernel

At design depth the outer product mean is 77% of an Evoformer block's MSA track
and about an eighth of the whole block -- 4.862 ms forward+backward at N=256,
against the triangle multiplication's 6.295.  AF2 computes it as

    act = einsum('acb,ade->dceb', left^T, right)      # [N, O, O, N]
    act = einsum('dceb,cef->dbf', act, output_w)

which forms a 134 MB intermediate and costs N^2*S*O^2 + N^2*O^2*Cz.  Contracting
`output_w` into `right` FIRST makes the shared index (s, c), of extent S*O,
instead of (c, e), of extent O*O:

    right_w = einsum('sje,cef->scjf', right, output_w)   # [S, O, N, Cz]
    act     = einsum('sic,scjf->ijf', left, right_w)

That is cheaper exactly when the alignment is shallower than the outer channel
(32).  **Design runs one or two rows, so it always is.**  Measured on an A10,
the whole module forward+backward: N=128 **1.214 -> 0.281**, N=256
**4.862 -> 0.761**, N=384 **10.722 -> 1.341** -- 4.3x to 8.0x, from about
twenty lines of JAX and no kernel at all.

Value-preserving: relL2 6.0e-07 in float32 against the order it replaces, and
through a whole Evoformer block out/pair is 7.5e-07 and grad/pair cos
0.999999962.  Committed on the fork's `opm-reassociate` branch.

**This is worth more than everything in this directory.**  The pair-track
kernels take 32.2 ms to 20.6; this takes ~10% off a trunk pass for a
reassociation, and it applies on every card rather than only where these kernels
are built.  Worth remembering before the next thousand lines of CUDA.

## What is native and what is not

A design step runs the Evoformer stack -- all 48 blocks, all seven module
types, forward and backward -- in CUDA, and nothing else. The rest of the
model is still XLA, and that is why the end-to-end number is below the
stack's own.

**RE-MEASURED 2026-10-09.  THE FIRST VERSION OF THIS TABLE WAS WRONG and
said 63.3%; the error is instructive.**  `NE_TIME` prints a running count,
and the grep that read it filtered to a list of `n=` values -- so the
FORWARD count was invisible and I assumed one forward a step.  The binder
preset runs `num_recycle = 1`, which is **two** trunk forwards a step
(`fwd n=82` against `bwd n=41` over 41 steps).  Unfiltered, L=224:

| | ms a step | |
|---|---|---|
| native Evoformer (fwd 227.1 x2 + bwd 481.7) | 935.9 | **83.4%** |
| everything else, XLA | 186.5 | **16.6%** |
| total | 1122.4 | (XLA end to end: 1618.8, so 1.44x) |

**And the step has no host-side overhead at all.**  One `Design.run()` --
the jitted forward+backward with its sync -- is 1118.6 ms against a full
design iteration's 1120.8: the optimiser, the metric floats and the
bookkeeping are **2.2 ms, 0%**.  An earlier `nvidia-smi` sample read a mean
48% utilisation and looked like idle; it was an artifact of a coarse
instrument on a 1.1 s step.  There is nothing to win outside the model.

**AND 3.8% OF IT IS A FREE WIN BEHIND A FLAG.**  The 4-block extra-MSA stack
has been ported since pass three and defaulted OFF because it needs a second
block of scratch -- 11.44 GB against 7.93 at L=224, which did not fit
alongside everything else.  This session's 3.2 GB of savings changed that:

| | ms a step | VRAM |
|---|---|---|
| native, stack only | 1118.3 | 14.1 GB |
| native + `--extra` | **1075.5** | 15.8 GB |

**-42.8 ms, -3.8%**, for `backend.enable(m, extra=True)`.  Multimer only --
the monomer network's extra stack has never been dumped.  Worth turning on,
and worth re-checking at every length, since whether it fits is now the only
question.

**THE STRUCTURE MODULE IS IN THE GRAPH BECAUSE OF ONE 0.1-WEIGHT TERM.**  The
af2 binder preset's loss set is `con + i_con + plddt + pae + i_pae + rg`.
Five of the six read a HEAD on the pair or msa representation.  `rg_target`
reads `structure_module['final_atom_positions'][:, CA]` (`losses.py:445`), so
that one term pulls eight IPA layers into every step, forward AND backward,
all of it XLA.  It is there for a radius-of-gyration
penalty at weight 0.1.

**BUT IT IS NOT THE LARGEST UNPORTED PIECE -- that was a guess and the
control killed it.**  Dropping `rg` measured 1122.5 ms against 1118.4, i.e.
nothing, and the byte-identical sub-timings of the first attempt (6.71s/6 and
17.90s/16 in BOTH arms) were the tell that the arms were not separated.  The
control strips FIVE terms, which should dead-code the pLDDT head, the PAE
head AND the structure module:

| loss terms | ms a step |
|---|---|
| all six (`con i_con plddt pae i_pae rg`) | 1118.4 |
| five (drop `rg`) | 1122.5 |
| **one (`con` only)** | **1106.4** |

**12.0 ms, 1.1%** for all of it.  So the heads and the eight IPA layers
together are ~12 ms at L=224, and the structure module being in the graph is
a correctness curiosity, not a cost.  **The 410 ms of XLA is somewhere else**
-- the extra-MSA stack (~80 ms, of which porting recovers 42.8), the embedder
and template pair stack, and host-side dispatch.  Sampling `nvidia-smi`
through a steady design loop read a mean 48% utilisation, which would put a
large part of the step in GPU idle; that is a coarse instrument and the next
person should measure it with events before believing it, but
`step_profile.py` found the same thing at small L.

| piece | forward | backward |
|---|---|---|
| triangle multiplication x2 | ours (cuBLASLt) | **done**, 2.28x |
| triangle attention x2 | localfold's flash | **done**, 1.45x |
| pair transition | ours | **done**, 1.71x |
| outer product mean | ours | **done** |
| MSA row / column attention | ours | **done** |
| MSA transition | ours | **done** |
| 48 blocks with per-block remat | **done** (`stack.cuh`) | **done** |
| the 4-block extra-MSA stack | XLA | XLA |
| embedder, template pair stack | XLA | XLA |
| structure module (8 IPA layers) | XLA | XLA |
| distogram / pLDDT / PAE heads | XLA | XLA |

localfold has a tuned FORWARD for every row in that table (~19.5k lines across
`native/af2` and the shared `native/af3`) and a backward for none of it, which
is what this port had to supply.

The honest sequencing from here, if it is continued. **Items 1 and 2 are
DONE**, and are left in the list because what they were worth is the useful
part:

1. ~~**The extra-MSA stack.**~~ Done, and it is 3.0% of a design step (2077.2
   -> 1571.4 -> 1523.8 ms at L=256). Off by default: it costs a second full
   block of scratch, 11.44 GB against 7.93 at L=224. Porting it is also what
   exposed the MSA-depth-1 bug, which was worth more than the 3%.
2. ~~**A fused forward for triangle multiplication, with the backward on the
   gates-and-`a` formulation.**~~ Done in pass four, and the prediction in
   this list was right: the formulation *is* the right choice once a fused
   forward stops materialising `pre`. `projGateK` + `combineDxK` + `scatterDxK`
   are 2.9% of fwd+bwd and 270 MB.
3. The embedder and the template pair stack.
4. The structure module and the heads -- the most line count for the least
   time, and the IPA layers are a different kind of kernel entirely.

None of 3-4 changes the headline much on its own. The stack was the 70%.

**What is left inside the stack, priced.** After four passes the pair track is
at 380-480 GB/s nearly everywhere, and what remains is not a list of knobs:

| candidate | worth | why it is hard |
|---|---|---|
| `flashBwd` core | ~34 ms | FD = 32 caps it; needs mma.sync + cp.async, not tuning |
| `projGateK` / `combineDxK` to cuBLAS-class efficiency | ~50 ms | they run at 50-60% of achievable bandwidth; software pipelining |
| a keep tier for trimul's activations | 53.8-72.4 ms | 8-10 GB, 6.7-7.5 ms/GB -- the price every other trimul keep was rejected at |
| f16 straight into `ta.f gemm out` | 7.2 ms | retires `halfToBf16`, but changes the module's output rounding |
| one in-place f32 residual buffer | 0 ms, 168 MB | aliasing risk in a 48-deep remat loop |

## Gates

`oracle.py` writes a block's inputs, real weights, outputs and
d(loss)/d(activations) as `.npz` plus flat `.f32` with a manifest.
`referee.py` recomputes the same module in float64 from that dump.  The CUDA
harness scores itself against both.

```
python dev/native_block/oracle.py --n 256 --part pair_transition --dump out
python dev/native_block/referee.py out/pair_transition-n256
nvcc -O3 -std=c++17 -arch=sm_86 transition.cu -lcublas -lcublasLt -o transition
./transition out/pair_transition-n256
```

Current numbers, against float64:

| path | forward relL2 | backward relL2 | backward cos |
|---|---|---|---|
| JAX f32 | 3.4e-07 | 1.85e-04 | — |
| ours f32 | 3.4e-07 | 2.33e-04 | 0.99999997 |
| ours bf16 | 3.5e-03 | 3.60e-02 | 0.99935 |

Stable across size: at N=128/256/384 the bf16 backward is cos 0.999356 /
0.999352 / 0.999358, and the f32 backward never leaves cos 0.9999999.

**Cosine leads, not max|d|.**  A gradient that is scaled, and one that is right
everywhere except a single element, are different failures, and only cosine
says "this would steer the optimiser the same way".

## Retracted

**"The extra-MSA stack is 4 blocks of pure waste."**  Said in an early audit and
repeated since.  Only half true: `pair_activations = extra_evoformer_output['pair']`
takes the PAIR out of that stack, and the pair track inside those four blocks
(two triangle multiplications, two triangle attentions, a transition) is real
work whose result is carried into the main stack.  What is dead is the stack's
MSA side -- its own MSA output is discarded, and its outer product mean
contributes exactly zero because the manufactured row is masked, so `left` and
`right` are zero.  At `extra_msa_channel = 64` that side is cheap, so the saving
is a fraction of a percent of a trunk pass, not the ~8% I claimed.

**"dnbb forces the score gradient into memory."**  See the triangle attention
section: the atomics win, and dS is not materialised at all.

## Traps this cost, in the order they were hit

1. **`zero_init` makes the gate vacuous.**  `utils.final_init` under
   `zero_init` initialises every residual branch's last Linear to exactly zero,
   so a block built by `hk.init` returns its input unchanged.  The first run
   reported `cos 1.000000000, max|d| 0.000e+00` against an implementation that
   could have computed anything.  `--weights` now takes the block from a real
   checkpoint; `params_model_1_multimer_v3.npz` holds all 48 on a leading axis.

2. **tokamax parses your argv.**  Its config reads `absl.flags.FLAGS(sys.argv)`
   lazily, from inside the Triton attention path, several frames below
   `jax.grad`.  Any argparse flag absl does not recognise raises
   `UnrecognizedFlagError` mid-trace.  It only fires when the Triton path is
   actually taken, so it appeared at N=256 and not at N=128.  The harness hides
   argv after parsing.

3. **The oracle was the one using TF32.**  JAX's default matmul precision on
   Ampere is TF32 -- 10 mantissa bits.  Against a true-f32 CUDA implementation
   that reads as relL2 3.8e-04 forward and 1.3e-02 backward, and it looks
   exactly like a port bug.  `jax_default_matmul_precision=highest` makes it a
   float32 reference.  (cuBLAS was innocent: `CUBLAS_PEDANTIC_MATH` changed
   nothing, bit for bit.)

4. **A relu makes `max|d|` meaningless for a gradient.**  f32 and f64 disagree
   on the sign of 3 of 33.5M pre-activations, each within 1e-6 of zero, and each
   flip injects an O(1) error into a whole row of the gradient.  That is the
   entire `max|d| 2.5` outlier; `rstd` never leaves 0.78-1.41, so there is no
   LayerNorm amplification and nothing to fix.

5. **cuBLASLt's bias type must equal the matmul's.**  A bf16 matmul with an f32
   bias is `NOT_SUPPORTED` for every bias epilogue, in both the NN and TN form,
   and the heuristic will not say which attribute it objected to.  `probe.cu`
   sweeps (dtype x bias dtype x transa x epilogue) and prints the table.

6. **A kernel parameter will shadow a namespace constant, silently.**  The
   flash kernel takes the projection's leading dimension as `int LDP`, and a
   `static const int LDP = 40` was later added beside it for the P scratch's
   stride.  Inside the kernel the parameter won, so the f32 scratch was placed
   at `Ps + BM*516` instead of `Ps + BM*40` -- an out-of-bounds shared write
   that compute-sanitizer pinned to one line in seconds and that I had spent
   three builds blaming on a WMMA `ldm` constraint.  The constant is `LDPS` now,
   and `ldm = 36` turned out to be perfectly legal.

7. **The free relu bitmask is not free.**  `RELU_AUX_BIAS` and `DRELU` let
   cuBLASLt write and consume the relu's sign inside the GEMM.  Asking for the
   aux tensor costs a **5.6x slower algorithm**: the same GEMM is 0.194 ms with
   `DEFAULT`, 0.199 with `RELU_BIAS`, and 1.098 with `RELU_AUX_BIAS`.  Doing the
   mask in one streaming kernel of our own is 0.409 ms against DRELU's extra
   0.755.

8. **A LayerNorm that straddles two layouts will be uncoalesced on one side.**
   The centre norm reads the contraction's channel-major output and feeds a
   row-major GEMM.  Written as one kernel, each thread wrote its position's 128
   contiguous channels while its warp neighbours sat 256 B away, so a warp
   issued 32 transactions for 64 useful bytes: 123 GB/s, and the single most
   expensive step in the module.  Removing a redundant read pass bought 3%; what
   fixed it was splitting into a coalesced statistics pass and a tiled
   transposing write, 0.525 -> 0.123 ms, with the gate unmoved to nine digits.
   The backward had the same fault mirrored, on its read.

9. **Timing on a contended card lies asymmetrically.**  Another session held the
   A10 at 100%.  Every absolute number was ~2.2x inflated, and min-of-N made a
   timed forward plus a timed backward come to *less* than a timed
   forward-and-backward (1.68 + 1.55 against 5.79 ms), because both halves have
   to be lucky at once.  It reproduced to three digits across runs, so it did
   not look like noise.  All numbers above are from an uncontended card.

## The campaign tables: a schema, not a summary

`campaign_output.py` is 607 lines and almost none of it is arithmetic. It
is a FILE FORMAT -- three stage tables, a column order, a rounding rule, a
summary, a metadata record -- and every tool downstream of a campaign
reads a campaign by reading those files. `bindcraft rank` reranks a
finished campaign on whichever check matters; `bindcraft filter`
reapplies thresholds to see what a different filter would have kept;
`parameter_sweep` compares arms; the desperation ladder counts what the
campaign is failing at. None of them loads a model. All of them take a
folder.

So the port wrote a table of its own with thirteen columns, one row per
trajectory in all three files, and a campaign it ran was a campaign
nothing bc2 ships could read.

`campaign_out.cuh` is the schema. What it writes:

```
<project>/
  1_Trajectories/!_Trajectories.csv     one row per trajectory, finished or dropped
  1_Trajectories/<design>/<design>_losses.csv   one row per gradient step
  2_Refolded/!_Refolded.csv             one row per MPNN CANDIDATE
  3_Ranked/!_Ranked.csv                 one row per ACCEPTED SEQUENCE, ranked
  summary.csv                           mean/std/min/max per (scope, metric)
  campaign_metadata.json                the settings, resolved
```

**The gate is bc2's own tools, run on the port's folder.** The component
half (`campaign_output_gate.py` with no arguments) drives the port's
`ordered_csv_columns`, its `recorded_number` and its summary arithmetic
against bc2's own functions on column sets the gate chooses -- including
shapes no campaign this port can run would produce. The end-to-end half
(`--campaign <folder>`) takes a folder the C++ driver actually wrote and
calls `rerank_campaign`, `refilter_campaign` and `summarize_campaign` on
it. They are not two strengths of the same check, and the second one
found three things the first could not:

| what | how it showed |
|---|---|
| the metadata record carried `targets` as a STRING | `rank.py` iterates it, so `'h'.get(...)` -> AttributeError and `bindcraft rank` died on the port's own folder |
| the metadata record carried the RAW `filters` block | bc2's `FINAL_CONFIDENCE_FILTERS` maps four flat settings onto it before anything reads it; `bindcraft filter` reapplying "the campaign's own filters" threw out both designs the campaign had accepted, on `i_pTM >= 0.7` for a campaign that asked for 0.0 |
| `i_pDAE` of -1 for a design with no interface | bc2 returns None there, and -1 is a number that RANKS: below every real reading and above the `-inf` an empty cell gets |

Two smaller ones came out of the component half. `strtod` stops at the
first character it cannot use and reports success, so the design identity
`74611a93a194c3ec` parsed as 74611 and the campaign summary carried a row
averaging the hash column -- `float(part)` requires the WHOLE string.
And `statistics.fmean` is `math.fsum(data) / n`, which is exact: a naive
accumulation differs in the last ulp, invisible until a value lands on a
rounding boundary, which for a summary rounded to two places is the only
kind of disagreement it can have.

**The loss column names are load-bearing.** `parameter_sweep.py`'s
autotune reads a trajectory's losses.csv with
`column.partition('.')[2].partition('.')[0]` and keeps the columns that
are in `build_losses(settings)`. A column called `complex.i_con` is a
column autotune cannot see, so the stage-length review runs, finds no
loss, and silently changes nothing -- and the file parses either way.
The columns are bc2's loss names, carrying the WEIGHTED value, because
af2.py records `metrics[name] = weighted_losses[name]`.

**The one divergence, stated up front.** `design_hash` hashes the
settings, the values DRAWN for the trajectory and a fingerprint of the
target conformation. The port's draws are not bc2's draws, so the hash
cannot be bc2's hash whatever basis it uses. It is an identity WITHIN a
folder and is not comparable across the two implementations.

## A mask built once, and a length drawn per trajectory

`binder_lengths` is a RANGE in most shipped presets, and the driver draws
a new length for every trajectory and rebuilds the states for it
(`buildComplex`, `buildBinderAlone`, `dev.upload`). It did not rebuild
the MASKS -- `bchain`, `bchain1`, `tchain`, the framework/paratope/target
sets, the coldspot and rigidity pair masks, interface_pae's weights, the
per-protomer `cb_*` sets, compactness's globular radius and protomer
count, distogram_cce's binned target distances. All of them are derived
from `cx` and `binder_len`, and all of them were built once, from the
campaign's FIRST length.

Measured on a `[56, 80]` campaign whose only trajectory drew 66:

| loss | bc2 | the port | rel |
|---|---|---|---|
| `interface_contacts` | 4.0026 | 4.9822 | 2.45e-01 |
| `binder_helicity` | 1.7409 | 1.8703 | 7.43e-02 |
| `binder_pae` | 0.5450 | 0.5287 | 3.00e-02 |
| `binder_contacts` | 3.9076 | 3.8414 | 1.69e-02 |
| `interface_pae` | 0.7081 | 0.6992 | 1.26e-02 |
| `compactness` | 1.0185 | 1.0255 | 6.91e-03 |

and the filters too, because `bchain` feeds `interface_contacts` and
`Interface_Residues` alike. After the rebuild: all 8 losses, all 18
filters and the gradients agree.

**Every campaign in `run_gates.sh` named ONE length**, which makes a row
set built once agree with the states rebuilt inside the loop whether it
is rebuilt or not. The same sentence as the Fab, as `copies`, and as the
four row-sets a de novo binder makes identical: a convention that
coincides when there is one of a thing is not checked when there is one
of it.

**And the first attempt to measure it read MATCH on everything**, because
`BC2_DUMP_FINAL` is overwritten per trajectory: a two-trajectory run was
gated on whichever length happened to come last, and the last one was the
campaign's first length. The gate runs ONE trajectory.

## What ProteinMPNN is not allowed to change

`mark_redesign_residues` is four lines of set algebra:

```
held  = mobile_residues                      induced_fit_mobile_residues
held |= binder x target contacts (4 A)       if keep_interface OR mobile
held |= the TEMPLATE rows                    if binder_scaffold
held |= the linker                           if multidomain and mpnn_fix_linker
```

and `keep_interface = not settings['redesign_interface']`, where
`redesign_interface` **defaults to false**. So bc2 holds the paratope
through ProteinMPNN on every default campaign, and the port held only its
non-designed rows -- it redesigned the part of the binder that makes it a
binder, and then scored the result as though it were bc2's candidate.

`redesign_interface` was on the IGNORED list and the entry was TRUE: the
setting is not read. What the entry did not say is that **not reading a
boolean takes a default**, and here the default it took was the opposite
of bc2's. An IGNORED list answers "is this implemented"; it does not
answer "what happens instead".

The second set is the one that is easy to miss. `induced_fit_mobile_residues`
is not an induced-fit feature: `prepare_binder_redesign` computes it
whenever a binder_alone prediction exists, which is every campaign that
weights a loss on the free fold -- so it is on by default too. And when
it is non-empty the interface is held *whatever* `redesign_interface`
says.

`mpnn_hold_gate.py` compares the SET, not a count, against bc2's own
`induced_fit_mobile_residues` and `binder_target_contact_masks` run on
the arrays the port used (`BC2_DUMP_MPNN`). A hold set one residue
different is a different candidate.

One trap paid for in the writing: the contact mask has to use the
PREDICTION's atom mask. `cx.atom_mask` is the state as built and has
nothing in the binder's rows, so a contact mask computed with it finds no
interface at all -- and an empty hold set looks exactly like the feature
not being there.
## relax_accepted_designs is a restrained minimiser, not a force field

TODO.md said this "needs an energy minimiser" and left it for last.
`protein.relax_protein_complex` is about seventy lines: Adam over the
RESOLVED atoms with three quadratic terms, all O(N^2), all
differentiable by hand. No OpenMM, no AMBER, no hydrogens.

```
clash      0.5 * w_clash * sum_ij C_ij * relu(closest_ij - d_ij)^2
bond       w_bond * sum_b (d_b - d_b0)^2
restraint  sum_i pw_i * |x_i - x0_i|^2
```

**The bond lengths are the input's own.** `bond_lengths` is read off the
structure being relaxed, so the bond term holds the geometry the model
predicted while the clash term pushes overlapping atoms apart. A
relaxation that pulled towards ideal bond lengths would be a different
operation with the same name, and would agree with this one on the clash
term while quietly rebuilding the backbone -- which is why the gate
checks the backbone moves less than an angstrom as well as checking the
coordinates.

**The gate's tolerance is measured, not chosen.** `relax_geometry` casts
`input_atoms` to float32, so the whole optimisation is float32 whatever
`jax_enable_x64` says -- running bc2's own relaxation under x64 moves
*nothing*, which is the control for that. And the restrained basin is
flat in the sidechain directions (restraint 0.5 against backbone 10.0),
so a tiny input difference is amplified: nudging bc2's own input by
1e-5 A moves its own output by up to **1.5e-02 A**. A fixed tolerance
here would be measuring the arithmetic. The port sits at **2.03x** that
floor with a median difference of 1.5e-05; a wrong `closest_approach`
reads 173x and a wrong bond target 189x.

**And an accepted design has no clashes to relieve**, because the
campaign rejected every design that did -- so a relaxation of one
relieves nothing and agrees with a reference that also relieves nothing.
The gate pushes the binder chain into the target until its backbone
overlaps, the same way `geom_probe` builds geometry for the terminus
relus, and then both sides take 2 clashes to 0.

## autotune: the campaign tunes its own stage lengths

`autotune` is **true in settings/core/default.json** and sat on the
port's unimplemented list, because `autotune_review` reads the
campaign's own tables -- `terminated` from the trajectory table and the
loss curve from each trajectory's losses.csv -- and the port had neither
until the schema landed. It is a feature the output format made
reachable.

Every ten trajectories it splits each tuned stage's rows into thirds by
step, takes the mean weighted loss in each, and moves the stage length
by +-25%:

```
closing >= opening            the stage made the loss WORSE: leave it
rejecting, or still falling   +25%   (final third holds > 50% of the fall)
flattened                     -25%   (final third holds < 10% of it)
```

held within [0.5x, 2x] of what the campaign configured, so a review can
never run away from the setting it is tuning.

**It changes nothing on a de novo campaign.** The loop skips an axis the
campaign has not SET, and default.json sets neither `screen_steps` nor
`refine_steps` -- four shipped modalities do (peptide, cyclic_peptide,
large_binder, multidomain), and those are the campaigns where this is
live. Worth saying because "autotune: true" in the defaults reads like
it is always on.

`autotune_gate.py` runs eleven trajectories of a campaign that sets both
axes -- one past the review threshold -- and asks bc2's OWN
`autotune_review` what it would have decided from the same tables. Both
decide `screen_steps 6 -> 8` with a block loss of 1.10475, for the same
reason. Lowering the port's "still descending" cut to 0.9 makes it
decide nothing where bc2 still decides 8, which is the gate's failure
direction.

**The weight half is refused, not half-implemented.**
`autotune_loss_weights` alternates a modality's loss weights between
review blocks and reverts them when the block got worse. The weights
here are resolved once and captured by every loss lambda, so honouring
it would mean rebuilding the objective per trajectory -- and a campaign
that asked for it and got a constant objective would look exactly like
one that asked for nothing. The driver says so at startup.

## The features nothing was auditing, and the cheapest of them

The settings guard classifies SETTINGS and the derived-filter guard
classifies FILTERS. Neither classifies FEATURES -- and a feature is what
a user asks for. An unimplemented one shows up as one ignored setting in
a list of twenty, and the run then produces a design for a campaign that
asked for something else.

`feature_audit.py` drives the driver with a campaign that REQUESTS each
of bc2's 26 `CAMPAIGN_FEATURES` in turn and records which of four things
happens: REFUSED, IGNORED, WARNED-and-ran, or HONOURED. It found five
gaps, none of which were in the work queue:

| feature | what the port did | now |
|---|---|---|
| `cyclize_peptide` | IGNORED -- designed a LINEAR peptide | implemented |
| `trajectory_only` | IGNORED -- accepted designs for a campaign that asked for none | implemented |
| multitarget | WARNED and designed against the first of the targets | implemented |
| `binder_shapes` | IGNORED | implemented (the shipped shape) |
| `idr_crop_count` | IGNORED | implemented |
| `initial_guess` | HONOURED by the guard and read by nothing | named by the audit |

**26 honoured, 0 refused, 0 ignored.** Every one of them is implemented
now, as a campaign writes it. What a refusal still covers is not a
feature but a SHAPE of one: `binder_shapes` with a third conformation,
with two states in one group, or with two bound groups -- each needs
either jax.random's threefry stream or a second complex fold. A binder
designed against the first of two conformations is not a worse answer to
the campaign's question -- it is an answer to a different question, with
the campaign's name on it.

The last one is a hole in the guard rather than only a gap in the port: a
key that names a shipped PROPERTY preset is skipped as "honoured by
composition", which is right for `forced_targeting` (whose layer is a
loss weight and a filter) and wrong for `initial_guess` (whose layer is
the switch itself).

**And `cyclize_peptide` turned out to be a flag.** `inputfeat::offsetK`
was already bc2's `cyclic_sequence_offsets`, wrapping the relative
position the short way round when both ends carry `F_CYCLIC` -- written
for this feature, and nothing in the driver ever set the flag. So the
work was: OR it onto the binder chains, give the host side the same wrap
(`dsin::sequenceOffset`, for `binder_helicity`'s i,i+3 band and the
separation masks -- on a 14-mer ring the ends are neighbours and a helix
through the junction is a helix), and add `Cyclic_Closure_Distance`.

Both gates read the FLAGS out of the dump, so bc2 wraps on its side too:
a port that set the flag and did not wrap, or wrapped and did not set it,
disagrees on `binder_helicity` and on the closure. All 19 filters and all
8 losses agree on a cyclic campaign, the closure at 1.5e-09.

`cyclic_offset_mode` has three values and this implements `direction`,
the default and the only one any shipped preset uses. The other two are
refused: both are a one-line change in two places, and a campaign that
asked for one and got `direction` would be a different relative encoding
wearing the right name.

## An exact sum is not enough; it has to be the same exact sum

`statistics.fmean` is `math.fsum(data) / n`, and the port's summary rows
go through it. `math.fsum` does not simply add carefully: it keeps a list
of non-overlapping partials, adds them **from the top**, stops at the
first one that makes the running sum inexact, and then applies a
half-even fix-up when the next partial shares the residue's sign --
fsum's own comment is that this is "needed so that `sum([1e-16, 1,
1e16])` will round up the last digit to two instead of down to zero".

Summing the same partials in reverse is also exact, and gives a different
last ulp. On six two-decimal readings out of a real campaign the mean
came out `0.49499999999999994` against Python's `0.49500000000000005`,
and `recorded_number` turned that into `0.49` against `0.5`. One cell of
a 123-row summary.

The check that sees it compares against `math.fsum` **bit for bit**, on
values two decimals wide (what a recorded table holds), values spanning
twenty orders of magnitude, and the cancelling sequences fsum's docstring
is about. "Agrees to 1e-12" would have passed the broken version.

## "Inert" is a statement about the verdict, not about the record

`load_settings` calls `add_design_property_records` for every campaign,
which installs a dozen checks with a **non-rejecting** threshold -- inf,
or 0.0 on a minimum. A port that does not compute them accepts exactly
the same designs, and the startup guard said so: it classifies a missing
metric as *rejecting* or *inert* by comparing the resolved threshold with
the `ModalityCheck` default, and these are inert.

That was true for as long as the output was a pile of PDBs and some
stdout. Writing bc2's tables turned every one of them into a COLUMN --
one `bindcraft rank` offers, `bindcraft filter` reapplies and
`summarize_campaign` summarises. On the default campaign bc2 installs 31
metrics and the port reported 20.

| added | |
|---|---|
| `All_Atom_Clashes` | the clash count over 37 atoms rather than CA alone -- and the other half of `Relaxed_All_Atom_Clashes`, which turns out to be the number that shows the relaxer working: 24 all-atom clashes to 0 where the backbone count was 0 to 0 |
| `SS_pLDDT`, `Binder_Loop_Fraction`, `Interface_{Helix,BetaSheet,Loop}_Fraction` | P-SEA again, over the structured residues and over the interface ones |
| `Binder_Mass_kDa`, `Binder_pI`, `Binder_Net_Charge`, `Binder_Cysteines`, `Binder_Extinction`, `Binder_Free_Cysteines` | sequence readings; the pI is fifty fixed bisections of [0, 14], not a convergence test, because that is what bc2 runs |
| `Interface_BuriedArea` and its fraction, `Surface_Hydrophobicity` | a SECOND solvent-accessible-area pass, over the binder chains ALONE. The area the port already had is the COMPLEX's; the difference between them is what the interface buries, and computing one and using it for both makes the metric identically zero |
| `Interface_Hydrophobicity`, `Receptor_Chains_Contacted`, `Target_RMSD` | |

`filter_gate` now agrees with bc2 on 38 of them. The twelfth,
`Receptor_Chains_Contacted`, is None on a single-chain target on both
sides, and the gate says that rather than skipping it.

**And the gate's own blind spot was `if name in cpp`.** It compared every
metric the PORT reported, so a metric the port did not report was
skipped -- which is how eleven of them stayed missing through months of
"all 18 agree with bc2". For the metrics bc2 installs on every campaign
it now computes bc2's value anyway and FAILS when there is one.


## Two targets: the round goes to ONE of them

A campaign that names two positive `targets` is asking for a binder that
holds BOTH, and bc2 does not answer it by scoring both every round. It
ROTATES. `target_schedule.py`'s `MultitargetSchedule` hands each round to
one prepared state, `losses_for_active_states` drops every loss whose
states are not in it, and the sequence update happens on that round's
gradient alone -- except at **anneal**, where `merged_gradient_targets`
turns the rotation into a sweep: every state contributes a gradient and
ONE update follows them all.

The port used to run the complex AND every off-target every step and sum
the gradients. That is a different optimiser on the same losses, and no
value gate can see it: `off-target repulsion` compares the TERM, and the
term was right. What it hid was worse -- the off-target's gradient was
accumulated into `grad` inside `runLosses` and the caller then
**overwrote** `grad` with a memcpy of `dev.d_sequence`, so the repulsion
steered the reported loss and never the search. It has its own
accumulator now.

Two transitions, and which one is in force is `_select_target_transition`:

* **fixed cadence** -- merged gradients, or an epitope rotation: swap
  after `rounds_before_target_swap()` rounds whatever the prediction says.
* **objective gated** -- the default. A positive target is left when its
  i_pTM clears `multitarget_swap_threshold` or patience runs out; an
  off-target when the binder has actually come off it, which is **two**
  criteria and not one. bc2's own note: a short binder holds a whole
  epitope at a confidence no threshold rejects, so the repulsion would
  keep the round forever. `detarget_avoided` asks for quiet at the
  interface AND no contact left.

**The round after a swap is not judged.** `predictions` holds only the
state that was just predicted, so the entry that has just become active
is not in it and the transition test does not run -- every target gets
one free round on arrival. Reproduced by `Observed::have`.

Three things follow from the rotation that are easy to miss:

* **The stage check cannot read the last round.** `judge_stage` filters
  the per-state PEAK of the stage (`stage_filter_predictions` with
  `cumulative`), so every state answers for the stage rather than
  whichever one ran last. The schedule keeps the peaks.
* **The budget doubles.** `rounds_per_binding_target` gives a campaign
  with two positive targets twice the DEFAULT rounds in every stage --
  the default only; a campaign that writes `anneal_steps` gets what it
  asked for.
* **Every state is judged at the end.** `design_stage_filters`
  instantiates each campaign filter once per state as `<filter>.<state>`
  and `evaluate_design_filters` fails the design if any copy fails. The
  port folds each target in turn on the final sequence and runs the whole
  battery on each -- with the PRIMARY last, because the buffers it leaves
  are what the MPNN decode, the structure writer and the dump read next.

And the tables do not widen. `target_ordered_row` COLLAPSES
`<metric>.<state>` into one `<metric>` cell holding the readings joined
by `;` in weight-descending order, with `targets` and `target_weights`
beside them; `rank.py` derives `_mean`, `_worst`, `_best`, `_spread` and
`_selectivity` from that single cell. A port that wrote two columns would
be unreadable to every tool bc2 ships.

Gated twice: `target_schedule_gate.py` drives the port's schedule and
bc2's own class through the same scripted rounds and compares the state
chosen for every one of them (11 cases, plus a negative control that
moves one knob and must disagree), and `multitarget_gate.py` runs a
two-target campaign end to end -- both states get rounds, both are
judged, bc2's `per_target_readings` and `on_target_mean` read the cells
back, and a threshold no design can meet has to be named once per state
and reject every candidate.


## And the protomers of an oligomer

`oligomer_tie` is `symmetric` by default and bc2 then hands ProteinMPNN
the binder's chains as `multi_chain_binders`: one tied group per
position, the protomers' logits averaged, one draw.  The port decoded
the FIRST protomer and replicated its letters -- the same sequence in
the end, and a different draw, because the other protomers' logits never
entered it.  It ties them now, and a campaign asking for `oligomer_tie:
none` (independent protomers, which this port cannot express: it carries
one logits array and scatters it) is refused rather than silently tied.


## A target with no structure, and the crop it is cut to

bc2's disordered-target modality points `target_path` at a FASTA.
`Protein.from_fasta` builds the chain with the SEQUENCE flag and nothing
else -- no coordinates, no atom mask, no TEMPLATE -- so the model folds
the target rather than being handed it, which is the point for an IDR.
And it is CROPPED: `crop_fasta_sequence` defaults to (10, 40), every
window of those lengths is a candidate, and the trajectory draws one it
has not sampled.

**The port read it with `readPdb`, which keeps ATOM records.** A fasta
has none, so the target read ZERO residues and the campaign built a
complex of the binder alone, designed against nothing, and wrote it to
the table as a binder to dynorphin A. `feature_audit` called the feature
HONOURED, because the run reached the weights check and an audit that
classifies the START of a run cannot see what the run then designs.
That is the sharpest version of the lesson this whole file is about: an
unimplemented feature is not the dangerous case; a feature that RUNS and
answers a different question is.

Three pieces, all gated against bc2's own functions
(`fasta_gate.py`):

* **the read** -- the same sequence, with no resolved atom, and the
  target rows marked SEQUENCE and not TEMPLATE (a post-pass, because
  `buildComplex` marks every target row from a structure).
* **the crop** -- the same candidate windows and the same draw, which is
  `random.Random(seed).choice(windows)` and therefore CPython's
  `_randbelow` on the list.
* **`idr_crop_count`** -- N crops become N PREPARED STATES named
  `<target>_epitope_1..N`, which the multitarget rotation then runs with
  a FIXED CADENCE (`epitope_rotation_interval` = the design-model count)
  rather than the objective gate, because "it bound" says nothing about
  which crop to move to. The N windows are distinct: the sampled set is
  shared across the draws of one call.

**And the refold sees more than the design did.**
`validation_crop_flank` (5 by default) widens a cropped target for the
CANDIDATE refold, so an MPNN draw is judged with the context the crop
cut away; the trajectory's own filters stay on the crop it designed
against.  The two tables therefore disagree about `Target_Crop_Length`
-- 11 in the trajectory row, 13 in the candidate rows of the same
trajectory -- and that disagreement IS the feature.  It was on the
IGNORED list, which on a port with no fasta target meant nothing and on
this one would have meant validating on the easier question.

And one bug the crops exposed in the rotation itself: `DeviceState::
upload` copies the ALLOCATION, not the row count, and the step reads `n`
from the device -- so swapping to a target of a different length
predicted the previous one's rows. hPDL1 is 115 residues and mPDL1 is
116, so the multitarget gate had been passing with the second state
predicted one row short, and nothing it checks could see it. The gate
now reads the predicted lengths out of the log.


## Two shapes of the same binder, and the last loss

`fold_switching` was the 37th of bc2's 37 registered losses and the one
left, and the reason it was left is that it is not really a loss
problem.  The arithmetic is `induced_fit_global_loss` -- the same
`relu(TM - tm_target)^2` between a binder's CA in two predictions -- so
the term itself is a third call to the comparison the other two induced
fit hinges already make.  What it needs is the pair of STATES, and those
come from `binder_shapes`, which drives a schedule:

    InducedFitSchedule.select_protein_states
      next_conformation = a draw over the conformations EXCLUDING the
                          current one
      next_state        = a draw inside that conformation
      predict           = {current state, next state}

The shipped shape is `[["<target>"], ["binder_alone"]]` -- two groups of
one.  Both draws then have a single outcome, because the first must
leave the conformation it is on and the second has one member, so every
round predicts the target complex and the binder alone: the pair this
driver already folds.  That is why the feature runs here with no
jax.random stream reproduced anywhere.  Every shape where the draw
actually matters is REFUSED and says which half is missing: a third
conformation or a group of two needs threefry (`jax.random.split`, then
`randint` over what is left), and two BOUND groups would need two
complex folds in a round where `cx`/`dev` is one.

**Its two parameters reach no setting.**  `LOSS_PARAMETERS` maps
`induced_fit_tm_target` onto `induced_fit_global`'s `tm_target` and maps
nothing onto this loss's, so a campaign can only write `tm_target` and
`binder_shapes` in the `losses` block -- which is exactly what
`examples/pdl1_fold_switching.json` does.  The driver reads them there
and DUMPS what it resolved, because a gate that built bc2's loss from
`default.json` would get `binder_shapes=()`, which makes bc2's own
function return a constant 0.0 and agree with a port that never ran the
term.  With them, and with a `tm_target` deliberately different from
`induced_fit_global`'s, the two terms are different numbers and the gate
can tell an alias from an implementation.

**And the MPNN stage changes under it.**  `prepare_binder_redesign` sets
`binder_alone_complex = None` when any group names `binder_alone`, so
`induced_fit_mobile_residues` is never computed on a fold switcher --
otherwise every residue that moves between the two states would be HELD
through ProteinMPNN, and on a design whose whole point is to move
between them that is most of the binder.  The interface hold still
follows `redesign_interface`.


## A campaign that is several campaigns: parameter_sweep

`parameter_sweep` does not change a campaign's settings -- it runs
SEVERAL campaigns and says which settings won.  A baseline arm and one
arm per swept level, each in its own subfolder with its own tables,
`max_trajectories` split between them, and a record at the top saying
which arm came out ahead and whether that answer is resolved.  A port
that ignored the setting ran every trajectory on the baseline and
reported the result under the campaign's name, which is why this was
refused rather than ignored until the arms existed.

**The axes are derived when the campaign names none.**  In bc2's order:
a modality switch that brings one (`forced_targeting` ->
`forced_targeting_shell`), then every `weights_*` the campaign moved off
bc2's own default (`modality_loss_weight_axes` -- the same function
autotune's weight half needed), then whichever of the five default
weight axes the campaign actually set, then the two stage lengths.
`max_arms` truncates that list AXIS BY AXIS rather than arm by arm, so
an axis is swept at every level or not at all.  One axis,
`weights_binder_helicity`, is swept at ABSOLUTE values (-0.6 and 0.3)
instead of multiplied -- a negative weight times 0.5 is still negative
and the question about helicity is which sign wins -- and only when the
campaign named no axes of its own.

**bc2 raises on three kinds of axis and so does this.**  An axis naming
a setting nothing reads, an axis read by the model constructor every arm
shares (`design_recycles`, `copies`, the backends), and an axis the
campaign wrote as an ARRAY, which is a draw per trajectory and so has no
single value for an arm to move.  This driver adds a fourth of its own:
an axis IT does not vary per arm.  `forced_targeting_shell` is the live
case -- the epitope cutoff is 10.0 everywhere in filters.cuh -- and a
sweep over it would run three identical arms under three names that say
otherwise, which is the feature's own failure mode wearing its output
format.

**The arms interleave.**  `sweep_block_budgets` gives (10, 20, ...,
arm_trajectories) and the loop is `for budget: for arm:`, so every arm
reaches trajectory 10 before any reaches 20 and a campaign killed
halfway holds a comparison rather than one finished arm and nothing
else.  The record is rewritten after every block for the same reason.
Inside an arm, `autotune` is OFF (bc2 passes `autotune=False`): a sweep
moves the settings on purpose and a tuner moving them underneath would
be measuring two changes at once.  And resuming an arm between blocks is
not a second mechanism -- bc2's `CampaignProgress` COUNTS THE TABLES
rather than keeping a journal, so "continue this arm" and "resume this
campaign" are the same code, here as there.

**The objective is PAIRED, and the honest column is `resolved`.**
Trajectory 7 of one arm and trajectory 7 of another start from the same
seed, so the arms are compared trajectory by trajectory and the variance
between trajectories -- which is far larger than the difference a sweep
looks for -- drops out.  An arm whose mean paired change is smaller than
twice its own standard error, or which has fewer than ten pairs, is NOT
RESOLVED, and the record says so rather than ranking it anyway.  Three
scenarios in `sweep_gate.py` drive that flag both ways: a clean
difference, a difference the size of its own noise, and four pairs with
a large mean.

**And what the arithmetic gate cannot see.**  A driver that computed the
arms correctly, wrote the subfolders and then ran the campaign's own
settings in all three would produce a record with a winner and nothing
would say the arms were identical.  So `sweep_campaign_gate.py` reads a
real three-arm run and checks that round 1 of trajectory 0 -- which
every arm starts from the same seed and the same logits -- has the
baseline's weighted `interface_contacts` times that arm's own
multiplier.  That is the cell that moves only if the override reached
the objective.


## The campaign tunes its own weights, and the seed is a string

`autotune_loss_weights` adds a second axis to the autotune review: every
`weights_*` the campaign SET away from bc2's default is ALTERNATED by
0.5x or 2x between review blocks, and the next block optimises that
objective. A design accepted since the last review puts the weights
back; a block that got WORSE than the one before undoes the draw
(`restores`), and the record carries `alternating` to say whether there
was one to undo.

Two things make it a port rather than a formula:

* **The weights have to be mutable.** They were resolved once into 36
  `const float`s captured by every loss lambda, so the alternation would
  have been inert. They are a registry of slots now, reset to the
  campaign's own value and then overridden per trajectory -- which is
  what bc2's `{**settings, **record['overrides']}` does on every call.
  Resetting cannot undo a DERIVED default (non_contact's 1.0 when a
  campaign names an off-target), because a weight the campaign never
  wrote is not an axis.
* **The multiplier is drawn from `random.Random(f"{seed}:{n}")`, and a
  STRING seed is not an integer one.** CPython does
  `int.from_bytes(a + sha512(a).digest())` and seeds init_by_array with
  that integer's limbs, so `pyrandom.cuh` carries SHA-512 and the limb
  conversion; `domains.cuh`'s MT does the rest. Checked against CPython
  on the digest, the limbs and the first five draws.

And one defect the gate found on the first run: `modality_loss_weight_axes`
skips a weight that EQUALS bc2's default, and the port compared a
campaign's double (-0.3) with its own float default (-0.3f widened).
Five axes where bc2 has one, and the extra four would have been
alternated -- a different objective every block, on a campaign that
asked for one axis.


## The tied decode: one letter for the same row in every state

The rotation designs a binder against two targets; the MPNN stage has to
REDESIGN it against both, and `prepare_binder_redesign` does not pick
one. `predict_tied_candidates` puts every state into ONE ProteinMPNN
input, translated 1000 A apart so no state sees another's backbone, and
TIES the binder's rows across them: the group is decoded in a single
step, its members' logits are AVERAGED, and one gumbel draw picks the
letter for all of them. bc2's own note on why the mean and the single
draw: *averaging a draw per state shrinks it by sqrt(N) and quietly
lowers the sampling temperature.*

Tying changes two things in the decoder and only one of them is obvious:

* **the draw** -- one letter per group rather than one per position.
* **`ar_mask`** -- every member of a group shares one decoding STEP, so a
  member is masked against its own group-mates rather than ordered after
  them. A port that gave each member its own step would still produce a
  plausible sequence, which is the worst way for it to be wrong.

`mpnn_tied_gate.py` gates both: the port's sampler, teacher-forced on the
same letters and the same group order, must reproduce bc2's `score()` on
that same `ar_mask` (3.4e-06 relative, argmax 96 of 96), every tied group
has to come back with one letter, and -- the control -- an UNTIED run on
the same input has to split a group, or the gate is measuring nothing.
`multitarget_tied_redesign: false` takes bc2's other branch, where
candidate i is decoded on state i % N, and the end-to-end gate drives
that one too.

**And the probe this gate runs had no source.** `mpnn_probe` was a binary
in the tree built from `mpnn_probe.cpp`, which the ignore file's
inversion drops (only `.cu/.cuh/.py/.md/.txt/.sh` are kept at the top
level) -- so it was never tracked and the file was gone. `mpnn_gate.py`
had been running a binary nobody could rebuild: green on whatever
`mpnn.cuh` looked like the last time someone compiled it, and blind to
every change since. It is `mpnn_probe.cu` now, and the gate rebuilds it
when the header is newer, which is what every other probe here does.


## All three exposed_loops measures

`protease_stable` asks for `measure="plddt"`, which reads loopiness off
the confidence head. The other two read the BACKBONE:

* **geometry** -- a smoothed soft "is residue i within 7 A of residue
  i+3", which is what separates a helix or a turn from an extended run,
  and `exposed_loops_distinguish_sheets` (TRUE by default) maxes in a
  STRAND propensity, because an extended strand fails the i,i+3 test
  exactly as a loop does and calling a sheet a loop is the error the
  measure exists to avoid. The strand half reads `-cos` of the CA
  PSEUDO-DIHEDRAL.
* **both** -- the elementwise maximum, so the gradient follows whichever
  branch won.

Two traps, both paid:

* **The smoothing kernel is `ones(4)/4`, and 4 is EVEN.** numpy's
  `'same'` is the middle m of the full convolution starting at
  `(k-1)//2`, so for even k the window is not centred -- it reaches two
  back and one forward. The port's `convSame` was written for odd k (the
  plddt measure smooths with 3) and would have been quietly off by one
  position. `y[i] = (1/k) * sum_j x[i + (k-1)/2 - j]` is the general
  rule.
* **The dihedral's gradient has to follow bc2's own expression.** bc2
  projects b1 and b3 off `b2_hat` and calls `atan2`; the textbook torsion
  formula is the same function with different eps handling, and jax
  differentiates the expression that is written. The port reverse-modes
  that expression by hand -- `atan2`, the two projections, the
  normalisation -- and agrees with `jax.grad` to 2e-09 on a random coil.

## statistics.pstdev is exact, and "close" is a different cell

`summary_row` writes mean, std, min and max per metric and every cell
goes through `recorded_number` -- two decimals. Two accepted designs at
i_pDAE 0.04 and 0.01 have a population standard deviation of exactly
0.015, and 0.015 is a TIE: the double just below it prints 0.01 and the
double just above it prints 0.02. CPython's `statistics.pstdev` sums the
squared deviations as RATIONALS and takes a correctly-rounded square
root, lands exactly on 0.015, and prints 0.01. The port had the right
FORMULA -- population, not sample -- and the float arithmetic landed on
0.015000000000000001 and printed 0.02.

`exact_stats.cuh` is CPython's algorithm with the integers it needs: every
double is `n * 2^-e` exactly, so with `E = max(e_i)` the sums of `a_i` and
`a_i^2` are integers, `mss = (count*B - A^2) / (count^2 * 2^2E)` is a
rational, and `_float_sqrt_of_frac`'s round-to-odd integer square root
makes the conversion correctly rounded. 219 vectors bit-identical,
weighted towards the ties.

Same lesson as `fsum`, one function along: **where bc2's number is a
CPython primitive, the primitive is the specification.** And the same
shape of miss -- it passed two full suite runs before a campaign produced
a tie.


## Files

* `oracle.py` -- builds one block (or any sub-module list) from the real binder
  preset, runs it with real weights, dumps inputs/outputs/gradients, benchmarks
  XLA.  `--part full|pair|msa|<module>,<module>`.
* `referee.py` -- float64 recomputation of the pair transition.
* `common.cuh` -- oracle loader, error metrics, timer.
* `lt.cuh` -- cuBLASLt plans, cached, in the TN form the epilogues require.
* `transition.cu` -- the pair transition, both precisions, with the gate.
* `probe.cu` -- which (dtype, bias dtype, transa, epilogue) cuBLASLt serves.
* `ldbench.cu` -- the attention GEMM at three leading dimensions.
* `flash.cuh` -- the FlashAttention forward (backward not written).
* `bf16_gap.py` -- XLA's own bf16 against its fp32, to size what bf16 costs.

* `dump_real.py` -- captures the model's own activations, masks and weights.
* `in_model_stack.py` -- scores a hook against the model's own scan, in situ.
* `real_ladder.py` / `real_modules.py` -- the depth ladder and the per-module
  table, both on real activations, both with XLA's own control beside them.
* `precision_control.py` -- loss and grad/seq at identical parameters, with
  XLA bf16 and XLA fp32 as the two controls.
* `design_outcomes.py` -- the acceptance test: seeds x iterations of real
  binder design, XLA against native, compared on i_con.
* `test_pad.py` -- the padded path at a length the kernels cannot tile.

## The precision story, resolved (2026-10-07)

An earlier version of this section claimed the port's bf16 residual chain was
the accuracy problem and that the fix was to carry f32 everywhere. **That was
wrong, and acting on it would have cost most of the speed for nothing.** What
settled it was `oracle.py --bf16-all`, which keeps the weights in bf16 so every
einsum is bf16 x bf16 -- exactly the port's arithmetic. Against fp32 it scores
**2.23e-04**, beside 2.06e-04 for f32 einsums. bf16 costs essentially nothing,
and 2.23e-04 is the floor any bf16 implementation can be held to. One module
was sitting at 6.85x it, which is a bug, not a dtype.

Two bugs, and no synthetic-input gate could see either.

* **A one-pass variance.** `centerStatsK` computed the centre LayerNorm's
  variance as `E[x^2] - E[x]^2`; AF2 sets `use_fast_variance=False` and
  computes `mean((x-mu)^2)`. Algebraically identical, catastrophically worse
  conditioned. The comment on the line read *"gated -- it moves nothing"*, and
  that was true of the gate it faced: N(0,1) input gives a near-zero channel
  mean, where the two forms agree to the last bit. Real pair activations have
  rms 19.8 and span +-262. 1.53e-03 -> 2.67e-04.
* **A cached bias pointer.** `Lt::plan` keys on `(M, N, K, epilogue, dtype)`
  and baked the bias POINTER into the cached cuBLASLt descriptor. One module
  never notices. A 48-block stack swapping 48 weight sets through one `Block`,
  every shape identical, means **blocks 1..47 used block 0's biases**. Depth 1
  was right by accident, so every module gate and every block gate stayed green
  while the depth ladder jumped 70x between depth 1 and depth 2. `run()` now
  re-sets bias and aux per call, which is host-side and free. Depth 8 forward
  1.74e-01 -> 2.70e-03.

Where that leaves the port, on the model's own activations at N=224:

| | before | after |
|---|---|---|
| stack forward, 48 blocks | 7.04e-01 | **6.94e-03** |
| grad/seq against XLA bf16, cos | -0.017 | **0.966** |
| design i_con (XLA 2.17) | 4.26 | **1.55** |

XLA's own fp32 manages cos 0.887 against its bf16, so the port is now closer to
the model than the model's other precision is.

### What the model's precision actually is

`global_config.bfloat16` casts only the activations entering the first module.
`common_modules.Linear` asks for `hk.get_parameter('weights', shape,
inputs.dtype)`, but a loaded checkpoint is float32, so every einsum promotes.
Only `oracle.py` pins `jax_default_matmul_precision=highest`; a real design run
uses JAX's default, so an f32 einsum on Ampere is a **TF32** GEMM. The model
during design is therefore bf16-in, f32 intermediates, TF32 GEMMs -- and the
oracle is MORE precise than the model it is standing in for.

**But do not read that as "a module returns f32" everywhere.** Triangle
multiplication ends `out = o * t2` and `dropout_wrapper` adds it to the
residual with no cast, so the source says f32. Fusing that product into the
residual add -- which gives f32 naturally, strictly better arithmetic --
measured the stack forward at 1.096e-02 against 6.942e-03, **58% worse**;
rounding the product to bf16 first reproduces the old path to the last digit.
So trimul's delta really is bf16. The dtype varies by path and reading the
source is not enough: round it, and see which way the error moves.

Making the CUDA chain f32 was tried too. It moved the error 7.04e-01 ->
7.13e-01 -- nothing -- at +9% time. The f32 residual chain is kept because it
is what AF2 does, not because it bought any accuracy.

### How to gate an AF2 port

Never claim parity from synthetic inputs. A gate on N(0,1) activations cannot
see either bug above: unit-normal input is exactly the case where a one-pass
variance is exact and a per-block weight error is diluted. Use `dump_real.py`
to capture the model's own activations, `oracle.py --inputs` to score against
them, `--bf16-all` for the floor and `--bf16-params` for the weight-rounding
share, `real_ladder.py` for the depth ladder with its control, and
`in_model_stack.py` to score the hook inside the model's own scan. `STACK_OUT`
and `GRAD_TRACE` dump the stack's activations and its per-depth gradient norms.

One trap: `install_hook.install()` needs the MODEL's parameters
(`m._build_runner().model_params[0]`), not a checkpoint loaded beside it, or
`check_params` is comparing that file against the file `weights.py` wrote from
the same file.

## Lengths the kernels cannot tile

The kernels tile 32, and a design does not oblige: a binder against PD1 is 115
target residues plus the binder, and N=211 faulted -- reported as a cuBLASLt
EXECUTION_FAILED from the GEMM after whichever kernel actually walked off the
end. `native_evoformer` now pads to the next multiple of 32 with masked-off
residues and slices the answer back.

The pad and the slice sit **outside** the `custom_vjp`, so JAX differentiates
them itself: the cotangent is padded on the way in and sliced on the way out
with no code of ours in the path. `test_pad.py` gates it against an oracle
dumped at the awkward length itself, checking the gradient as well as the
forward, because a mask that leaked would show up as the real residues being
wrong rather than as an error.

The cost is quadratic in the pair track: 211 -> 224 is 1.13x the pair work.

## Where the time goes now

`BLK_PROF=1`, N=224, depth 48: forward 242 ms, backward 620 ms. The backward
splits into the REPLAY of each block's forward and the actual gradient:

| | replay | gradient |
|---|---|---|
| trimul out / in | 45.8 / 44.8 | 62.6 / 63.5 |
| triangle attention start / end | 25.0 / 30.4 | 114.7 / 122.0 |
| pair transition | 10.4 | 35.3 |
| the whole msa track | 22.2 | 33.9 |
| **total** | **178.6 (29%)** | **432** |

The replay is 29% of the backward and 21% of the whole stack; the two triangle
attentions are 39% of the backward between them, and `flashBwd` alone is 147 ms
of that. `BLK_PROF=2` instruments only trimul and triattn, which is why its
rows sum to 458 rather than 620 -- the missing 174 ms is everything else's
replay plus the residual adds.

### The same question asked again from scratch (2026-10-08)

"Can another pass speed up the backward?" -- re-profiled rather than argued
from the table above. At N=256 depth 48, `BLK_PROF=1`: forward 298.6 ms,
backward 777.4. Inside the backward, replay 222.9 (28.7%), gradient 554.4.
`BLK_PROF=2` ranks the individual kernels, and the ranking is not close:

| kernel | ms | share of the instrumented 592 |
|---|---|---|
| **`ta.b flashBwd`** | **199.4** | **33.7%** |
| `tm.b combineTrans` | 50.6 | 8.5% |
| `tm.f gateTrans` (replay) | 34.1 | 5.8% |
| `ta.b scatter` | 25.8 | 4.4% |
| `tm.b centerLnBwd` | 25.1 | 4.2% |
| `ta.b gemm dx` / `tm.b gemm dx` | 25.0 / 24.7 | 4.2% each |
| the other 20 kernels | 2-4% each | |

So it is `flashBwd` or nothing: it is four times the next kernel, and a pass
that does not touch it cannot win more than about 10% of the backward.

**Three things were tried on it, and all three say stop.**

1. **`BM=128`, the one lever this file recorded as open and worth 2.45x --
   RETRACTED.** The 2.45x was a timing of a kernel that returned cos 0.186.
   The dV/dK sections were generalised (`TILES`/`SPLIT` in `flash_bwd.cuh`) so
   any `WARPS >= 4` is correct; `BM=128` now gives a gradient identical to
   `BM=64` to every printed digit, and is **6% SLOWER** (3.08 ms against 2.89
   at the best padding of each). Seven (BM, LDH, LDF) points swept, every
   `BM=128` row loses. The wider tile costs more occupancy than the halved K/V
   loads return.
2. **The fourth block an SM is unreachable.** This file said "under 25 KB a
   block would buy it". `cudaOccupancyMaxActiveBlocks` on the real kernel
   returns 3 at the floor (LDH=LDF=32, 25088 B), with and without the 100 KB
   carveout. LDH cannot go below FD=32 nor LDF below BN=32, so that IS the
   floor for this buffer set.
3. **The atomics are only a third of it.** Ablated: dK/dV 44.4 ms (22%), dnbb
   20.3 (10%), MMA + softmax core 134.9 (68%). The persistent-block rewrite
   retires the atomics and costs a third of the occupancy -- it is aimed at
   22% of one kernel.

**And the kernel is not slow for its shape.** Its five matmuls are 21.5 GFLOP
a call at N=256, so 2.079 ms is **10.3 TFLOP/s**. On the identical shape,
localfold's hand-tuned *forward* gets 6.4 and XLA's Triton flash gets 4.2 over
forward+backward. We are already the fastest thing measured here, at 8% of the
A10's fp16 peak -- and `FD = 32` is why: a 16x16x16 wmma tile gets two k-steps
out of a 32-wide head before the accumulator has to go through shared memory
to reach the `lane == key` layout the softmax needs, five times a tile. That
ceiling is set by the head dimension, not by the code, and beating it means
`mma.sync` with register-resident fragments and `cp.async` -- a rewrite of the
hardest kernel in the model to chase a bound that nothing else on this shape
has reached either.

**Conclusion: no. The backward is done as a kernel problem.** What is left is
not kernels -- the replay's 29% (enumerated and priced below) and the pieces
still on XLA.

### Which of those XLA pieces, measured (2026-10-08)

The sequencing list at the bottom of this file ranks the unported pieces by
guesswork. `extra_msa_cost.py` replaces the extra-MSA stack with the identity
and differences whole design steps, so it is ranked by measurement instead:

| | ms a step | share |
|---|---|---|
| native design step, L=256 | 1567.9 | |
| the same with the extra-MSA stack removed | 1415.6 | |
| **the 4-block extra-MSA stack** | **152.3** | **9.7%** |

`layer_stack calls=[4, 48, 4, 48]` confirms the right stack was hit -- 4 is
the extra-MSA depth, 48 the main one, each twice because a design step runs
two trunk forwards.

**9.7% of a native step is still XLA, in one place.** At the 1.32x the native
path reaches end to end at this length, porting it is worth about **2.4% of a
step** -- five times what the keep grading below bought, and the largest
single item left anywhere in this directory.

### It is ported (2026-10-08), and the port found a bug in the main stack

`install_hook.install_extra()` runs those 4 blocks natively, and
`backend.enable(m, extra=True)` turns it on. Measured at L=256, same seed,
one design:

| | ms a step |
|---|---|
| XLA | 2077.2 |
| native main stack | 1571.4 |
| native main + extra | **1523.8** |

**47.6 ms, 3.0% of a native step** -- and `extra` is nevertheless OFF BY
DEFAULT, because it costs a SECOND full block of scratch: 11.44 GB against
7.93 at L=224, about 3.5 GB. Memory is this port's weak side and it is what
caps the length a design can run at, so 3.5 GB for 3% is the wrong default
for the same reason the flash keep tier is granted whole-or-not-at-all.

**THE PORT IS A SPECIALISATION AND THE README WAS WRONG ABOUT WHY IT IS
ALLOWED.** The retraction below says the extra stack's outer product mean
"contributes exactly zero because the manufactured row is masked". It does
not. AF2 finishes the OPM with `(einsum + output_b) / (epsilon + norm)`, so
with a zero mask it is `output_b / 1e-3` -- a per-channel CONSTANT of rms
5.04 and absmax 34.2 against pair activations of rms ~63. The conclusion (the
MSA side is dead) survives; the arithmetic does not, and anyone who deleted
the OPM on the strength of that sentence would have changed the model
silently. `addConstNarrowK` adds the constant with `addOpmNarrowK`'s own two
bf16 roundings applied, so the two agree bit for bit.

`tests/test_af2_extra_msa_is_dead.py` holds the precondition -- an all-zero
`extra_msa_mask` -- so that teaching the featuriser to fill the extra MSA
fails a test rather than corrupting a design.

### The bug the gate found: the main stack is wrong at MSA depth 1

Gating the port meant comparing a whole design step against XLA on something
other than the binder, and the first thing tried -- `af2.model('256')`, de
novo monomer -- disagreed by 14x. It is not the extra stack: `main` and
`main + extra` agree with each other and both disagree with XLA.

    one design iteration, same seed, L=256 de novo monomer
      xla     con 4.521840   rg  19.191845   loss 6.701693
      native  con 0.067696   rg  -0.999440   loss 0.391756

A collapsed structure reported as an excellent score. In situ, against the
model's own stack on identical activations, masks and weights:

| | pair cos | relL2 |
|---|---|---|
| S=1 (de novo monomer) | **0.0229** | 4.92 |
| S=2 (binder, the gated shape) | 0.9003 | 0.44 |

**Every dump this port has ever been scored against has S=2** -- "the
designed row plus the template row", as the top of this file says -- and at
S=1 the output is uncorrelated with the model's. `design_outcomes.py` is a
binder, which is why the acceptance test never saw it.

**The cause is NOT the outer-product-mean ordering**, which was the first
hypothesis and is refuted by its own control: the monomer preset does have
`outer_product_mean.first = False` against the multimer order block.cuh
hardcodes, but forcing `first=True` on both backends leaves S=1 at cos
0.0202. That ordering mismatch is real and now has its own guard
(`require_opm_first`); it is not this.

Both guards are in: the hook raises at trace time unless the alignment depth
is 2 (`NE_ALLOW_ANY_DEPTH=1` to investigate), and `backend.enable` checks the
config. Until S=1 is found, the native path is binder-shaped only.

**This also means `scale_sweep.py`'s table was timed on a monomer**, so the
runtimes there are a stack doing the right amount of work and the wrong
arithmetic. The timings stand; nothing in that table was ever a correctness
claim, and none is implied now.

**And it is cheaper to port than the list assumed.** `trimul.cuh`,
`triattn.cuh` and `transition.cuh` do not reference `CM` at all, so the pair
track -- which is ~90% of one of those blocks -- needs no second set of
dimensions. Only `opm`, `msarow` and `msacol` hardcode `CM = 256` against the
extra stack's 64, and `block.cuh` already carries `do_msa` / `do_pair` flags
and `n_msa` / `n_pair` counts. What a port needs is the 4 blocks' weights
converted, a second `Stack` at depth 4, an FFI entry and a hook in the fork.

### The replay's keep space, enumerated and priced

The other 29% of the backward is the replay, and "keep it instead" had only
ever been priced for the two things already kept. `trimul::backward` says
exactly what the replay must produce -- `o`, `t2`, `t`, `cmu`/`crstd`, `a`,
`bb`, `pre`, `mu`/`rstd` -- so the candidate list is closed and finite. Every
one of them, measured at N=256 depth 48 by running the stack with and without
the keep and diffing the `BLK_PROF=2` table:

| keep | what the replay stops doing | ms | GB over 48 | **ms/GB** |
|---|---|---|---|---|
| trimul `cmu`/`crstd` | centre-norm statistics pass | 6.8 | 0.05 | **136** |
| triattn `biasH` | nbb GEMM + channel scatter | 5.2 | 0.05 | **104** |
| triattn `avH`/`lseF` | the flash forward | 37.8 | 1.71 | **22** |
| trimul `pre` | `gemm pre` | 21.9 | 3.2 | 6.8 |
| trimul `a`/`bb`/`t2` | `gateTrans` | 34.1 | 4.8 | 7.1 |
| trimul `t` | the contraction | 11.3 | 1.61 | 7.0 |
| trimul `o` | `gemm out` | 8.2 | 1.61 | 5.1 |

**Re-priced after the pass-four fusions, and the verdict does not move.**
`pre` and `gateTrans` are both gone, so the two trimul rows above merge into
one candidate: keeping `a`/`bb`/`gl`/`gr`/`t2` would let the replay skip
`projGate` outright, 53.8 ms for 5 x 1.61 = 8.05 GB -- **6.7 ms/GB**, the same
number the rejected rows had. Adding `t` and `mu`/`rstd` to skip the
contraction and the input LayerNorm as well gets 72.4 ms for 9.7 GB, 7.5
ms/GB. So the fusions did not open this door: they made the thing a keep would
skip cheaper, which makes keeping it worth less, not more.

**Nothing in trimul is worth keeping.** Every row of it prices at 5-7 ms/GB,
three times worse than the flash outputs this port already takes grudgingly
and twenty times worse than the statistics. The earlier note here rejected
`pre` on "memory is the weak side" and quoted 9 ms/GB against the flash keep's
26; the direct measurement says 6.8 against 22, so the rejection was right and
mild. **Correction to that note:** keeping `pre` does NOT pay for two
recomputes. `gateTrans` produces `a`, `bb` and `t2` from `pre`, and the
backward reads all four, so keeping `pre` skips `gemm pre` and nothing else --
21.9 ms, not 45.5.

### ...and the bug that enumeration exposed: the keep was all-or-nothing

The three tiers worth keeping differ by **6x in value per byte**, and the
flash tier is **97% of the bytes**. `Stack::alloc` took them as one
allocation. So at the one length where the replay costs most -- N=384 on a
24 GB card, where the 1.71 GB tier does not fit -- the stack also threw away
the 0.1 GB of statistics and bias that is six times better value per byte.

The keep is now a **greedy budget**: tiers in ms/GB order, and a tier that
does not fit whole is granted for as many DEPTHS as do fit, with `use()`
restoring the module's own buffer for the depths it did not reach. Measured at
N=256 by forcing the budget (`KEEP_BUDGET_MB`, which exists so the
partial-grant path can be gated on a card with room to spare rather than only
at N=384):

| budget | granted | fwd+bwd |
|---|---|---|
| unconstrained | all three whole | 1083 ms |
| 500 MB | flash 11 of 48 depths | 1115 |
| 120 MB | stats + bias whole, flash none | 1126 |
| 40 MB | stats 38 of 48 only | 1134 |
| 0 | nothing (**what L=384 used to get**) | 1142 |

Monotonic, and `grad/pair` is within the backward's own non-deterministic band
at every row. The partial row is the one that earns its keep as a test: with
the flash tier at 11 of 48 depths, a stale slab pointer in `use()` would have
depths 11+ reading another depth's keep and the gradient would move. It does
not -- `use()` restores the module's own buffer for every depth a tier did not
reach.

**But the flash tier is granted WHOLE OR NOT AT ALL, and that is a measurement
not a preference.** Partial grants are fine value per byte -- 11 of 48 depths
is 27 ms/GB, the same rate as a whole grant -- and value per byte is the wrong
unit at the top of the length range. Run at L=384 with the greedy allocator
free to take the remainder, native came out at **3882 ms and 19.57 GB**
against the all-or-nothing path's 3898 ms and 18.57 GB: **16 ms, 0.4%, for a
gigabyte** of the headroom that is the only thing between that length and the
`cudaMalloc` failure this guard exists for -- spent on an estimate read before
JAX has grown into the backward. So `keepWholeOnly(KEEP_FLASH)`. The two cheap
tiers are ~1% of the bytes and keep their partial grants.

So a length that cannot afford the big tier now keeps the cheap ones -- 9 ms
of 1135 at N=256, and the tiers scale as N^2 so about 27 ms at N=384. **Under
1%.** It is worth having because it removes a cliff, not because it is fast.

`num_recycle`, the third lever on the old list, is not a kernel at all.

## What optimisation actually worked

Thirty-odd commits, and the pattern is uniform: **every optimisation that paid
removed work. Nothing that tried to make existing work faster paid.**

The three tri-mul/attention fusions in pass four do not break the rule, they
are the cleanest case of it: they remove 270 MB of intermediate that was
written and read back for nothing. The arithmetic is unchanged to nine
digits.

Measured and rejected: truncated backprop through the stack (the gradient
*grows* backward, 1.0 -> 24.4, so there is no tail to drop), `BM=128` in the
attention backward (correct now, and 6% slower -- the "2.45x" was a broken
kernel), Welford variance, a max shared-memory
carveout, `LD=32`, `BM=96` and `BM=32`, `cublasGemmStridedBatchedEx` algorithm
selectors, `LOCALFOLD_FLASH_REG=1`, dropping the `dnbb` atomics (grad/pair ->
9.34e-01), and a persistent-block rewrite of the flash backward -- killed by
`-Xptxas`, which reports 128 registers a thread with shared memory as the
3-blocks/SM cap, so the rewrite would have run 8 warps where the current one
runs 12.

What paid: skipping computes whose results are discarded (the `fusedOut` flags,
the replay `tail` flag), fusing elementwise passes into the residual add,
f16 projections in the attention forward, warp-per-row LayerNorms, and letting
cuBLASLt autotune over 16 candidates.

Two notes for anyone reading a measurement here. The backward is **not
bit-reproducible** -- the `dnbb` float atomics give +-1.5% run to run, and
`NO_DNBB` makes it deterministic at the cost of correctness. And compile times
quoted earlier in this file were cache hits on both sides; cold, it is native
98.4 s against XLA 136.3 s.

## Pass four: the two tri-mul fusions, and why the pair track was at the roof

A fourth from-scratch pass at the backward, starting from a step-level profile
instead of a hypothesis. At N=256, S=2, depth 48, forward / replay / backward
per step in ms:

| step | fwd | replay | bwd |
|---|---|---|---|
| trimul_out | 61.3 | 56.7 | 79.7 |
| trimul_in | 60.9 | 56.4 | 79.5 |
| triattn_start | 54.9 | 32.2 | 148.9 |
| triattn_end | 62.0 | 39.2 | 159.2 |
| pair_trans | 31.1 | 12.5 | 44.6 |
| opm + msa_* | 28.1 | 26.3 | 35.8 |

Two things fall out of the middle column that no earlier pass had looked at.
**The triangle-attention replay is 61% of its forward -- the flash keep is
working. The tri-mul replay is 93% of its forward, so the graded keep bought
tri-mul essentially nothing**, and the two tri-muls are 272 ms of a 782 ms
backward.

### The projection GEMM is not compute-bound

`gemm pre` is a [M,128] x [128,640] matmul. Doubling its K costs 4%:

```
N=640 K=128   0.256 ms   41.9 TFLOP/s   393 GB/s
N=640 K=256   0.267 ms   80.5 TFLOP/s   440 GB/s
N=640 K=512   0.521 ms   82.4 TFLOP/s   290 GB/s
```

It is bound by writing a 640-channel intermediate, which `gateTrans` then
reads all of back to produce 384. One memory pass pretending to be two.

`projGateK` does the projection and the gating together, so `pre` never
reaches HBM. It writes the GATE and the GATED PRODUCT instead of the raw
projection and the raw gate, which is the same bytes and loses nothing,
because

    dpre[gate] = dleft * p * mask * gl * (1 - gl) = dleft * a * (1 - gl)

-- no division. **The 640-channel buffer is gone entirely**, 84 MB a module
against 34 for the two gates, and `combineTransK` now reads six channel-major
planes where it read two plus `pre`.

### The backward mirror, which is the bigger half

`combineTrans` + `gemm dx` are the same waste on the way back: combine writes
a 640-channel `dpre` and the GEMM reads all of it. 302 MB of traffic for a
contraction whose inputs are 117 MB and whose output is 17, measured together
at 0.784 ms a call -- 385 GB/s, at the roof, for bytes that need not move.

`combineDxK` builds `dpre` a group at a time in shared and contracts it
immediately:

    dX = sum over the five groups  dpre_g [BMG][I] @ W_g [I][C]

`W_g` is rows `g*I..` of `wcatT`, which is already `[PRE][C]` row-major -- the
layout a wmma `matrix_b` wants, contiguous, no transpose. The dX accumulator
lives in registers across all five groups. `dpre` is gone too. Module
backward at N=256: **1.718 -> 1.529 ms**, with the gradient identical to nine
digits (cos 0.999963581 both ways). `TM_SPLIT=1` restores the old pair.

### What it is worth, three runs each

```
                 fwd ms    bwd ms   fwd+bwd    VRAM
  before          303.4     782.0    1085.4   6.97 GB
  both fusions    294.3     761.8    1056.1   6.70 GB
```

Backward -2.6%, forward -3.0%, total -2.7%, and **270 MB of scratch**, which
matters more than the time: memory is what caps design length, and both
retired buffers are O(N^2).

### Four hypotheses that the measurements killed

The fused forward kernel reaches 0.423 ms against a 0.249 ms floor (a
pure-copy kernel moving the same bytes in the same layout), so it is NOT at
the roof. The gap is none of the obvious things, and each was measured rather
than argued:

* **Not the layout.** Writing four channel-major planes plus one row-major
  costs 0.249 ms; writing one contiguous [M][640] costs 0.211. 404 against
  476 GB/s -- channel-major is 15% off, not 2x off.
* **Not the write granularity.** Staging more row tiles to widen a channel's
  write from 32 to 64 to 128 bytes measured 0.478 / 0.503 / 0.656 ms. The
  shared it costs buys back less than the occupancy it spends.
* **Not occupancy.** BMF = 16 / 32 / 64 / 128 measured 0.902 / 0.505 / 0.473 /
  0.829. BMF=32 has 74 registers and 3 blocks an SM against BMF=64's 126 and
  2, and is slower anyway.
* **Not cold weights.** The stack hands a different 164 KB weight to each of
  48 depths, where the microbenchmark reuses one. Rotating over 48 copies:
  0.430 ms against 0.435. No effect.

What is left is the serialisation between each group's MMA phase and its write
phase, and closing that is a software-pipelining project, not a knob. **One
knob that IS real and was not free: a lane must walk `ij`, not the channel.**
`a` is channel-major, so a lane stepping `c` jumps M*2 = 128 KB and the warp's
store becomes 32 scalar transactions. Getting that backwards cost 2.1x and
left the output bit-identical -- the kind of bug only a timing finds.

### The same pattern, a third time: triangle attention's scatter

Having found it twice, it was worth looking for a third. `ta.b scatter` +
`ta.b gemm dx` land dK/dV/dnbb in the 516-wide combined gradient and read all
of it back: 291 MB at 0.528 ms a call, 384 GB/s, at the roof, for a
contraction whose inputs are 102 MB and whose output is 17. `scatterDxK`
builds each of the five column groups in shared and contracts it straight
away; dQ and the gate gradient are already in `dcomb` and are read from
there, dK/dV come from the f32 atomic buffers directly. `wcatT` is already
`[TOT][C]` row-major -- the layout a wmma `matrix_b` wants -- except that
TOT = 516 is not a multiple of 16, so it is now allocated with 528 rows and
the tail zeroed. `TA_SPLIT=1` restores the pair.

Module backward at N=256: 3.189 -> 3.073 ms ending, 2.902 -> 2.793 starting,
gradient identical (cos 0.999969926 against 0.999969925 -- the last digit is
the `dnbb` atomics, which are not reproducible run to run anyway).

### All three, three runs each

```
                 fwd ms    bwd ms   fwd+bwd    VRAM
  before          303.4     782.0    1085.4   6.97 GB
  all three       297.6     756.8    1054.4   6.70 GB
```

**Backward -3.2%, total -2.9%, 270 MB of scratch.** Real binder design on
PD1, same contig and seeds, finishes in family: best `con` 3.0770 native
against XLA's 3.0719, at 1.47x the speed.

A caution for the next person who reads a microbenchmark here. **Every one of
these fusions delivered roughly half of what it measured standalone.**
projGateK: 0.18 ms a call on the bench, 0.085 in the stack. scatterDxK: 13.9
ms of profiled kernel time, 5.0 ms off the wall clock. Some of that is the
event-pair overhead `BLK_PROF=2` adds to a two-kernel path that a one-kernel
path does not pay, and the rest is that a kernel measured alone in a hot loop
is not the kernel that runs between forty-seven others. The stack-level
numbers above are the ones to quote.

### The flash backward, closed with evidence instead of an argument

The header has claimed since pass two that FD=32 is the ceiling. The obvious
control -- run the kernel at FD=64 -- cannot be run without a rewrite, because
the dQ epilogue writes `h * FD + lane` and so assumes FD == warpSize == 32.
But the evidence was available all along and nobody had put it together:
**our own flash FORWARD, same shape, same wmma machinery, runs at 19.2
TFLOP/s** (43.1 ms over 96 calls, 8.6 GFLOP a call). The backward's core,
with the atomics ablated out, is 15.3. A backward doing 2.5x the work with
longer dependency chains, within 20% of the forward's rate, on a shape where
a mature autotuned Triton kernel manages 4.2 -- that is what an FD-capped
kernel looks like. The remaining 24% is real but it is 34 ms, and it is the
hardest 34 ms in the stack.

## Pass four, second round: five more ideas, two of them right

With the three fusions in, the same method again -- profile, microbenchmark,
A/B every factor one at a time with the source frozen. Module numbers are
`trimul out` and `triattn ending` at N=256, 10 reps, forward and backward in
ms.

**What won:**

| change | before | after |
|---|---|---|
| `projGateK` 32 rows a block, not 64 | fwd 0.938 | **0.885** |
| fragment-contiguous weights, tri-mul only | fwd 0.967 / bwd 1.592 | **0.885 / 1.562** |
| the per-row mask hoisted out of the element loops | — | folded into the above |

The second one is worth stating plainly because it costs nothing at runtime.
`projGateK` was loading its weight fragment straight out of the [C][PRE]
matrix with `ldm = PRE`: sixteen rows of 32 bytes, 1280 bytes apart, sixteen
sectors a fragment, forty fragments a warp. The weight is permuted ONCE a
depth into `[block][k][16]` so the same fragment is 512 contiguous bytes and
`ldm = 16`. Same bytes, same count, one `cudaMalloc` and one permutation
kernel at setup.

**And it is per module, which was not the expectation.** The identical change
in `scatterDxK` LOSES -- 3.147 against 3.034 ms. Both kernels read the same
sixteen weight rows from all eight warps, so in the original layout one
256-byte line serves every warp, while the permuted layout gives each warp its
own 512-byte run: fully used, shared with nobody. `scatterDxK` contracts five
groups out of one weight and reuses those lines more than `projGateK`'s five
separate projections do. `TM_FRAGPERM` and `TA_FRAGPERM` are separate flags
for that reason.

**What lost, all four measured rather than argued:**

| idea | the theory | the measurement |
|---|---|---|
| fuse the centre LayerNorm's three passes into one | 67.2 MB -> 34.1 | fwd 1.047 -> **1.228**, 17% WORSE |
| read the centre-norm cotangent row-major | deletes a 33.6 MB transpose | bwd 1.503 -> **1.562** |
| four tiles in `combineDxK` / `scatterDxK` | 6 barriers -> 2, better MLP | 1.559 -> **1.864**, 3.073 -> **3.260** |
| rotate the group order by warp index | de-phase MMA and writes | fwd 0.886 -> **0.907** |

The first two share a cause and it is worth internalising: **`t` and `dcn` are
16.8 MB each and live in L2, so the extra passes those changes were buying
back were nearly free, while the shared memory and the strided access they
spent to buy them were not.** Halving the HBM traffic of a step whose data
already fits in cache buys nothing. The four-tile variant is the memory story
in reverse -- 78 KB a block is one block an SM, and the occupancy cost more
than the barriers did.

### Two bugs, and what each one says about the gates

**A launch over 48 KB fails SILENTLY, and nothing here was checking.** The
four-tile variant measured 1.028 ms and looked like a 34% win. It was a kernel
that never ran: `combineDxSmem()` had grown to 78 KB with no
`cudaFuncSetAttribute`, so every launch returned an error nobody read. The
gradient said cos 0.0727. Every fused kernel now sets the attribute and every
launch is followed by `CU(cudaGetLastError())`.

**A single-module gate cannot see a missing weight slot.** `wcatS` was added
to `setWeights` and left out of `weightSlots()`, so the whole 48-deep stack
ran block 47's attention weights in its backward: `grad/pair` cos 0.607 with
the forward exact to nine digits, and `trimul`, `triattn` and `block` all
green, because one depth cannot tell the difference. Only the stack gate sees
this class. **Every buffer `setWeights` allocates belongs in `weightSlots()`,
and the check for it is a depth > 1 run.**

### Where that leaves the stack

N=256, S=2, depth 48, three runs each:

```
                      fwd ms    bwd ms   fwd+bwd    VRAM
  start of pass four   303.4     782.0    1085.4   6.97 GB
  the three fusions    297.6     756.8    1054.4   6.70
  + this round         287.5     734.7    1022.2   6.73
```

**Backward -6.0%, forward -5.2%, total -5.8% over the pass, and 240 MB.**
Real PD1 binder design, 3 seeds x 125 iterations, same contig both backends:
XLA 18.7 min against native 13.6, **1.38x**, with i_con mean 2.148 native
against XLA's 2.267 -- inside the +-0.7 single-run spread this test is known
to have.

## Pass four, third round: the stores, and two more negative results

The profile after the second round said something the roofline had not:
**`projGateK`'s FORWARD, which moves 67 MB, is SLOWER than its replay, which
moves 101 MB** (0.442 against 0.424 ms a call). A kernel where more traffic is
faster is not bandwidth-bound, and the remaining suspect was the store ISSUE
rate -- the write loop was emitting a two-byte scalar store per element, 25
million of them a call.

Two rows a lane, one four-byte store. `a`, `b`, `gl` and `gr` are
channel-major, so two consecutive `ij` of one channel are adjacent and pack
into a `__nv_bfloat162`; eight lanes still cover a channel's sixteen rows as
32 contiguous bytes, so the coalescing is unchanged. `t2` and `dx` are
row-major, so there the pair is two channels. The same for `combineDxK`'s six
channel-major reads and `scatterDxK`'s reads of `dcomb` (bf16x2) and
`dKf`/`dVf` (float2).

| | before | after |
|---|---|---|
| trimul fwd | 0.885 | **0.866** |
| trimul bwd | 1.496 | **1.433** |
| triattn bwd | 3.062 | **3.009** |

Output bit-identical everywhere.

**And two more that lost.** Computing all five projections in ONE k-loop --
so the x fragment is loaded once a k-step instead of five times and five
independent MMA chains overlap -- measured 0.911 against 0.885 ms. Extra ILP
is not what this kernel is short of. And vectorising the RESIDUAL CHAIN the
same way (`addNarrowK`, `copyNarrowK`, `addGateNarrowK`, ~48 ms of the
forward, one element a thread) measured 1004.0 against 1006.3 ms on the whole
stack: **nothing.** That chain really is at the bandwidth roof, 100.8 MB a
call at ~450 GB/s, so there was no issue rate to recover. The fused kernels
are the exception, not the rule.

### The pass, end to end

N=256, S=2, depth 48, three runs each:

```
                      fwd ms    bwd ms   fwd+bwd    VRAM
  start of pass four   303.4     782.0    1085.4   6.97 GB
  the three fusions    297.6     756.8    1054.4   6.70
  + round two          287.5     734.7    1022.2   6.73
  + round three        286.2     721.0    1007.2   6.73
```

**Backward -7.8%, forward -5.7%, fwd+bwd -7.2%, and 240 MB of scratch.**

Eleven ideas tried across the pass, five of them right. The five: three
fusions of a wide intermediate, a fragment-contiguous weight layout, and
vectorised stores. The six that lost -- a fused centre LayerNorm, a row-major
cotangent, four tiles, warp rotation, one k-loop for all five projections, and
a vectorised residual chain -- are all documented above with their numbers,
because **the useful output of a pass like this is which arguments do not
work on this shape.**

## Pass four, fourth round: the CUDA-graph divergence, explained

The graph path has been off since it was built, with this note: *"STILL OPEN.
Even on a private stream, a 40-iteration design diverges: i_con 4.4977 against
the stream path's 2.3456 ... The obvious suspect -- a module enqueueing onto a
stream that is not `st` -- was CHECKED AND CLEARED."* It was worth 10% / 6% /
3% at L = 64 / 128 / 256 and nobody could explain it.

**A captured graph bakes in the pointers it was captured with, and most of
them are XLA's.** `NE_PTRS=1` prints them on every call. Twelve design
iterations:

```
NE_PTRS fwd call 1: pair MOVED 0x7d235cfdc000 -> 0x7d2227800000
NE_PTRS fwd call 2: pair MOVED 0x7d2227800000 -> 0x7d22b5e8a900
NE_PTRS fwd call 3: pair MOVED 0x7d22b5e8a900 -> 0x7d2227800000
```

`pair` moves on every call, `msa_mask` on every call, `msave` on every call.
XLA cycles between a handful of addresses, which is why the symptom was
finite rather than a fault: **the graph was reading a PREVIOUS iteration's
activations.** Plausible numbers, wrong iteration, and only visible once the
caller has iterated enough for the allocator to cycle -- exactly why a
single-shot FFI gate agreed to 1.697e-01 and a 40-iteration design did not.

The inputs were already staged into stack-owned buffers (`stage()`), which is
presumably why this survived as long as it did. **The save set was not.**
`pin[i] = psave + i*np` and `min_[i] = msave + i*nm` are slices of XLA
outputs, so the forward wrote its remat save-set to an address XLA had since
reused and the backward's replay recomputed the whole stack from whatever was
there. `useOwnSaveSet()` makes it the stack's own, and `_fwd_call` declares
the XLA-side residual `(0,)` so nothing is allocated twice -- the dataflow
edge that orders the backward after the forward survives at zero bytes,
which is the same mechanism the detached pass already used.

**And a second one found on the way.** One captured forward graph was serving
BOTH save modes. A design step runs the forward twice -- a detached recycle
pass that writes its remat set into one block's scratch, and the
differentiated pass that writes 48 slices -- same kernels, different
destination pointers. They cannot share a graph. They did, and it was correct
only by the accident that the save=1 pass happened to be the second forward
and therefore the one captured; the save=0 pass then replayed it and merely
wrote 1.6 GB it did not need. Reverse the order and the save set would never
be written at all. Three graphs now, not two.

**The ownership is one decision, not two.** Python read `STACK_GRAPH_UNSAFE`
at import and C++ read it per `Stack`. If those ever disagreed in the
direction "Python says token, C++ says use XLA's", the forward would write a
1.6 GB save set into a zero-size buffer. It is now an FFI attribute,
`own_save`, computed next to the shape it has to agree with -- and the
backward derives it from `psave.shape == (0,)` rather than consulting the
environment a third time.

### Does it pay? At design lengths, no.

Four runs of the same 60-iteration design, alternating:

| | i_con | loss |
|---|---|---|
| stream | 1.9550 | 3.5269 |
| graph | 2.3983 | 4.3104 |
| stream | 2.6285 | 4.5625 |
| graph | 2.5962 | 4.4711 |

Both graph draws land INSIDE the stream-to-stream range, which is 0.67 wide
on i_con -- the backward is not bit-reproducible, so that spread is the
instrument's resolution. Against the old 4.4977-versus-2.3456 that is a
closed bug.

**But the wall clock does not move.** The standalone win was 9.6% at N=64 and
1.3% at N=224, and at a binder's length the stack is a smaller fraction of
the step than it is of the standalone gate. Design lengths are N = 160-450,
which is the wrong end of that curve. So the honest result is: the mystery is
explained, the landmine is gone, and `STACK_GRAPH_UNSAFE` stays off by
default because it buys nothing where this port is actually used. It is worth
having documented rather than worth turning on.

### ...and the flash backward's atomics, re-priced

The note above says the dK/dV atomics are 22% of `flashBwd` and the dnbb ones
10%. Re-measured now (`NO_DKV`, a probe whose gradient is wrong by design):
the whole stack backward is 725.2 ms and 701.7 without dK/dV, so **the dK/dV
atomics are 23.5 ms, not 44.4** -- the kernel has changed under that note (dS
is no longer materialised). That closes a tempting rewrite: inverting the loop
order to outer-K/inner-Q would make dK/dV free in registers and make dQ
atomic instead, and dQ is one tensor where dK/dV are two, so the prize is
about 12 ms -- 1.6% of the backward, for a rewrite of the most delicate kernel
in the directory. `NO_DNBB` is not usable as a price at all: it selects a
different template instantiation and measures 98 ms SLOWER than doing the
work, so whatever it costs, that probe cannot say.

### What the MSA row attention's cost actually is

The last unexamined region, and it is not what the name suggests. `msa_row`
is 8.9 ms forward and 17.1 backward at N=256 S=2, which looks enormous for an
alignment of two rows -- and it is, because **most of it is the PAIR side**:
the module LayerNorms the whole [N][N][128] pair to make its attention bias
(50 MB forward, ~85 MB in the backward), and that is at the bandwidth roof.
The MSA attention itself is a flash kernel whose grid is S*H*(N/BM) = 2*8*4 =
**64 blocks on a 72-SM card**, 6% of the machine -- but the work is so small
that retiring it entirely would save about 7 ms, 1%. Underfilled and not
worth filling.

### The in-model timing, and a correction to the NE_TIME note

`NE_TIME` exists because of this comment: *"The standalone driver says 1526 ms
at N=256 depth 48 and the same stack inside the model behaves like 1895."*
Measured now on the PD1 binder, device time on the handler's own stream:

| | fwd ms a call | bwd ms a call | host issue (wall) |
|---|---|---|---|
| graphs off | 379.9 | 963.5 | 321 / 836 |
| graphs on | 384.8 | 965.5 | ~45 / ~180 |

**The in-model stack is not slower than the standalone one; it is bigger.**
`af2.model('A:109/0 64')` has `max_length = 288`, not the 256 the standalone
gate runs. The pair track is N^2-dominated, so scaling this directory's
N=256 numbers (fwd 286, bwd 721) by (288/256)^2 = 1.266 predicts 362 and 913
against the measured 380 and 964 -- **5% apart, which is L2 and clocks.** The
old 1526-versus-1895 gap is 1.24x and the N ratio alone is 1.266x, so that
note was comparing two different lengths and there was never a mystery.

**And it is why the graph buys nothing here.** The two columns above are the
whole story: the graph does exactly what it is supposed to -- host issue time
falls from 321 ms to about 45 -- and the device time does not move, because
321 ms of issue was already hidden behind 380 ms of execution. At N=64 the
device time is ~84 ms and the issue cost is not hidden, which is where the
9.6% came from. A graph helps a stack that is waiting on its host, and at
design lengths this one never is.

## RETRACTED: "the native stack is wrong at MSA depth 1"

That was the headline finding of the previous session and it is **wrong**. The
stack is not wrong at S=1. What was wrong is that `af2.model('<L>')` de novo
is the MONOMER preset, which is a different network, and the harness was
feeding the native stack the other one's weights.

**Two checks, either of which would have settled it.**

```
  the standalone stack gate at S=1 (out_s1, oracle.py --n-seq 1)
    forward   out/pair cos 0.999927   out/msa cos 0.999998
    backward  grad/pair cos 0.986883  grad/msa cos 0.966506
```

Green, and the gradient is BETTER than S=2's 0.9786. The kernels were never
the problem. And:

```
  check_params with the MONOMER MODEL'S OWN parameters
    ptm_weights    PASS
    stack_weights  FAIL -- msa_row_attention_with_pair_bias/query_norm/scale
```

**How the wrong conclusion got made.** `test_design.py` loads
`params_model_1_multimer_v3.npz` from disk and hands THAT to `install()`. So
`check_params` compared the file against the directory the file was dumped
from, passed, and the model went on holding `alphafold2_ptm` weights -- the
exact trap `backend.py`'s own docstring warns about in writing. Every number
from that harness was a different network's weights, and the alignment depth
was simply the thing that correlated with which preset was running: binder
presets are multimer and feed S=2, the de novo preset is ptm and feeds S=1.

**And the control that "refuted" the ordering hypothesis was void.** It forced
`outer_product_mean.first` on the model and found S=1 unmoved (cos 0.0229 ->
0.0202). Both of its rows ran on the wrong weights, so flipping an ordering
flag could not have shown anything. The ordering mismatch was real all along.

### So the monomer network now runs natively

The two presets differ in exactly two things that reach this stack, and they
have to be taken together:

| | multimer_v3 | ptm |
|---|---|---|
| block order | OPM first | **OPM last** |
| tri-mul weights | fused `projection`/`gate` [.,2C] | separate left/right |
| everything else | 48 blocks, 256/128/32 channels, bf16, zero_init, parameter names | identical |

* `Block::opm_first` implements both orders. It is not just the OPM's
  position: under ptm the row attention reads the block's INPUT pair and the
  OPM reads the msa AFTER its transition, so two bf16 views the multimer order
  never needs (`p0b`, `m3b`) have to exist, and `d:opm` moves to the front of
  the backward.
* `weights.py --ckpt ...ptm.npz` fuses the monomer's separate projections and
  gates and renames its two LayerNorms. This is a WEIGHT TRANSFORM, not a
  kernel branch -- the fork's af2 sets `fuse_projection_weights = True` for
  both presets, so the real runner does the same conversion, and
  `check_params` against the model's own parameters confirms ours matches it
  exactly. (First attempt keyed the rename on the leaf name alone and renamed
  the OUTER PRODUCT MEAN's `layer_norm_input` too, which genuinely has
  separate left/right projections in both configs; a manifest diff against the
  multimer one caught it.)
* `backend.enable` reads the order off `runner.cfg.model` and picks the
  matching directory, because the failure mode of taking one without the other
  is a whole session of wrong conclusions.

One monomer design step, `af2.model('256')`, three iterations:

```
  xla      31.7s  loss 5.399281  con 4.073489  plddt 0.613023  pae 0.760526
  native   15.5s  loss 5.891616  con 4.295477  plddt 0.608500  pae 0.780758
```

Against what the broken pairing gave -- native loss 0.391756 against XLA's
6.701693, a collapsed structure reported as an excellent score.

**The MSA-depth guard is gone.** The guard that matters is the one that was
there all along and was being bypassed: `check_params`, against the model's
OWN parameters.

### RETRACTED: "the monomer BACKWARD does not gate, and it is off"

This section used to say the monomer forward gated at cos 0.9969 and its
backward did not gate at all -- NaN at full cotangent scale, cos 0.412 with
the cotangent scaled down -- and `backend.enable` refused the monomer path on
the strength of it. **That measurement was the instrument, not the port.**

`oracle.py`'s default activations are N(0,1) with all-ones masks, and **one
monomer block takes a unit-normal pair to rms 185** -- the oracle does it too
(185.6 against our 185.9), so the forward agrees; what cannot survive is a
backward through a map that amplifies like that. The control settles it: the
MULTIMER stack, green everywhere else, reads **grad/pair cos 0.970** on the
same synthetic SINGLE block, and 0.984 at depth 48. The synthetic regime has
no resolution left for either network (see [[trunk-gate-resolution]]).

On the MODEL'S OWN activations -- `dump_real.py` for the activations, masks
and weights, then `oracle.py --inputs` at the same depth -- at depth 48:

```
                 out/pair    out/msa    grad/pair   grad/msa
  multimer_v3    0.999581   0.999990    0.967039    0.958525   (N=224, S=2)
  ptm            0.999850   0.999989    0.991349    0.990324   (N=128, S=1)
```

**The monomer is BETTER than the multimer in the regime the stack actually
runs in**, forward and backward, and there is no NaN anywhere in it. The
dynamic loss scaling added for the MSA attention's f16 backward
(`maxAbsHalfK`/`pickScaleK`) was a fix for a problem that only the synthetic
inputs had; it is exact up to range and stays.

Reproduce:

```bash
OUT=real_ptm.npz L=128 python dump_real.py
for K in 1 8 48; do
  ORACLE_PTM=1 python oracle.py --n 128 --n-seq 1 --stack --blocks $K \
      --inputs real_ptm.npz --weights /home/ubuntu/params/params_model_1_ptm.npz \
      --dump out_ptm_real
done
OPM_LAST=1 ./stack out_ptm_real/stack48-real-n128 ptm_weights 1
```

and `monomer_stack_gate.py` is the same question asked in situ, which is what
runs in the suite: it builds the ptm model, reads `opm_first` off its config,
picks `ptm_weights` by that and not by a flag, and scores the hook against the
model's own scan (relL2 2.07e-02 pair, 1.72e-02 msa at N=128).

### alphafold2_ptm is not a second network

Diff the two checkpoints by NAME and by SHAPE and the whole difference is:

```
  evoformer/preprocess_1d, left_single, right_single   (22, .) vs (21, .)
  evoformer/pair_activiations (65,128)    vs ~_relative_encoding/
                                             position_activations (73,128)
  triangle_multiplication_*   left/right projection + gate, two LayerNorms
                                          vs one fused projection and gate
  structure_module IPA        q_scalar / kv_scalar / q_point_local /
                              kv_point_local, WITH BIASES
                                          vs six separate projections
  structure_module            affine_update (384,6)
                                          vs quat_rigid/rigid (384,6)
  template_embedding          a genuinely different module
  masked_msa_head             23 classes vs 22   -- design never reads it
```

Every head, every Evoformer width, the whole structure module below the IPA's
projections and the template PAIR stack are identical in name and shape. So
the monomer runs on these kernels with **three runtime decisions** --
`Plan::monomer`: the 22-wide embedder, the outer product mean last,
`position_scale` 10 instead of 20 -- plus layout transforms. Two of the
"differences" are the same operation under another name:

* **`affine_update` IS `quat_rigid`.** `QuatAffine::pre_compose` forms
  `q + q(0,v) = q(1,v)` and normalises, which is `q` composed with
  `normalise(1, v)` -- exactly what `QuatRigid` builds from `(1, v)` -- and
  both add `R_old @ t_update` to the old translation. A rename.
* **the IPA's monomer projections are the multimer ones concatenated.** The
  scalars reshape to (head, per) and split PER HEAD; the points are
  coord-major where multimer is head-major, so they transpose. And the
  monomer's scalar biases are NOT zero (max 0.90) where no multimer
  checkpoint has them at all -- `ckpt.cuh` supplies zeros only when the key is
  absent, so this lands by itself, but it is exactly the sort of thing that
  would have been a plausible-everywhere wrong answer.

`ckpt.cuh::monomerise` does all of it in C++ and `ckpt_ptm_gate.py` holds it
to `alphafold3/af2/convert.py` composed with bc2's own dump writers -- two
independent transcriptions of the same map: **4081 trunk files, 341 extra,
101 template-stack and 68 structure-module files, byte for byte.**
`embed_ptm_gate.py` holds the monomer embedder to bc2's own module at relL2
3.5e-08 forward and 3.0e-07 backward; feeding it the multimer's 21-wide
layout instead moves the answer by 6.9e-02, so the gate can see the column
shift it exists to catch.

**What is left is the TEMPLATE EMBEDDER.** The monomer's is one `embedding2d`
of 88 features into 64 channels plus an attention over the template axis,
where the multimer has nine `template_pair_embedding_*`, a
`query_embedding_norm` and an `output_linear`. `model_1_ptm` and
`model_2_ptm` -- the two bc2 validates on -- both enable templates, so a
monomer run here would be a prediction with the target's template thrown
away. The driver therefore REFUSES a monomer checkpoint, in all three places
one can arrive (the design model, the design pool, the validation ensemble),
and says which module is missing; `refusal_gate.py` checks it.


### And the whole monomer network now runs, end to end, in C++

```
  bindcraft camp.json params=/home/ubuntu/params/params_model_1_ptm.npz \
      steps=2 mutate_steps=1 validation_models=0
  ...
  MONOMER (alphafold2_ptm): the 22-wide embedder, the outer product mean
  last, position_scale 10, and its own template embedder
  ...
  plddt mean 0.3930 over 175 residues   ptm 0.3545   iptm 0.4990
```

Four modules, each with its own gate, plus two things that are NOT a port:

* `embed.cuh` grew a `monomer` flag: target_feat 22 wide with the 20
  sequence columns at **1**, one 65-bin `pair_activiations` (AF2's own
  typo), and no feature mask. Against bc2's own module, relL2 3.5e-08
  forward and 3.0e-07 backward; fed the multimer's 21-wide layout instead,
  the answer moves 6.9e-02, which is what makes `embed_ptm_gate.py` able to
  see the column shift it exists for.
* `tplembed_ptm.cuh` is the monomer template embedder -- and the reason it
  is 250 lines rather than a second `tplembed` is that **the query never
  reaches the output**. Monomer `SingleTemplateEmbedding` reads
  `query_embedding` for its dtype and nothing else; all 88 of its features
  are template-only; and `TemplatePointwiseAttention` softmaxes over a
  template axis of length ONE, so the weight is 1 whatever the logit is. So
  the whole contribution is a constant with no backward, and the attention
  folds to one `[64, 128]` Linear. Both measured against bc2's module rather
  than argued: **max|d| 0.000e+00 on a seven-times-larger query**, and
  2.1e-06 for the fold.
* its pair stack runs the **attentions first** where the multimer runs the
  multiplications first (`tplblock::attn_first`). Running the multimer order
  on monomer weights gates at cos 0.919 -- a number that reads as a
  precision problem and is a reordering.
* `tpl1d.cuh::featPtmK` is the 1-D template row: all seven torsions, their
  alternatives under the naming ambiguities, and seven masks -- 57 features
  against the multimer's 34. Every DEFINED feature is bit-exact against
  bc2's `atom37_to_torsion_angles`. The undefined ones are NOT JUDGED and
  the gate says so: residue 0's previous atoms are zeros, AF2 does not
  replace them (`placeholder_for_undefined = not zero_init`, and zero_init
  is True), the Gram-Schmidt then subtracts two near-equal vectors and
  divides by `sqrt(|.|^2 + 1e-8)`, and a ONE-ULP move of the input
  coordinates flips the result by 2.0. There is no bound there that both
  means something and can be met.
* `ckpt.cuh::monomerise` does the layout transforms in C++, so
  `params=...ptm.npz` still needs no Python, and `ckpt_ptm_gate.py` holds it
  to `alphafold3/af2/convert.py` composed with bc2's own dump writers:
  4081 + 341 + 100 + 68 files byte for byte.

**And bc2's own regime runs: multimer design, AF2-ptm validation, one
process.** `Plan::monomer` is read at ALLOC time -- which template embedder
is built, where the outer product mean sits in both stacks, `position_scale`
-- and one DesignStep serves the whole design pool by rebinding weights, so
the validation pool gets its own pair:

```
  validation    2 held-out model(s), resident  (2 monomer, 0 multimer)
  validation    a SECOND step for the monomer (alphafold2_ptm) regime
                (predict-only)
  ...
      ensemble mean  i_pTM 0.3681  pLDDT 0.3874
```

Built only when the pool actually holds a model of the other kind, and
`Plan::lean` because validation never asks for a gradient -- which is what
makes a second complex-length object affordable. `ensembleEval` repoints
`step`/`stepb` per model and puts them back before it returns: the POINTERS
move, because every lambda in the design loop captures them by reference and
threading a step argument through `runLosses`, `predictBinderAlone`,
`computeActConst` and everything they call would touch a hundred lines to say
one thing. `monomer_now` moves with them, because `actConstFor` reads it --
the monomer embedder has no constant to hoist.

A mixed DESIGN pool is still refused, and that one is not the same gap:
`pickModel` installs a design model for a GRADIENT step and the second pair
has no backward buffers at all. `refusal_gate.py` checks that the drop is not
silent.

**And `monomer_campaign_gate.py` is not a parity gate**, which its own
docstring says before it says anything else. It reaches the branch and checks
the answer is in family. The gate that sees a WIRING error is
`ds_compare.py`, against the JAX step -- and that one found the only real bug
in all of this.

### The one bug, and the gate that was written for it and could not see it

`embed_ptm_gate.py` exists to catch ONE error: the monomer's `target_feat` is
22 wide with the 20 sequence columns at **1**, and reading them at 0 applies
`preprocess_1d`'s weight rows 0..19 where the model applies 1..20. The gate
measures that mistake itself, at relL2 **6.9e-02**. It reports 3.5e-08.

It passed while the DESIGN STEP was making exactly that mistake: `reColsK`
wrote 21 columns at offset 0 into a buffer the monomer embedder read 22 wide.
**The gate builds its own input**, so all it could ever test was the module.
`ds_compare.py` -- the whole native step against the JAX step it replaces,
read as a multiple of the model's own bf16 floor -- read 6-24x on the head
outputs; after the fix every field is 0.57-1.79x, where the MULTIMER reads
1.3-2.2x on the same instrument.

The bisection, in the order it was done, each step cheap:

```
  1. the hand-offs      post-trunk pair cos 0.998, msa cos 0.965
                        -> the msa track, not the pair
  2. the known-good     the MULTIMER trunk, same instrument, reads WORSE
     path               (msa row0 5.8e-02 against the monomer's 7.6e-02)
                        -> the trunk is not it
  3. the trunk's INPUT  monomer 3.0e-02, multimer 3.7e-03 -> 8x, pre-trunk
                        (captured by patching EvoformerIteration eagerly:
                        bc2's monomer modules.py has no native hook)
  4. an independent     agreed with JAX and not with the port, and the
     f32 restatement    error was 48% a per-channel constant -- which is
                        what a shifted weight row looks like
```

`ds_compare_gate.py` now runs the monomer chain in the suite. And `ffi.cu`
reads the regime off the WEIGHTS -- `preprocess_1d`'s row count, 22 against
21, the same shape test `ckpt.cuh::looksMonomer` uses -- rather than taking
an attribute: a new attribute would have to be threaded through every caller
in two repositories and could then disagree with the directory it was passed
beside.


### A multi-chain binder is not an oligomer, and six conventions say so

A Fab or an scFv is ONE binder spanning two scaffold chains with a SEQUENCE
EACH; `copies` is N chains sharing one. bc2 RAISES on the pairing
(`resolve_binder_chains`), so the two never co-occur and
`max(copies, scaffold chains)` is the chain count -- but BOTH name their
chains `binder_0, binder_1, ...`, so **the chain count does not say which
layout you are in**. Only the setting does, which is why `copies` now
travels in the dump: `loss_campaign_gate` counted distinct asym ids among
the binder rows, read a Fab as two copies, and applied `configure_oligomer`
-- `per_protomer=True` on three losses bc2 leaves merged.

Everything that reads "the binder chain" then splits again. Every one of
these was wrong, and every de novo and oligomer gate was green:

```
  the per-trajectory rebuild   scaf_chains defaulted to 1, so the target's
                               asym collapsed onto the second binder chain
  Backbone_Clashes             every chain pair; L-H counts        2 vs 39
  Scaffold_Framework_RMSD      one Kabsch PER CHAIN, pooled    16.79 vs 1.19
  Binder_Chain_Breaks          per chain; the L-H junction is not a bond
  Induced_Fit_TM               binder_0 alone                 0.178 vs 0.119
  Interdomain_*, Domain_Sep    binder_0 alone -- 55 rows, so bc2 returns
                               None and the port reported two filters bc2
                               does not run
  interface_pae (loss)         binder_0 x target            0.748 vs 0.763
```

The rule that makes them derivable: **anything taking a chain NAME gets
`binder_0`; anything calling `binder_copy_chains` fans out.** Key the masks
on the ASYM ID, never on `i < binder_len` -- those agree only while the
binder is one chain repeated.

**And one that was not multi-chain at all.** The per-trajectory rebuild
called `buildBinderAlone` WITHOUT the scaffold, so from the first trajectory
on, a VHH's free fold was an all-designed chain while the complex held the
framework -- two states disagreeing about what the binder is, which is the
thing `buildBinderAlone`'s own comment warns about. Nothing caught it
because no gate compares the binder_alone STATE: the dump carries its atoms
and not its flags. It surfaced only because a second call site had to change
for the chain layout.

`state_gate.py` now builds Fab.cif, scFv.cif and VHH.cif through bc2's own
`Protein.with_scaffold` and compares residue_index, asym, entity,
interface_asym, seq_mask and the DESIGN flags -- it caught the target's
entity id (2 behind two binder entities, not 1) -- and `multi-chain
filters` / `multi-chain losses` run the whole campaign on a two-chain
scaffold. That scaffold is a SMALL one (`make_test_scaffold.py`, 55 residues
a chain) because bc2's own Fab is 434 and this port's two states do not fit
beside a 115-residue target on a 24 GB card: a 440-residue DE NOVO binder
fails identically, so it is the existing length ceiling and not the layout.

## Pass four, fifth round: occupancy asked for directly

The BMF/BMG/BMS sweeps moved occupancy *indirectly*, through the tile size,
which conflates it with the work per block. `__launch_bounds__`'s second
argument moves it on its own: it tells ptxas how many blocks an SM must fit
and lets it cap registers to get there. Module forward+backward at N=256:

| | no hint | 2 | 3 | 4 |
|---|---|---|---|---|
| tri-mul bwd | 1.432 | **1.349** | 1.388 (spills 16 B) | — |
| attention bwd | 3.166 | 3.152 | **2.973** | 3.035 |

**But the first version of this was a wash**, and why is the useful part.
Hinting BOTH fused kernels moved the stack's backward 721.0 -> 718.1 and its
forward 286.2 -> 288.8: `projGateK` runs in the forward AND in the replay, so
capping its registers costs the forward about what it buys the backward.
`combineDxK` and `scatterDxK` are backward-only and have no such tax. Hinting
only those two:

```
                  fwd ms    bwd ms   fwd+bwd
  before           286.2     721.0    1007.2
  after            286.9     717.1    1004.1
```

Gradient unchanged (0.9787). Small, and it is the shape of what is left.

### Is there anything left? Measured, mostly no

Every kernel in the backward, against the ~450 GB/s this card gives:

| kernel | ms | GB/s | |
|---|---|---|---|
| `ta.b flashBwd` | 194.5 | — | FD-capped; its atomics are 23.5 ms |
| `tm.b combineDx` | 47.0 | 274 | **68% of roof** |
| `tm.f projGate` (replay) | 38.7 | 251 | **63% of roof** |
| `ta.b scatterDx` | 32.2 | 355 | 89% |
| `tm.b centerLnBwd` | 25.2 | 448 | at roof |
| `ta.b gate` | 20.0 | 403 | at roof |
| `tm.b dOutGate` | 16.7 | 483 | at roof |
| `tm.b gemm dcn` | 7.9 | 410 | at roof |

Everything is at the roof except `flashBwd` and the two kernels this pass
wrote. **The entire recoverable budget is ~33 ms, 4.5% of the backward**, and
eight hypotheses have now failed on it (layout, write granularity, tile size,
cold weights, warp rotation, one k-loop for all five projections, four tiles,
and -- half -- the occupancy hint).

And one fusion that looks like the three that worked, and is predicted NOT to:
`dOutGate` + `gemm dcn` moves 117.6 MB where the math needs 84, a ratio of
1.4x. The three that paid were 2.33x, 2.25x and 2.44x, and they realised
58-61% of their byte ratio. **At 65% efficiency the break-even ratio is about
1.55x**, so 1.4x loses -- which is also why the fused centre LayerNorm (1.97x
on paper, but spending 18 KB of shared and a block-wide reduction) lost.

## Pass four, sixth round: the keep tier the enumeration rejected

The enumeration priced tri-mul's activations at 6.7 ms/GB and rejected them.
That pricing was right and the conclusion was half wrong: it answered *should
this always be kept*, and the greedy budget exists to answer a different
question -- *is there room today*.

`KEEP_ACTS` holds `a`, `bb`, `gl`, `gr`, `t2` and the input LayerNorm's
`mu`/`rstd`, which is everything `lnFwd` + `projGate` would rebuild. It is
ranked last of the four tiers (136 / 104 / 22 / 6.7 ms per GB), so the greedy
budget grants it only out of memory the better tiers and the model did not
want.

```
                       bwd ms   fwd+bwd    VRAM
  standalone, off       717.1    1004.2   6.73 GB
  standalone, on        667.0     956.7  14.83      -7.0%
  in-model, off         963.5        --      14     (NE_TIME, PD1 binder)
  in-model, on          921.5        --    19.3     -4.4%
```

Gradient unchanged (0.9787 either way) -- the forward cannot change, since
keeping an activation is not computing a different one.

**It is OFF by default** (`NE_KEEP_ACTS=1`). 82% of the card for 4.4% is the
wrong default for a tool that runs multi-hour campaigns and whose weak side
is already memory; it is the right option for anyone with room. The budget
would refuse it on a smaller card or at a longer length anyway -- that is
what ranking it last is for.

**CORRECTION: the in-model figure above is a PARTIAL grant, not the tier.**
The allocator's own diagnostic reports the depth-set at **10.26 GB** at
N=288 (and ~12.3 with `t`), against a budget of 7.78 (11.76 GB free, 3.98
held back). In-model VRAM rose by 5.3 GB, not 10.26 -- so roughly half the
48 depths were kept and the -4.4% was bought with half the tier. The graded
budget doing exactly its job, but it means **the standalone -7.0% at N=256
is an upper bound that design lengths do not reach**, and the honest in-model
number is the partial-grant one. Quote 4.4%, not 7.0%, for a real design.

### ...and the flash backward's occupancy, settled

Pushing on `flashBwd` (194.5 ms, 27% of the backward) produced a correction
rather than a speedup, which is worth as much. The note above said
`cudaOccupancyMaxActiveBlocks` returns 3 "at the floor LDH=LDF=32 (25088 B)"
and concluded the 4th block was unreachable. Asking the driver at 128-byte
granularity instead of at two points, **the step is at 24576 B** -- and
25088 - 24576 = 512 is exactly the two `[BM]` float staging arrays, `lseS`
and `DS`. Deleting them does reach four blocks:

| LDH | LDF | smem | blocks | bwd ms |
|---|---|---|---|---|
| 40 | 40 | 30720 | 3 | 3.324 (2.973 WITH the staging) |
| 32 | 40 | 27136 | 3 | 3.541 |
| 40 | 32 | 29184 | 3 | 3.429 |
| 32 | 32 | **24576** | **4** | **3.701** |

Two findings. Staging lse and D is worth 12% on its own -- a per-element
global read inside the softmax loop is not free even out of L1. And the
padding floor that the 4th block requires costs more in bank conflicts than
+33% occupancy returns. **Occupancy is not what this kernel is short of**,
which is the fourth time this pass an occupancy experiment has come back
negative.

## Pass four, seventh round: the rule finds the last one

With the break-even rule in hand (fuse when the pair moves >~1.55x the bytes
the math needs, given a hand-written kernel at ~65% of cuBLAS streaming),
scanning every remaining adjacent producer-consumer pair in the backward is a
few minutes of arithmetic instead of a few days of kernels:

| pair | bytes now | bytes needed | ratio | |
|---|---|---|---|---|
| transition `gemm3` -> relu mask -> `gemm4` | 251 MB | 151 | **1.66x** | build it |
| `dOutGate` -> `gemm dcn` | 117.6 | 84 | 1.40x | no |
| `ta.b gemm dag` -> `ta.b gate` | 117.6 | 84 | 1.40x | no |
| `gemm dcn` -> `centerLnBwd` | 117.6 | 84 | 1.40x | measured, a wash |
| `combineDx` -> `tm.b lnBwd` | 184 | 168 | 1.1x | no |

**A THIRD CONDITION ON THE RULE, and the case that found it.** The
transition's FORWARD scores 5.0x -- LN + gemm1 + relu + gemm2 moves 167.6 MB
where the data is 33.6 -- which is three times the break-even and should be
the best fusion in the stack. It is not worth building, because a fused MLP
keeping `h` in shared stops being bandwidth-bound: 33.6 MB is 0.075 ms at the
roof, while the arithmetic is 17.2 GFLOP. cuBLAS does those two GEMMs at
~41 TFLOP/s (0.205 + 0.213 ms of the step's 0.663); the fused kernels in this
directory run 19-25 TFLOP/s, so the same work takes 0.69 ms and the fusion
LOSES to the pair it replaces.
​
So the rule needs all three: the pair must move >~1.55x the bytes the
mathematics needs, the intermediate must actually reach HBM, **and the fused
kernel must still be bandwidth-bound afterwards.** When the third fails the
fusion becomes a GEMM-efficiency contest against cuBLAS, which a hand-written
kernel here loses by 40%. That is the same arithmetic as the 65% realisation
factor, read from the other end.

Only one scored. `maskDxK` builds the masked `dr` a 128-column chunk at a
time in shared and contracts it against `w1t` -- the forward's own transposed
weight, which is already the `[K][N]` row-major a wmma `matrix_b` wants -- so
the masked `dr` never reaches HBM:

```
  pair transition, module fwd+bwd at N=256
    split   fwd 0.452   bwd 1.029
    fused   fwd 0.452   bwd 0.863     -16.1%
```

Gradient identical to nine digits (0.999207747 against 0.999207746), and the
same at N=128 and N=384. It is built for C = 128 only -- the PAIR transition,
44.6 ms of the backward; the MSA transition is 2.2 ms and keeps the split
path. `TR_SPLIT=1` restores it.

### Where the pass ends

N=256, S=2, depth 48, three runs each:

```
                        fwd ms    bwd ms   fwd+bwd    VRAM
  start of pass four     303.4     782.0    1085.4   6.97 GB
  the three fusions      297.6     756.8    1054.4   6.70
  + round two            287.5     734.7    1022.2   6.73
  + round three          286.2     721.0    1007.2   6.73
  + rounds five-seven    288.9     711.7    1000.6   6.73
  ...with NE_KEEP_ACTS   289.6     660.5     950.1  14.83
```

**Backward -9.0% by default, -15.5% with the activation keep.** Gates: trimul
0.999964, triattn 0.999970, transition 0.999208 (and 0.999192 / 0.999220 at
N=128 / N=384), block 0.993880, stack 0.978667 at S=2 and 0.986916 at S=1,
FFI 1.880e-01, padded 1.623e-01, primal-vs-vjp 0.000e+00.

### `t` joins the activation keep

The contraction's output `t` costs 1.6 GB over 48 depths and retires the
replay's batched GEMM, 11.5 ms -- **7.2 ms/GB, slightly better than the 6.7
of the tier it joins**, so it rides along rather than earning a tier of its
own. With `NE_KEEP_ACTS=1` the backward goes 660.5 -> 650.6 ms and the keep
16.45 GB; the default path is untouched.

The rest of tri-mul's replay prices worse and is not kept: `cn` and `o`
together would retire `centerLn` + `gemm out`, 16.8 ms for 3.2 GB = 5.2
ms/GB, below everything already in.

## Pass four, last round: pricing the rewrite instead of writing it

The one item left was `flashBwd` (194.5 ms, 27% of the backward) on
`mma.sync` with a swizzled shared layout -- swizzling is conflict-free with
no padding, so in principle it collects the 4th block for free. Days of work
on the most delicate kernel in the directory, so the prize is worth pricing
first.

`FB_PAD_SMEM` asks the launch for bytes the kernel does not use. That holds
the block count down and separates the two effects the LDH/LDF sweep
conflates:

| LDH | LDF | pad | blocks | bwd ms | |
|---|---|---|---|---|---|
| 40 | 40 | 0 | 3 | **2.980** | conflict-free baseline |
| 32 | 32 | 6144 | 3 | 3.564 | conflicts alone, **+19.6%** |
| 32 | 32 | 0 | 4 | 3.376 | the 4th block, **-5.3%** |

**The extra block is worth 5.3%; the conflicts that buying it costs are worth
19.6%.** A swizzled four-block kernel therefore lands at 2.980 - 0.188 =
2.792 ms, -6.3% -- which on flashBwd's share is **12 ms, 1.7% of the
backward**. For days of work on a kernel with a history of two
wrong-gradient incidents. No.

An earlier estimate of "up to 86 ms" came from comparing this kernel's 10.6
TFLOP/s against our own flash FORWARD's 19.2 on the same shape and assuming
occupancy was the gap. **Occupancy is 5% of it.** The rest is the serial
dependency chain inside a warp -- five `store_matrix_sync` -> shared ->
reload round trips a tile -- which more resident warps only partly hide.
That is a property of doing a flash backward with the WMMA API at FD=32, and
it is where this port stops.

## Pass four, in total

```
                        fwd ms    bwd ms   fwd+bwd    VRAM
  before                 303.4     782.0    1085.4   6.97 GB
  after                  288.9     711.0    1000.0   6.73
  after, NE_KEEP_ACTS    289.6     650.6     940.2  16.45
```

**Backward -9.1%, or -16.8% with the activation keep. Forward -4.8%. 240 MB
by default.** The gradient is unchanged at every step: stack `grad/pair`
0.9787 at S=2 and 0.9869 at S=1, from first commit to last.

Seventeen ideas were tried. Six paid: three fusions of a wide intermediate, a
fragment-contiguous weight layout, vectorised stores, `__launch_bounds__` on
the backward-only kernels, the transition's mask fusion, and the activation
keep tier. Eleven did not, and they are all written up above with their
numbers, because on a stack already at the memory roof **the useful output is
which arguments do not work on this shape.**

## And then a 22% that is not a kernel at all

With the stack at its floor, the question is what fraction of a design step
it even is. `NE_TIME` on a PD1 binder says: two forwards at 397.3 ms and one
backward at 968.1, so **1763 ms of the step is this stack**.

**CORRECTION to the step total.** An earlier version of this put the step at
~2180 ms and the non-stack remainder at 456 ms, from dividing a design run's
wall clock by its iteration count -- which includes the compile and the
validation. Differencing an 8-iteration run against an 18-iteration one
instead:

| | stack (NE_TIME) | step | non-stack |
|---|---|---|---|
| num_recycle=1 | 1762.7 ms | 2042.0 | **279 ms (13.7%)** |
| num_recycle=0 | 1365.4 ms | 1601.4 | **236 ms (14.7%)** |

So the stack is **85-86%** of a design step, not 79%, and the embedder,
template pair stack, structure module and heads together are 236-279 ms.
Porting all of them at the 1.3x this port achieves would save about 55 ms --
**3.4% of a step.** That is the number that says not to.

One of those two forwards is the DETACHED recycle pass. `_RECYCLE_LAST`
already sets `recycle_backprop=False`, so its gradient is never taken -- but
it still runs, at a full 397 ms. `num_recycle` is 1 on the af2 binder preset,
and AF2 counts EXTRA passes, so 1 means two trunk passes and 0 means one.

```
  RECYCLES=1   fwd n=12 (2 a step) @ 397.3 ms + bwd 968.1  =  1763 ms
  RECYCLES=0   fwd n=6  (1 a step) @ 412.6 ms + bwd 968.2  =  1381 ms
```

**382 ms a step, 22%**, and the warm wall clock agrees: 2.3 -> 1.8 min over
2 seeds x 30 iterations. Measured a third way, per step with the compile
differenced out (8 iterations against 18): **2042.0 -> 1601.4 ms, -21.6%**. (The first RECYCLES=0 draw read 3.0 min and looked
like a REGRESSION -- that was a cold compile against a cached one. The
compilation cache makes any total-time comparison between two configs
meaningless here; the call count is what settles it.)

### The outcomes, and why this is not a recommendation

Two draws each side, 2 seeds x 30 iterations:

| | i_con mean | i_con best | plddt mean |
|---|---|---|---|
| RECYCLES=1 | 2.8697 , 3.5304 | 2.6529 , 3.3442 | 0.585 , 0.597 |
| RECYCLES=0 | 3.1176 , 2.9992 | **2.0544 , 1.8253** | 0.552 , 0.537 |

The objective gets BETTER without the recycle -- best i_con 1.94 against 3.00,
well outside the 0.66 spread two identical runs show -- and pLDDT gets WORSE,
0.544 against 0.591.

**That combination is a warning, not a win.** i_con is a contact loss read off
the predicted structure; a model given less refinement is a model that is
easier to satisfy, and "the objective improved while the structure's
confidence fell" is the same signature as the stage-1 decode gap, where a
soft sequence scores well and its argmax does not. The honest reading is that
0 recycles makes the loss easier, not the binder better, and the only way to
tell those apart is the acceptance protocol (3 seeds x 125) plus an
orthogonal validator on the resulting sequences.

So: **measured, documented, and nothing changed.** `RECYCLES=0` is worth 22%
of a design step if it survives that, which is a bigger prize than this whole
pass, and it is a model-quality decision rather than a kernel one.

### The common-evaluator test, and an underpowered first attempt

`m.validate()` re-predicts `result['seq']` with the full model -- diffusion
and confidence heads on -- so it scores both arms by the same code and the
design-time recycle setting never reaches the scoring. That is the clean test
for "easier loss or better binder" and needs no external validator.

**The first attempt was underpowered and would have produced a conclusion.**
At 50 iterations both arms came back as failures -- i_ptm 0.12-0.16, ipSAE
0.01-0.05, pLDDT 36-44 against a passing binder's 0.73 and 0.87. The
schedule is `three_stage(75, 45, 5)` = 125 iterations and the interface forms
during the TEMP stage around 120, so everything measured was the soft stage,
before the design starts working.

What it did establish, weakly, is that the worry points the wrong way:

```
                  i_ptm           ipSAE           pLDDT      wall
  num_recycle=1   0.160 0.116     0.049 0.011     36.0 43.5  5.0 min
  num_recycle=0   0.206 0.368     0.127 0.427     46.9 37.0  4.0 min
```

Under the same evaluator the 0-recycle arm is BETTER on i_ptm and ipSAE, not
worse. Both are failures, so this is not an answer -- but "0 recycles games
the objective" predicted the opposite and did not get it.

**One caveat that the powered run cannot remove**: the recipe that produced
the first passing binder used `recycles 1`, so the whole ramp was tuned
around it. A fair test of 0 recycles would re-tune the ramp. What the 125-
iteration run can settle is the narrower question that matters in practice --
whether dropping the recycle costs a pass at the settings everything else was
tuned for.

### The answer: 0 recycles is faster AND docks better

3 seeds x 125 iterations (the full `three_stage(75,45,5)` ramp), both arms
scored by `m.validate()` -- the full model with diffusion and confidence
heads on, so the design-time recycle setting never reaches the scoring:

| | i_ptm | ipSAE | pLDDT | i_pae | wall |
|---|---|---|---|---|---|
| num_recycle=1 | 0.167 0.314 0.118 | 0.015 0.219 0.016 | 56.7 55.5 34.2 | 22.5 19.6 25.8 | 13.2 min |
| num_recycle=0 | 0.513 0.393 0.280 | 0.415 0.414 0.229 | 50.6 50.8 42.5 | 14.2 19.4 21.2 | **10.3** |
| mean | **0.395** vs 0.200 | **0.353** vs 0.083 | 48.0 vs 48.8 | **18.3** vs 22.6 | **-22%** |

**My hypothesis was wrong.** I predicted that dropping the recycle would make
the loss easier to satisfy without making the binder better -- objective up,
structure down, the stage-1 decode signature. The common evaluator says the
opposite: i_ptm doubles, ipSAE quadruples, i_pae falls, pLDDT is unchanged,
and EVERY 0-recycle seed beats the 1-recycle MEAN on both interface metrics.
The 50-iteration run pointed the same way before it, so this is two
independent runs agreeing.

**What it does not establish.** No seed in either arm produced a passing
binder (the bar is pLDDT > 70 and i_ptm > 0.5; one 0-recycle seed reached
i_ptm 0.513 at pLDDT 50.6). Binder design has a low per-seed hit rate --
15/25 and 3/25 on other campaigns here -- so 0/3 is unremarkable rather than
a regression, but it means this compares the DISTRIBUTION OF FAILURES, not
pass rates. A pass-rate comparison needs ~20 seeds an arm, about four hours.
And the ramp was tuned with recycles 1, so 0 is being judged on someone
else's settings -- which, if anything, understates it.

So: **worth adopting, pending that pass-rate check.** It is 22% of every
design step, it is one line in the preset, and on the evidence it costs
nothing in quality and may add some. It is the user's call, and it is a
bigger number than every kernel in this directory combined.

### Four elements a lane, not two

The vectorisation that won earlier packed two elements per lane. Four is the
same argument again -- one 8-byte access where there were two 4-byte ones --
and it applies to every element loop in the three fused kernels: `projGateK`'s
two gate writes and its `t2` write, `combineDxK`'s six-plane build and its
`dx` write, `scatterDxK`'s `dcomb`/`dKf`/`dVf` reads and its `dx` write, and
`maskDxK`'s masked build and write.

| module, N=256 | 2-wide | 4-wide |
|---|---|---|
| tri-mul fwd | 0.868 | **0.853** |
| tri-mul bwd | 1.354 | 1.350 |
| attention bwd | 2.974 | 2.966 |
| transition bwd | 0.863 | **0.825** |

The transition gains most (-4.4%) because its build reads two planes and
writes one across four k-chunks, so the loop is the whole kernel. Stack:
forward 288.9 -> 287.6, backward 711.0 -> **705.5**. Output and gradient
bit-identical everywhere -- cos 0.999983610 and 0.999963581 on tri-mul,
0.999969925 on attention, 0.999207747 on the transition, to the digit.

Diminishing, and that is the point: the first widening was worth 4.2% of the
tri-mul backward and the second 0.3%. There is no third.

### The backward, kernel by kernel, with nothing left above 1.2%

After the widening, the backward's own kernels (the replay excluded):

| kernel | ms | GB/s | state |
|---|---|---|---|
| `ta.b flashBwd` | 194.5 | — | FD-capped; rewrite priced at 1.7% |
| `tm.b combineDx` | 40.8 | 315 | 70% of roof |
| `ta.b scatterDx` | 30.4 | 376 | 84% |
| `tm.b centerLnBwd` | 25.2 | 320 | 71%; second pass priced below |
| `tm.b bmm` | 22.4 | — | cuBLAS batched, 37 TFLOP/s |
| `ta.b gate` | 20.0 | 403 | at roof |
| `tm.b dOutGate` | 16.8 | 483 | at roof |
| `ta.b lnBwd` / `tm.b lnBwd` | 26.6 | — | at roof |
| `gemm dcn` / `gemm dag` | 15.5 | 410 | at roof |
| `d:opm` | 13.5 | — | 61 TFLOP/s, at the COMPUTE roof |

**`centerLnBwd`, priced and not built.** It reads `t` and the cotangent
twice -- once for the mean, once for the write -- so 84 MB where 50.4 would
do: 1.67x, above break-even, and still bandwidth-bound after. `PROBE_ONEPASS`
makes the second pass read a cache-resident slice (wrong arithmetic, right
traffic) and says the second pass costs **0.087 ms a call = 8.4 ms of the
backward, 1.2%** -- the upper bound. A staged version collects less than
that, this directory's realisation rate is ~60%, and the closest precedent is
the fused centre-LayerNorm FORWARD, which made the identical trade and lost.
~5 ms expected, for a kernel needing a cross-thread reduction. Not built; the
probe stays so it can be re-priced in thirty seconds.

**`d:opm` is at the COMPUTE roof**, which is worth recording because it looks
cheap: 13.5 ms for 48 calls moving 16.8 MB reads like 60 GB/s and an easy
win. It is not -- the outer product mean's backward contracts [N][N][128]
against the [128][1024] output projection, 17.2 GFLOP a call, which at
0.28 ms is **61 TFLOP/s**. Half of this card's bf16 peak, from cuBLAS, on a
kernel nobody has touched.

So the backward is finished: one item above 1.2% (`flashBwd`, needing a
different instruction set for 1.7%) and ~33 ms spread across two kernels that
have resisted ten structural hypotheses.

## What the roof actually is

Every "at the roof" claim above rested on ~450 GB/s, taken from a copy kernel
written for another purpose. `bandwidth.cu` measures it properly -- 256 MB,
`float4` accesses:

```
  copy (read+write)    1.099 ms    489 GB/s
  read only            0.494 ms    544 GB/s
  write only           0.541 ms    496 GB/s
```

So the mixed read+write roof is **489 GB/s**, 9% higher than assumed, and
the kernel table should be read against that: `dOutGate` at 483 is at 99%,
`gemm dcn` at 410 is at 84%, `ta.b gate` at 403 is 82%, `scatterDx` 77%,
`centerLnBwd` and `combineDx` 65%, `projGate` 51%.

**But 489 is not reachable by every access pattern, and that is the real
answer to "is there more".** The benchmark gets it with 16-byte `float4`
accesses to contiguous memory. `projGateK` writes CHANNEL-MAJOR out of a
16-row stage tile, so a channel's contiguous run is 16 rows = **32 bytes**,
full stop -- the transaction size is capped by the stage, not by how many
elements a lane handles. Going from 4 elements a lane to 8 (a 16-byte store)
was measured and is WORSE, 0.857 -> 0.895 ms: with eight rows a lane only two
lanes' worth of rows exist per channel, so consecutive lanes step the CHANNEL
and the wide stores scatter M*2 bytes apart. Remapping does not rescue it,
because the run is still 32 bytes. Lengthening it means staging more row
tiles, which is the RSTEP experiment that lost on occupancy.

So the honest figure for this access pattern is the one the pure-copy probe
gave for the same layout -- 404 GB/s -- and `projGateK` at 0.423 ms is at
**59% of its pattern's floor, not 51% of a roof it cannot reach**. The 489
number is still worth having: it is the right denominator for the row-major
kernels, and it says `gate`, `gemm dcn` and `scatterDx` have 16-23% in them
rather than the 9-11% the old roof implied.

### Going after the row-major kernels, and three negatives

With a real roof (489 GB/s read+write) the row-major kernels looked like they
had 16-23% in them, against the 9-11% the old 450 implied. Three attempts to
collect it, all measured, all negative:

1. **`scatterDxK`'s `dcomb` read at 16 bytes** -- FAULTS. `TOT = 516` is not
   a multiple of 8, so a `dcomb` row is not 16-byte aligned and a `float4`
   load off it is a misaligned address. It surfaced as a silent crash with no
   output. 4 x bf16 needs only 8-byte alignment and 516 is a multiple of 4,
   so that is the widest legal read of that array.
2. **`scatterDxK`'s `dKf`/`dVf` read at 16 bytes** -- legal (stride C = 128)
   and SLOWER: module backward 2.962 -> 3.123 ms.
3. **`dGateK` vectorised 1 -> 4 elements a thread** -- the last genuinely
   scalar kernel in the backward, 65536 blocks of 128 with one element each,
   and the remap even preserves the per-head shuffle reduction (a head
   becomes eight consecutive lanes, `shfl_down 4,2,1`). Three alternating
   repeats: 4-wide {3.063, 2.990, 3.170}, scalar {2.981, 3.146, 2.966}.
   **No gain**, with a 0.2 ms spread swamping both.

**So access width is not what these kernels are short of** -- which the
channel-major side had already said, and which now holds for a kernel that
was fully scalar. One more denominator check, since the 489 figure comes
from a TWO-stream copy and `dGateK` touches five arrays:

```
  copy (1r + 1w)        489 GB/s
  five streams (3r + 2w)  470
```

4% apart, so the stream count is not it either. `dGateK` at 403 is **86% of
its own pattern's roof**, `scatterDx` 80%, `combineDx` 67%. The row-major
headroom is real but it is 10-20%, not 23%, and nothing that changes how the
bytes are addressed reaches it.


### What the backward kernels are actually bound by, probed

Three probes, each replacing a kernel's global reads with a cache-resident
slice -- wrong arithmetic, right instruction mix and write traffic -- so the
difference is what those reads cost. They are independent: taken together
the savings add (-0.085, -0.083, -0.176 on the module backward), so each
measures its own kernel.

| probe | traffic removed | time saved | reading |
|---|---|---|---|
| `PROBE_CDX` (combineDx's six planes) | 100.8 MB | 0.085 ms | 40% exposed -- **MMA-bound** |
| `PROBE_ONEPASS` (centerLnBwd's 2nd pass) | 33.6 MB | 0.083 ms | **more than its bytes** -- latency |

`combineDx` is the important one. 0.425 ms a call reads either as 315 GB/s
(67% of roof) or 25 TFLOP/s (20% of peak), and those imply opposite fixes.
Removing three quarters of its input traffic saves a fifth of its time, so
it is **MMA-bound**, and no access-width or traffic work can reach it. Its
load:mma ratio is 5:4 -- one weight fragment and four activation fragments
per four mma -- and improving that needs either more row tiles a warp
(BMG=128, one block an SM, measured worse) or more column tiles a warp
(fewer warps, also worse).

`centerLnBwd`'s second pass costs 0.083 ms against 0.071 ms of traffic, so
it is latency rather than bandwidth and should be recoverable. **It is not.**
Staging the values in REGISTERS between the two passes -- four threads a
position, 32 channels each, 16 registers an array packed as bf16x2, no
shared memory and no spills -- measures **1.745 ms against 1.353, 29%
WORSE**. Four threads a position means consecutive threads cover different
CHANNEL ranges, so a warp's read fragments into four segments of eight
elements where the original has thirty-two contiguous. The coalescing loss
dwarfs the saved re-read, and the obvious fix (consecutive threads to
consecutive `ij`) puts a position's four lanes in different warps, which
breaks the shuffle the reduction needs.

So both remaining kernels are at a structural limit, and it is the same one
`flashBwd` hit: not bytes, not access width, not occupancy, but the
arrangement of work within a warp. That is an instruction-set change, and it
was priced at 1.7% where it was measurable.


### Where the backward actually is, measured per module

Every module driver has had `PROF_STEP` brackets all along and silently
dropped them: only `stack.cu` and `ffi.cu` ever called `prof().report()`.
A destructor on `Prof` reports anything still accumulated at exit, so
`BLK_PROF=2 ./triattn ...` now profiles, and so does every other driver.

Per module at N=256, and per BLOCK (the stack runs two trimul and two
triattn a block):

| module | x/block | fwd ms | bwd ms | bwd/fwd |
|---|---|---|---|---|
| **triangle attention** | 2 | 0.845 | **2.687** | **3.18x** |
| triangle multiplication | 2 | 0.855 | 1.345 | 1.57x |
| pair transition | 1 | 0.453 | 0.829 | 1.83x |
| outer product mean | 1 | 0.240 | 0.334 | 1.39x |
| msa row attention | 1 | 0.162 | 0.336 | 2.07x |
| msa column attention | 1 | 0.036 | 0.052 | 1.44x |
| msa transition | 1 | 0.026 | 0.030 | 1.15x |

9.65 ms a block, and **triangle attention is 5.37 of it, 56%**.  Trimul,
which the whole last optimisation pass went into, is 19%.  Triangle
attention is also the only module whose backward is not ~2x its forward.

Inside it (`BLK_PROF=2`; the two forward GEMM slots are inflated by
one-time cuBLASLt autotuning, the backward slots sum to the unprofiled
total and are clean):

| kernel | ms/call | share of bwd |
|---|---|---|
| **flashBwd** | **1.84** | **65%** |
| scatterDx | 0.31 | 11% |
| gemm dag | 0.23 | 8% |
| gate | 0.21 | 7% |
| lnBwd | 0.14 | 5% |

flashBwd is 1.84 x 2 x 48 = **177 ms, a quarter of the whole 705 ms
backward**, and 4.7x its own forward.  It is the only thing left in the
backward big enough to be worth a rewrite -- and three measurements this
pass say it is not available:

* **The K/V load is 12% and it is BYTES, not latency.** `PROBE_KV` (pin the
  tile to j0 = 0 so it comes from L1) returns 2.852 -> 2.506 ms.  A register
  prefetch pipeline -- next tile loaded into registers right after the
  barrier, consumed at the top of the next iteration, no extra shared, so no
  occupancy cost unlike cp.async -- is bit-identical and **a wash**, 2.739
  against 2.723.  The arithmetic that should have come first: 32 KB a block
  over 4096 blocks is 128 MB a call, and 128 MB at this card's 489 GB/s is
  0.26 ms, essentially the whole probe.  `FB_PREFETCH=1` keeps it for a card
  with a different L2.
* **The atomics are 6%, not the 32% the old stack-level note says.**
  Re-ablated on the module: `NO_DKV` 2.695 -> 2.599 (3.6%), `NO_DNBB`
  -> 2.625 (2.6%).  Swapping the loop order so a block owns a KEY tile
  would retire dK/dV's atomics and create dQ's, so it is worth ~1.8% -- for
  a restructure of the kernel with the worst wrong-gradient history here.
* So flashBwd is ~94% core MMA+softmax, which `FD = 32` pins to five shared
  round-trips a tile.  That needs `mma.sync`, priced at 1.7% above.

And the same probe method closes trimul.  `PROBE_LDS` makes `combineDx`'s
matrix_a address kk-invariant so the compiler hoists all four loads out of
the kk loop -- identical mma count, one eighth the LDS traffic -- and saves
**0.036 ms, 2.7%**.  So combineDx is bound neither by HBM (`PROBE_CDX`) nor
by the LDS path, which prices `ldmatrix` there under 2.7% too.

**Every remaining lever in the backward is measured at 3% or less.**

### And where the 48-deep STACK's backward goes, which is not the same thing

The module sum is 9.65 ms a block, but the stack runs at 13.35 -- and its
FORWARD is 5.94 against the modules' 4.32, the same 1.38x.  A factor that
lands equally on both sides is not rematerialisation, so do not infer it by
subtraction.  `BLK_PROF=1` measures it instead (the `d:` rows are the real
backward, the bare rows are the forward REPLAY the backward runs first):

| | ms | share of bwd |
|---|---|---|
| **rematerialisation, all nine modules** | **206.6** | **30%** |
| d:triattn_end + d:triattn_start | 293.6 | 42% |
| d:trimul_in + d:trimul_out | 125.9 | 18% |
| d:pair_trans | 35.4 | 5% |
| d:opm + d:msa_* | 37.4 | 5% |

Two things follow.  Triangle attention is **42% of the stack's backward**,
which is the module-level finding confirmed at depth -- and the kernel
inside it is closed, above.  And the replay is the single biggest item,
bigger than flashBwd's 177 ms.

The replay is NOT reducible by skipping work inside it.  It already does:
recomputing both triangle attentions costs 71.7 ms against the 119.0 the
same two cost in the real forward, because the keep tiers let it skip what
they hold.  What it cannot skip is a whole module, because module i's
replay consumes module i-1's output -- the chain is why only block INPUTS
are saved.  So the only lever on the 206.6 ms is the save-set, and that is
already a measured ms/GB greedy (KEEP_STATS 136, KEEP_BIAS 104, KEEP_FLASH
22, KEEP_ACTS 6.7) spending a budget that is 16.44 GB of 23.68 here and
tighter in a real design step.  **Remat is bounded by VRAM, not by code.**

### The largest lever in the backward is not a kernel

`NE_KEEP_ACTS=1` is **-8.3%** (703.31 -> 644.91 ms at N=256 depth 48,
gradient equivalent) and it is off by default because it costs 9.71 GB.
That number was nearly retracted as stale: an A/B run as `NE_KEEP_ACTS=0`
against `NE_KEEP_ACTS=1` measured 640.6 vs 643.8 and looked like a dead
tier.  Both arms had it ON -- every boolean flag here was `getenv(X) !=
nullptr`, so `X=0` enables.  `envOn()` now honours the value, across all
seventeen flags.  **Any A/B in this module run before that commit with
`FLAG=0` as the control arm is wrong.**

To profile a kernel whose first call pays a one-time cost (cuBLASLt
autotunes, and reads as 73% of the trimul forward if you let it), run
`BLK_PROF=2` at two rep counts and difference them.  Steady state, ms a
trimul call:

| forward | | backward | |
|---|---|---|---|
| projGate | **0.35** | combineDx | **0.39** |
| centerLn | 0.15 | centerLnBwd | 0.26 |
| bmm | 0.11 | bmm | 0.22 |
| outGate | 0.10 | dOutGate | 0.17 |
| lnFwd | 0.07 | lnBwd | 0.14 |

**projGate is 3x the bmm**, which is the opposite of what the name suggests
and the reason `haveActs` is worth what it is: it skips lnFwd + projGate +
bmm = 0.53 ms a call, 66% of that the projection, and 0.53 x 96 calls is
the 58 ms the tier returns.

The tier's size is now fully accounted -- six tensors (a, bb, gl, gr, t2
and t) x 16.8 MB x 2 trimul x 48 depths = 9.67 GB against 9.71 measured --
which made a per-TENSOR split look attractive: keep `t` alone (1.61 GB) and
the replay skips the bmm.  It is not.  Keeping the projGate group prices at
5.0 ms/GB and keeping `t` at 6.6, against 6.4 for the whole, because the
tier is homogeneous.  Confirmed rather than inferred, with the per-DEPTH
partial grant the allocator already has:

| depths granted | bwd ms | saved |
|---|---|---|
| 48 of 48 | 641.1 | 62.2 |
| 25 | 672.6 | 30.7 |
| 10 | 694.4 | 8.9 |
| 3 | 701.7 | 1.6 |

Linear.  **So the backward is bounded by memory, not by code**: every kernel
lever is <=3%, and the one lever worth 8.3% is bytes.

### KEEP_ACTS on by default, and 2.46 GB off the price of it

`NE_KEEP_ACTS` now defaults ON.  The greedy budget already degrades it
safely -- a card that cannot afford the depth-set gets a partial grant whose
value is linear, and one that can afford nothing replays as before -- so the
default costs nothing it cannot give back.  `NE_KEEP_ACTS=0` opts out.

Then the tier itself got cheaper, by noticing that **three of its six
tensors are sigmoids**.  `gl`, `gr` and `t2` are all written by `projGateK`
as `sigmoidf(...)`, so they live in [0, 1], where a uint8 is uniform to
1/255 while bf16's 8-bit mantissa is 1/256 RELATIVE -- coarser than the
uint8 above 0.5 and finer below.  `gl`/`gr` are read only by the backward,
so narrowing them cannot touch the forward at all; `t2` multiplies the
module output, so it has its own flag (`TM_GATE8`, `TM_GATE8T`).

Stack, N=256 depth 48:

| | VRAM | fwd ms | bwd ms | out/pair relL2 | grad/pair cos |
|---|---|---|---|---|---|
| bf16 gates | 16.44 GB | 285.9 | 643.6 | 9.107e-03 | 0.978696 |
| gl/gr uint8 | 14.80 | 285.4 | 643.1 | 9.107e-03 | 0.978606 |
| + t2 uint8 | **13.98** | 285.1 | **641.6** | **9.014e-03** | **0.978818** |

**-2.46 GB, -15%, and not slower -- faster.**  The tier goes 9.72 -> 7.26 GB
a depth-set, so its rate improves from 6.4 to 8.6 ms/GB.  At module level the
narrowing is a clear win on time as well, because `combineDx`'s gate read
becomes four gates in one 4-byte load where bf16 fetched two: trimul fwd
0.857 -> 0.836 (-2.5%), bwd 1.351 -> 1.306 (-3.3%).

Accuracy does not degrade and at stack level reads very slightly BETTER on
both sides, which is run-to-run noise either way; the point is the sign.
Gated at every level -- trimul grad/pair 0.999961 against 0.999964, the full
block out/pair relL2 6.888e-03 against 6.887e-03 and grad/pair 0.994950
against 0.994954, and triattn, pair_transition, opm and msa_row untouched.

`a`, `bb` and `t` are the other three tensors and they are NOT sigmoids, so
they stay bf16.  That is the whole of what this trick reaches.

### `t` gets its own tier, because the greedy's unit was wrong

Once the three sigmoids are uint8 the acts set is no longer homogeneous.
Measured at N=256 depth 48 by dropping each half:

| | VRAM | bwd ms | saved | rate |
|---|---|---|---|---|
| both tiers | 13.98 GB | 639.16 | | |
| drop `t` only | 12.37 | 652.84 | `t` 13.7 ms / 1.61 GB | 8.5 ms/GB |
| neither | 6.68 | 705.53 | five 52.7 ms / 5.69 GB | 9.3 ms/GB |

Inside ONE tier the budget can only trade whole DEPTHS, so a card with
5.69 GB spare had to take all six tensors for 78% of the depths (51.8 ms)
when it could take the better five for all of them (52.7).  `KEEP_TMUL_T`
ranks after `KEEP_ACTS`, so the greedy drops it first.  Both follow
`NE_KEEP_ACTS`.

**This is worth about 2%, not the 8% I first wrote.**  That estimate priced
`t` at the 10.6 ms its bmm costs STANDALONE, giving 6.6 ms/GB; in the stack
it is 13.7 ms, so the two halves are far closer in value than they looked.
The split is kept because the ranking is now correct and automatic, not
because it buys much.

It needed one new flag through the call chain -- `haveT` beside `haveActs`,
so `tm.f bmm` is skipped on its own rather than with the projection -- and
the four combinations are all consistent: with the projection set kept and
`t` dropped the bmm re-runs from the KEPT `a`/`bb`; with `t` kept and the
projection set dropped, `projGate` rebuilds `a`/`bb` and the kept `t` is
still the same contraction of them.

### Four harness faults in one session, every one of them a reported number

Each of these ran, produced a plausible number, and was measuring something
other than what it said.  They are recorded together because the family
matters more than any one of them.

1. **`getenv(X) != nullptr`** meant `X=0` turned a flag ON.  An A/B of the
   activation-keep tier had it enabled in both arms and read as "the
   documented win no longer delivers".  `envOn()` now honours the value.
2. **A stale `.so`.** `backend.ready()` checked only that
   `libnative_evoformer.so` exists.  It is gitignored, so any newer source
   means the process runs code that is not in the tree.  An in-model A/B ran
   both arms against a library three hours older than the change under test.
   `ready()` now names the newer files and refuses.
3. **`step_bench --length` builds a MONOMER**, which the native backward
   refuses, so the run measured XLA end to end.  It now takes
   `--contig`/`--pdb` and warns when the protocol is monomer.
4. **`step_bench` never called `install_hook.install()`.**  Native is a hook
   a harness installs, not a default -- `design_outcomes.py` and
   `compare_design.py` do it, this script never did.  So EVERY number it has
   ever printed was XLA, monomer or not.  The tell was `NE_TIME=1` printing
   nothing at all.  With `--native`, on the same PD1 binder at length 224:

   | | ms an iter |
   |---|---|
   | XLA | 1618.8 |
   | native | **1129.3** |

   1.43x, with the backward 494 ms of it.  `--native` refuses with
   `ready()`'s reason rather than falling back, which is the point.

And with the harness finally measuring the thing it names, the
activation-keep default pays in-model too.  PD1 binder, length 224, native
both arms:

| | ms an iter | backward |
|---|---|---|
| `NE_KEEP_ACTS=0` | 1181.5 | 544.0 ms |
| **default (on)** | **1139.6** | |

**-3.5% of the whole design step**, which on a 544 ms backward is -7.7% --
the standalone -8.3% arriving where it was supposed to.  Both tiers are
granted in full at this length: acts is 4.35 GB a depth-set after the uint8
gates and `trimul t` 1.23 GB, against a 12.25 GB budget once the 3.58 GB
margin is held back.

The shared shape: **a measurement that cannot fail loudly will fail quietly.**
Three of the four were only caught by asking what the run should have printed
if it were doing what it claimed -- a tier diagnostic, an `NE_TIME` line, a
protocol name -- and noticing it was absent.

### The softmax never needed to go through shared

`flashBwd` is **192.9 ms of the stack's 635 ms backward, 42.6% of every
kernel the profiler sees** and four times the next one, so it is the only
place left where a rewrite repays itself.  Everything above priced the
`mma.sync` rewrite at 1.7% -- and that pricing was about OCCUPANCY and bank
conflicts, which is not what the file itself says the kernel is.  What it
says is "five store_matrix_sync -> shared -> reload round trips a tile".
Nothing had priced *removing one*.

The arithmetic says the round trips are not about bandwidth: 160 KB of
shared traffic a block over 4096 blocks is 655 MB, and at this card's ~13
TB/s of shared that is 0.05 ms of a 2.0 ms kernel.  It is the barrier and
the write-to-read dependency, so the fix is to have one fewer, not a cheaper
one.

**And the softmax is elementwise.**  The forward already stored the row
log-sum-exp, so `P = exp2(S*log2e + bias - lse)` needs no reduction at all --
every element of P is a function of the element of S in the same place.  S
arrives in a wmma accumulator, so it only goes through shared to reach the
"lane == key" layout the old loop wanted.  Measure the accumulator's
element -> (row, col) map instead of assuming it (`fraglayout.cu`: fill a
tile with its own linear index, `load_matrix_sync` it into an ACCUMULATOR
fragment, read back what each lane holds):

    row = (lane / 4) + 8 * ((e / 2) % 2)
    col = (lane % 4) * 2 + (e % 2) + 8 * (e / 4)

and the softmax runs in registers, P going straight out as half2 since e and
e+1 are adjacent columns.  Gradient bit-identical, 0.999970716 to the digit.

| | triattn bwd | stack bwd |
|---|---|---|
| through shared | 2.746 ms | 641.05 ms |
| in registers | **2.486** | **620.37** |
| | **-9.5%** | **-3.2%** |

(three reps each; the module number swings ~7% run to run, which is why it
is a mean and not the best draw.)

**The same change on the OTHER accumulator loses, by a lot: 3.802 ms against
2.822.**  `dS = P * (dP - D)` is elementwise too and dP arrives the same way,
so "do both" is the obvious next thought.  The difference is where the
outputs go.  P is written to SHARED, where the fragment layout costs nothing
-- `LDH = 40` happens to put the 32 lanes of that layout on 32 distinct banks.
dS writes the **dnbb atomics to global**, and 32 contiguous keys becoming
eight rows of scattered pairs costs far more than the round trip saves.
`FB_REGDS=1` re-runs it.

The lesson generalises past this kernel: **a fragment-layout conversion is
free when its consumer is shared memory and expensive when its consumer is a
coalesced global access.**

### A fully-granted tier makes the module's own buffer dead

`use(i)` points each kept slot at the tier arena for every depth the tier
covers and falls back to `forig[k]` only for the depths it does not.  So when
`tdepth == depth` that fallback is unreachable and the buffer `alloc()` made
for the module is never read again -- and it was still being held.  **189.8 MB
at N=256 depth 48**, growing as N^2 like everything else, so ~420 MB at
N=384, out of exactly the headroom the margin exists to protect.

13.98 -> 13.79 GB, gradient unchanged (0.978712), and the partial-grant path
still correct (`KEEP_BUDGET_MB=4000`: 0.978725 at 8.81 GB).  `forig` is nulled
rather than left dangling, so a mistake here is a null dereference instead of
silent corruption.

Where the remaining scratch is (`BLK_MEM=1`, one block's worth, which is all
the stack holds):

| module | MB | | module | MB |
|---|---|---|---|---|
| triattn_end | 503.3 | | trimul_out | 285.2 |
| triattn_start | 400.6 | | pair_trans | 201.3 |
| block | 306.2 | | opm | 119.5 |
| trimul_in | 287.3 | | msa_* | 54.5 |

2.16 GB in all.  The modules run strictly in sequence, so the BACKWARD-only
buffers in that table (`da`/`db`/`dt`/`dcnT`, `dcomb`, `dr`, ...) are live
only during their own module's call and could share one arena -- roughly
600 MB of the 2.16, so ~450 MB recoverable.  Not done: it needs an allocator
every module routes through, and the forward intermediates mixed in with them
CANNOT share, because the replay writes all nine modules' before the first
backward reads any.

### Where this session left the kernels

Every gate green and at its historical value (grad/pair, N=256): trimul
0.999961 / 0.999963, triattn 0.999971 / 0.999970, pair_transition 0.999208,
msa_transition 0.998968, opm 0.999985, msa_row 0.998855, msa_col 0.997153,
block 0.994950, stack-48 0.978730.

|  | before | after | |
|---|---|---|---|
| stack backward, N=256 d48 | 705.5 ms | **617.8** | **-12.4%** |
| VRAM, same | 16.44 GB | **13.79** | **-16.1%** |
| in-model backward, PD1 L=224 | 544.0 ms | **480.0** | **-11.8%** |
| in-model design step | 1181.5 ms | **1112.5** | -5.8% |
| the same step on XLA | 1618.8 ms | | **1.46x** |

Three things got it: the activation-keep tier switched ON by default (-8.3%
of the backward), three of that tier's six tensors stored as uint8 because
they are sigmoids (-2.46 GB and slightly FASTER), and the flash backward's
softmax moved out of shared into the accumulator's own layout (-11.3% of the
single biggest kernel).

The flash paddings were re-swept afterwards, because the softmax change moved
what `Ss` is used for and the old optimum was found before it: BM=64
LDH=LDF=40 is still best (2.462 ms against 2.556 / 2.864 / 2.984 for the
tighter layouts, and 2.854 for BM=128).  Unchanged, but now checked rather
than inherited.

What is left, measured rather than guessed: flashBwd is still 27.7% of the
backward and all five of its round trips are priced; combineDx (9.4%) and
scatterDx (7.1%) run at 22-30% of tensor peak, which is what fusion costs
against cuBLAS and the break-even rule says to accept; dGateK (4.6%) moves
84 MB a call against a 0.17 ms roof and measures 0.208, **82% of bandwidth**.
The one sizeable thing not done is the backward-scratch arena, ~450 MB.

### One f32 pair buffer, not six

`p1..p6` are the forward's residual chain -- `p_{i+1} = p_i + module_out` --
and **nothing else reads them**: every backward takes the bf16 NARROW
(`p1b..p5b`), which is still kept per step.  Each link is elementwise, so a
thread reads its own index before writing it and the whole chain can alias
onto one buffer, making every add in-place.  `p0` is the exception and the
only aliasing constraint there is: it is the block INPUT, owned by the
stack's save-set, so `p1 = p0 + opm()` must still write somewhere else.

**13.79 -> 13.62 GB**, 5 x N^2 x CZ x 4 = 168 MB at N=256 and N^2 like
everything else here.  No speed change either way (the traffic is identical:
four streams before, four after).

Gated as an aliasing change should be: the block's `out/pair` is identical to
nine digits and `out/msa` likewise, grad/pair 0.994949739 against 0.994949619
and grad/msa 0.881402 against 0.881390, stack-48 0.978722 against 0.978699.
`OPM_LAST` and `PAIR_ONLY` read garbage against a multimer oracle in BOTH
arms, which is the point -- they agree with each other (0.136468 vs 0.136776,
0.0136135 vs 0.0136135), so the aliasing does not change those branches
either.  Only `b.p6` is read from outside the block, and only after the
forward has finished.  `BLK_ONEPF=0` restores the six.

### Two buffers nothing could reach, and the last three inverted flags

`dQf` was allocated and never written.  `flashBwdK` writes dQ through the
last of three destinations -- `if (dQb) ... else if (dcomb) ... else dQ[...]`
-- and every call site here passes `dcomb` as `dQb`, so the `dQ` parameter is
dead; `scatterDxK` takes dK, dV and dnbb but not dQ, and the `TA_SPLIT` path
passes nullptr for it too.  **M*C*4 = 33.6 MB an instance, 67 MB a block.**

`out` in tri-mul is written only by `outGateK`, which runs only when the
CALLER does not fuse the output gate into its residual add -- and `block.cuh`
fuses by default.  Allocated on first use instead: **34 MB a block** for a
buffer the default path never touches.

**13.62 -> 13.52 GB**, gates unchanged (trimul 0.999961, triattn 0.999971,
block 0.994950, stack-48 0.978725), and `NO_FUSE_OUTGATE=1` still works and
still allocates what it needs (13.55 GB, 0.978706).

`NO_FUSE_OUTGATE`, `TM_ROWMAJ` and `TM_CENTERF` were the last three flags
written as `getenv(X) == nullptr`, the inverted form of the same trap
`envOn()` fixed everywhere else: `TM_ROWMAJ=0` turned the row-major path ON.
All three now go through `!envOn(X)`, so **every boolean flag in this
directory honours its value.**

**And `TM_ROWMAJ=1` does not run at all** -- `lt.cuh:215 cuda: an illegal
memory access`.  Not caused by any of this: a binary built from the commit
this session started at crashes identically.  It is a tracked A/B flag that
rotted -- the third way a gate stops running.  Recorded rather than fixed:
knowing the flag is dead is the useful part.

(An earlier draft of this paragraph said "the paddings table above quotes
numbers from it".  **It does not** -- that table is an `FB_*` sweep of the
flash kernel and `TM_ROWMAJ` selects tri-mul's row-major dcn staging.  The
two are unrelated, and the claim was made without checking.)

### The paired modules share their backward scratch

`backward()` runs d:pair_trans, d:triattn_end, d:triattn_start, d:trimul_in,
d:trimul_out in that order on one stream, so the SECOND instance of each pair
is finished with its `d*` buffers before the first instance starts.  And every
`d*` buffer in these modules is internal: the only output of `backward()` is
the caller's `dpair`, and nothing reads `da`/`db`/`dt`/`dcnT`/`dO`/`dt2pre`/
`dx` or `dcomb`/`dag`/`davH`/`dKf`/`dVf`/`dnbbF` after it returns.  So
`tmi` can hand its own back and use `tmo`'s, and `tae` `tas`'s.

**13.52 -> 13.20 GB, -320 MB** at N=256, no speed change.  The named buffers
account for 304 of that (118 for the tri-muls -- da/db/dt/dcnT/dO/dt2pre/dx
at 16.8 MB each -- and 186 for the triangle attentions, of which dcomb alone
is 67.6); the remaining ~16 is the conditional `dcn`/`dpre` and 256-byte
alignment.

The FORWARD intermediates are the ones that cannot do this and they are not
touched: the replay writes all nine modules' before the first backward reads
any, which is the same reason only block INPUTS are saved.  Nor is the ending
node's transposed pair, which only one instance has.  `dKf`/`dVf`/`dnbbF` are
memset at the top of each backward and accumulated by atomics within it, so
they share on the same argument.

It also has to hold for a PARTIAL block -- the extra-MSA stack runs with
`n_pair` below 5 -- and it does, because the guards are nested: `k > 1`
(tmi, the borrower) implies `k > 0` (tmo, the lender), and `k > 3` (tae)
implies `k > 2` (tas).  The borrower can never run without the lender.  The
pair track's order is also fixed regardless of `opm_first` and `pair_only`,
which only move where the outer product mean sits among the MSA modules.

Gated where the aliasing actually happens, which is the block and not the
module drivers (they instantiate one of each): block `out/pair` identical to
nine digits, grad/pair 0.994949667 against 0.994950116, grad/msa 0.881396
against 0.881414; stack-48 over three reps each, 0.97870/0.97870/0.97879
against 0.97861/0.97871/0.97883 -- overlapping, which is what two draws of
the same number look like here.  `BLK_SHARE_DSCRATCH=0` restores it.

### Gated at three sizes, not one

Everything above was measured and gated at N=256, and two of this session's
changes touch INDEXING -- the uint8 gates and the register softmax, whose
`if (i < N)` / `if (j + 1 < N)` guards only matter where a tile runs off the
end.  A single size cannot see that.  grad/pair for the four changed modules:

| | N=128 | N=256 | N=384 |
|---|---|---|---|
| trimul outgoing | 0.999961255 | 0.999961287 | 0.999961200 |
| trimul incoming | 0.999963204 | 0.999963243 | 0.999963321 |
| triattn starting | 0.999970865 | 0.999970716 | 0.999970603 |
| triattn ending | 0.999970231 | 0.999969926 | 0.999969605 |

Flat to six digits across a 9x range in N, which is what shape-independence
looks like.  The dumps for the other sizes were already in `out/`; nothing
had been run against them.

### Two re-sweeps the uint8 gates invalidated, and the half of that win that only shows under pressure

`TM_BMF`, `TM_BMG` and `TA_BMS` were tuned before the gates became uint8,
which changed how many bytes `projGate` writes and `combineDx` and
`scatterDx` read -- the same reason the flash paddings needed redoing.  The
optima did not move:

| | shipped | alternatives |
|---|---|---|
| TM_BMF (projGate, fwd) | **32** -> 0.838 ms | 64 -> 0.897, 96 -> 0.889 |
| TM_BMG (combineDx, bwd) | **64** -> 1.310 | 32 -> 1.335, 128 -> 1.468 |
| TA_BMS (scatterDx, bwd) | **64** -> 2.464 | 32 -> 2.490, 128 -> 2.595 |

**And the uint8 gates are a SPEED win under memory pressure, which is
invisible when everything fits.**  The greedy grants `depths =
budget / stride`, so a smaller depth-set is more depths kept and less replay.
The acts set goes 8.10 -> 5.69 GB, so at a fixed budget:

| budget | | depths granted | stack bwd |
|---|---|---|---|
| 3000 MB | bf16 | acts 7/48 | 675.46 ms |
| | uint8 | **acts 10/48** | **667.82** |
| 5000 MB | bf16 | acts 18/48, t 4/48 | 662.19 ms |
| | uint8 | **acts 26/48**, t 3/48 | **654.22** |

~8 ms either way.  Small, but it is the regime a real campaign runs in: the
recycle comparison above, at L=288 on a 24 GB card with JAX resident, got
**27 of 48 depths**.  At full budget both builds land at ~617 ms and this
effect does not exist.

### Three more axes priced on the two biggest kernels, all zero

After the softmax win the lesson was: look for the axis nobody PRICED, not
the one nobody swept.  Three candidates remained on `flashBwd` (27.7% of the
backward) and `combineDx` (9.4%), and all three are worth nothing.

**Barriers cost nothing.**  `flashBwd` runs five block-wide `__syncthreads`
a key tile, 40 a block, and "too many barriers" is the first thing one
guesses about a kernel shaped like this.  `PROBE_BAR` turns all of them into
`__syncwarp` -- wrong across warps, since dV and dK each read every warp's
rows, but the same instruction count and traffic -- and measures **2.512 ms
against 2.500** over three reps.  The block-wide synchronisation is already
hidden behind other latency.

**Nor is it short of ILP.**  Unrolling the key loop by two: 2.527 ms, +1%.

**Nor is the load->mma dependency the cost in `combineDx`.**  Its inner step
is `load fb; for r: {load fa[r]; mma}`, so each mma waits on the shared load
immediately before it, and `TM_LOADALL` issues all four activation fragments
first so the loads overlap and the four mma then issue back to back.
**+6.4%: 1.389 ms against 1.305.**

**And that is the third hoist of this session to lose on registers**, after
`FB_HOISTA` (+5.4%) and `FB_REGDQ` (+13%, 96 -> 222 registers).  The rule:
**in these kernels, explicitly holding more fragments in registers always
loses.**  They already sit where occupancy is the binding constraint, the
compiler is already scheduling the loads, and an explicit hoist only takes
away its freedom.  Stop proposing it.

So `flashBwd` has now had round trips, barriers, ILP, K/V bytes, atomics,
occupancy, tile sizes, paddings and the LDS path all priced; one of the nine
(the softmax round trip) paid.  At 12.1 TFLOP/s it is 9.7% of this card's
fp16 peak, and that number is set by `FD = 32` giving a 16x16x16 tile only
two k-steps -- which is the shape of AF2's attention, not of the code.

### The last unmeasured thing in the backward: the residual adds

The level-1 and level-2 profiles differ by ~164 ms, and the modules missing
from level 2 account for ~111 of it.  Subtracting gives ~53 ms for the pair
track's residual adds -- which is exactly the kind of cross-harness
subtraction that produced the bogus "242 ms of remat" earlier in this file,
so it is measured instead.  `addNarrowK`/`addGateNarrowK` now carry their own
`PROF_STEP` (inert unless `BLK_PROF=2`):

    blk residual   42.0 ms   8.9%   n=192

192 calls is four adds a block over 48 depths, so 0.219 ms a call.  Each
moves 100.8 MB -- read p (f32 33.6) and the module's out (bf16 16.8), write p
(33.6) and its bf16 narrow (16.8) -- which is **460 GB/s, 94% of this card's
489**.  At the roof, and irreducible without giving up AF2's float32 carry
between modules.

The estimate was 49-53 and the measurement is 42.  Close enough to have been
believed, and wrong enough to be worth the five lines.

**The backward is now fully accounted for**, every part of it measured
rather than inferred:

| | ms | state |
|---|---|---|
| flashBwd | 171.1 | nine axes priced, one paid |
| module forwards (the replay) | ~106 | bounded by VRAM, not code |
| combineDx | 40.6 | MMA-bound (PROBE_CDX, PROBE_LDS) |
| **residual adds** | **42.0** | **94% of bandwidth** |
| d:pair_trans | 35.2 | |
| scatterDx | 30.7 | MMA-bound, 22% of peak |
| lnBwd x2 | 26.7 | |
| centerLnBwd | 25.2 | second pass probed, register staging failed |
| bmm | 22.5 | cuBLAS at 37 TFLOP/s |
| gate (dGateK) | 20.0 | 82% of bandwidth |
| dOutGate, gemm dcn/dag, opm, msa | ~97 | |

### The transition, finally instrumented

`d:pair_trans` is 35.2 ms, 5.7% of the backward, and had no kernel-level
brackets at all.  With them:

| | ms | n | per call |
|---|---|---|---|
| tr.b maskDx | 19.8 | 48 | 0.413 ms |
| tr.b gemm dr | 9.6 | 96 | ~0.198 (the pair one) |
| tr.f gemm1+relu | 9.6 | 96 | replay only |
| tr.f gemm2 | **0.0** | **0** | `tail=false` doing its job |

`gemm dr` moves 84 MB in 0.198 ms = 424 GB/s, 87% of roof.  `maskDx` moves
151 MB in 0.413 = 366 GB/s and 20.8 TFLOP/s, so it is in the same MMA-bound
band as combineDx and scatterDx.

**And 44% of its traffic is `h`, read for one bit per element.**  `h` is the
relu output and the backward wants only its SIGN.  A bitmask is 8.4 MB where
`h` is 67.1, and the same mask would let the replay skip `gemm1+relu`
entirely, because the only other thing the transition's backward needs from
the forward is `mu`/`rstd`.  Priced end to end:

    replay skips gemm1+relu      -9.5 ms
    maskDx reads 8.4 not 67.1 MB -6    (it is only 75% bandwidth-bound)
    writing the mask each forward +7.4  (read h 67.1, write 8.4)
                                 ----
                                 -8.1 ms, for 0.2 GB

0.9% of a step, at 40 ms/GB -- a rate that would rank third among the keep
tiers.

**AND THE RECYCLE PASS MAKES IT WORSE, which the first pricing missed.**  The
binder preset runs `num_recycle = 1`, so there are **two** forwards a step and
only the second is differentiated.  The mask would be written by both unless
the write is gated on the save pass -- and ungated that is +14 ms against
15.3 saved, i.e. **0.1%, nothing**.  Gated it is the 0.8% above.  So the
change is not just a tier and a kernel edit: it also has to know which
forward it is in.

**Not done.**  Both halves are needed for the sign to come out
positive (the mask alone is +1.4 ms, since its write costs more than maskDx
saves), so it is a new keep tier plus a kernel change plus a forward pass,
~80 lines across four files, for 0.9%.  Recorded with the numbers so the next
person can decide rather than re-derive.

The free version does not exist: cuBLASLt will hand back the relu sign
bitmask from the forward GEMM (`RELU_AUX_BIAS`, and `DRELU` consumes it), but
on this shape that epilogue costs a 5.6x slower algorithm -- measured at the
top of `transition.cuh`, and the reason the mask is not already free.

### combineDx, fully accounted

Four probes now cover it, and they add up.  Per call, of 0.423 ms:

| | ms | share | probe |
|---|---|---|---|
| global plane reads | 0.085 | 20% | `PROBE_CDX` |
| LDS matrix_a loads | 0.036 | 8.5% | `PROBE_LDS` |
| build arithmetic | 0.007 | 1.7% | `PROBE_BUILD` |
| weight fetches from global | ~0 | 0% | `TM_LDCS` |
| **mma + the shared stores** | **~0.30** | **70%** | by difference |

Two of those were new and both are negatives worth having:

**The weights are not being evicted.**  combineDx streams 100.8 MB of
activation planes a call with no reuse, while its weight fragments come
straight from GLOBAL and are re-read every tile -- 164 KB a block over 1024
blocks is 168 MB if they fall out of L2, and L2 is 6 MB.  Marking the plane
loads `__ldcs` (evict-first) so they stop pushing the weights out measures
**1.306 ms against 1.307**.  They were staying cached already.

**The build's arithmetic is free.**  Five multiplies and a subtract an
element over 16384 elements a block looked, on a cycle count, comparable to
the MMA.  Dropping only the arithmetic and keeping every load, store and
conversion: **1.300 against 1.307, 0.5%.**

So 70% of the kernel is the mma instructions and the shared stores that feed
them, at 25-31 TFLOP/s -- 20-25% of this card's peak, which is what WMMA
gives on a contraction this shape.  The two ways to improve the 5:4
load:mma ratio -- more row tiles a warp (`TM_BMG=128`) or more column tiles
a warp (fewer warps) -- were both swept and both lose on occupancy.

Same picture in `scatterDx` (22% of peak), `maskDx` (17%) and `flashBwd`
(9.7%, where `FD = 32` halves the k-steps again).  **Those four are 262 ms,
43% of the backward, and all of it is the WMMA model's efficiency against
cuBLAS -- which the fusion break-even rule says to accept, because fusing is
what removes the HBM round-trips that would cost more.**

### The mma.sync rewrite, finally priced on the right axis

It was declined at 1.7%, from `FB_PAD_SMEM` -- an occupancy-and-bank-conflict
experiment.  That is not what `ldmatrix` is for.  `ldmatrix` loads four
fragments in one instruction where WMMA's generic path issues several LDS
ops, so its benefit is *fewer LDS instructions*, and nothing had measured
whether this kernel is short of LDS issue.

It should have been the first suspect: `flashBwd` issues **forty
`load_matrix_sync` a warp a key tile against twenty mma**, a 2:1 ratio and
worse than combineDx's 1.25:1.

`PROBE_FBLDS` pins the `u` index in dV's and dK's loads so the compiler
hoists them -- 12 of the 40 go, for two extra fragments, with the mma count
unchanged.  **2.484 ms against 2.471.**  Nothing.

So the kernel is not LDS-bound, `ldmatrix` has nothing to collect, and the
rewrite is dead on the axis that actually describes it.  With `PROBE_LDS`
saying 8.5% in combineDx, the same holds there.

**Everything in these kernels is now priced except the mma instructions
themselves.**  For combineDx that is 70% of it at 25-31 TFLOP/s; for
flashBwd, 12.1 TFLOP/s against a 125 peak, set by `FD = 32`.  There is no
measurement left to take that does not require a different algorithm.

## Implementing the rest in CUDA: what is actually left

With the native share corrected to **83.4%**, the porting budget is the
other **186.5 ms**, and every piece of it is now sized rather than guessed:

| piece | its cost | porting it saves | verdict |
|---|---|---|---|
| extra-MSA stack (4 blocks) | ~80 ms | **42.8 ms, 3.8%** | **already written -- now automatic** |
| embedder, relpos, recycling, residual | ~107 ms | ? | mostly Linear and one-hot; XLA is near roofline on those |
| template pair stack (2 blocks, C=64) | ~25 ms | ~8 ms, 0.7% | needs C=64 **and** FD=16 kernel variants |
| heads + structure module | **12 ms** | ~4 ms | measured; not worth it |

The template pair stack is the only substantial model piece left, and its
dimensions are the problem: `config.py:469-472` gives `num_channels: 64,
num_block: 2`, so it is 2 of 48 blocks at half the width -- about 2% of the
main stack's pair work.  `trimul.cuh` and `triattn.cuh` hardcode `C = 128`,
and the flash kernels are built around `FD = 32` (the whole "FD caps it"
analysis).  Serving C=64/FD=16 means templating the two most delicate
kernels here on width AND head dim, then re-tuning every tile constant for
the new shape, to collect 0.7%.  **That is the wrong trade and the numbers
say so.**

The heads and the structure module were measured directly, by stripping the
loss terms that reach them: **12 ms, 1.1%**.

### So the one piece worth shipping, and it is now automatic

`backend.enable(..., extra='auto')` is the new default.  The extra-MSA stack
is ported, worth 3.8%, and was off because it needs a second block of
scratch:

| L=224 | ms a step | VRAM |
|---|---|---|
| extra off | 1118.3 | 14.1 GB |
| **auto (on)** | **1071.9** | 15.8 GB |

and at L=375 `extra=True` dies with `stack.cuh:88 cuda: out of memory` -- a
hard abort inside a cudaMalloc, the worst thing it could do mid-campaign.
There is no formula that predicts where it stops fitting, because the
stack's footprint ADAPTS (the keep tiers shrink to whatever is left) while
the second block's scratch does not.  So the guard is a **measured bound and
not a model**: on at or below the length measured to work, off above it, and
additionally off on any card smaller than the one it was measured on, since
the bound is a memory fact and not a length fact.  `extra=True` forces it
for anyone who has checked their own card; `extra=False` and `--no-extra`
force it off.

**The gap is now measured**, one process a length (the FFI cannot free a
Stack, so two lengths in one process starve the second -- see the warning
`get()` prints), extra forced on, idle card:

| L | with extra | | L | with extra |
|---|---|---|---|---|
| 224 | 1071.9 ms | | 320 | 2394.5 ms |
| 256 | 1387.8 | | 352 | **OOM** |
| 288 | 1857.8 | | 375 | **OOM** |

So the bound is **320**, and `_extra_fits` carries it along with two
conditions the lengths above were measured under: a card of at least 22 GB,
and at least 18 GB of it free at `enable()` time.  Both matter because the
failure is a hard abort, and because what fits is not a function of N alone
-- the stack's footprint ADAPTS while the second block's scratch does not.
When auto declines it now says WHICH condition failed, since a user who
expected the 4% should not have to guess.

### A consequence of the stale-.so guard, learned the hard way

`backend.ready()` refuses when any `.cu`/`.cuh` is newer than
`libnative_evoformer.so`.  That is the right behaviour -- it is what stops a
measurement running against code that is not in the tree -- but it means
**editing a kernel source while a run is in flight breaks that run's later
steps.**  It happened here: a four-length sweep had its first length measured
and the other three refused, because an unrelated `ffi.cu` edit landed
between them.  The refusal is silent in a harness that greps only for its
own success pattern, which is why the sweep script now has a branch for it.

Two practices follow.  Edit `.md` and `.py` freely during a run -- the guard
only looks at `.cu`/`.cuh`.  And when a sweep must span a kernel edit,
rebuild the `.so` first: replacing the file does not disturb a process that
has already mapped it.

### Does turning it on change the design?  The control is the whole answer.

It is a default now, so it needs a gate.  The first attempt compared ten
design steps with the stack native against XLA and **failed, 4-15% on every
metric** -- and that was the wrong gate: ten gradient updates amplify a 1e-3
difference, so it measured chaos, not correctness.  (It also ran both arms in
one process, where the second is memory-starved; the FFI cannot free a Stack.)

One step, one process an arm, is better but still unreadable on its own,
because **nothing here is bit-identical to XLA** -- the native 48-block
stack's own gate is grad/pair cos 0.978.  So the question is not "does extra
change the numbers" but "does it change them more than the stack already
does".  Three arms:

| metric | XLA | native | native+extra | native/XLA | extra/native |
|---|---|---|---|---|---|
| con | 4.073440 | 4.175316 | 4.219736 | 2.5e-02 | 1.1e-02 |
| i_con | 4.130709 | 4.219882 | 4.183971 | 2.2e-02 | 8.5e-03 |
| i_pae | 0.651312 | 0.667427 | 0.665891 | 2.5e-02 | 2.3e-03 |
| loss | 6.881998 | 6.977097 | 7.019051 | 1.4e-02 | 6.0e-03 |
| pae | 0.701064 | 0.703655 | 0.706975 | 3.7e-03 | 4.7e-03 |
| plddt | 0.686238 | 0.695321 | 0.701386 | 1.3e-02 | 8.7e-03 |
| rg | 5.107069 | 4.629168 | 5.177874 | 9.4e-02 | 1.2e-01 |
| **median** | | | | **2.2e-02** | **8.5e-03** |

**Enabling the extra stack moves the step-1 metrics 2.6x LESS than the
native stack already moves them against XLA.**  `rg` is the outlier in both
columns, and for the same reason: it is the one metric read off the
structure module's atom positions, which amplify a pair-level difference.

So the default is defensible, and the reason it is defensible is the control.
A deviation with no baseline deviation beside it says nothing.

## With approximations allowed: an 8-bit keep tier, built and reverted

Told that minor approximations were fine, the obvious target is the keep
tier's three remaining bf16 tensors -- `a`, `bb` and `t`, 4.84 GB of the
13.20.  Priced with `TM_Q8`, which rounds them to 8 bits in place (and must
run BEFORE the centre-LN, so its statistics come from the quantised values;
a first version rounded after and the stack went NaN, which was the probe and
not the idea):

| quantised | module relL2 | **stack relL2** |
|---|---|---|
| nothing | 8.80e-03 | 2.062e-01 |
| **a + bb** | 3.23e-02 | **2.063e-01 -- free** |
| t | 3.30e-02 | 5.403e-01 -- 2.6x worse |

`t` is out: it is what the centre-LN's statistics are computed from.  `a` and
`bb` are numerically free, and `TM_RANGE` says a FIXED scale suffices -- |a|
over 96 calls has median 14.4 and max 37, a spread of 2.6x, so 48 clears it
with margin and no reduction pass is needed.

**So it was built**, and it works: VRAM **13.20 -> 11.66 GB, -1.54** with the
gradient unmoved (relL2 2.069e-01 against 2.062e-01).  `a`/`bb` stay bf16
module scratch because the FORWARD's batched GEMM reads them, and the tier
holds the 8-bit copy that `combineDx` reads.

**And it was reverted, for three measurements.**

1. **It costs 2%.**  fwd+bwd 905 -> 926 ms.  projGate pays for the extra
   8-bit store and -- the part the design missed -- the BACKWARD's batched
   GEMM also reads `a`/`bb` in bf16 (`dA = dT B`, `dB = dT^T A`), which the
   replay no longer produces, so they have to be widened back first.  That
   was the bug behind a grad/pair of 0.8608 until it was found.
2. **The memory does not buy the time back.**  At a fixed budget the saved GB
   should return depths and therefore replay: 3000 MB 671.45 -> 685.28,
   4500 663.79 -> 664.22, 6000 649.82 -> 646.66, unlimited 628.25 -> 637.75.
   A wash at best.
3. **It unlocks no length.**  L=375 with the extra-MSA stack still dies, and
   at `stack.cuh:88` -- the MANDATORY f32 SAVE-SET, which the tiers never
   reach.  Saving tier memory cannot help a failure in the save-set.

**Which is the useful part: the save-set is the target, not the tier.**  It is
`np * 4 * depth` = 3.62 GB at N=384 and the one allocation that caps the
length.  `SAVE_BF16` already rounds it as a probe, and this file rejected it
on oracle deviation (3x on the gradient).  Re-read against a design step
instead, its median step-1 metric deviation is **8.1e-04 -- 27x smaller than
the native stack's own 2.2e-02 against XLA.**  Halving it is the next move,
and it is the one that raises the length ceiling.

The probes (`TM_Q8`, `TM_Q8_SCALE`, `TM_RANGE`) are kept so none of this has
to be re-derived.

### The save-set was allocated twice, and one copy was never read

`own_save` follows `STACK_GRAPH_UNSAFE`, which is off by default.  So the
in-model path runs on XLA's save-set, handed in as an FFI output -- while
`alloc()` had ALREADY allocated ours, which nothing then reads.  That is
`np * 4 * depth`: **1.23 GB at L=224, 1.61 at N=256, 3.62 at N=384**,
allocated and untouched for the whole run.  And `useOwnSaveSet()` made it
worse: it allocated a SECOND set (`osp`/`osm`) rather than reusing the one
`alloc()` had just made, so the graph path held it twice.

Both fixed.  `setSaveBuffers` releases ours the moment it is handed a
different buffer, and `useOwnSaveSet` reuses `psave` when it exists (falling
back to its own allocation if it has already been freed).

| | peak VRAM | ms a step |
|---|---|---|
| before | 15.80 GB | 1071.6 |
| **after** | **14.18** | 1072.3 |

**-1.6 GB in-model for nothing** -- no approximation, no slowdown.  The
standalone harness is unchanged (13.20 GB), because it passes its own
`psave` and so nothing is released.

This is the kind of thing the memory accounting in this file was supposed to
catch and did not: the table of what the stack allocates counted `psave`
once, and never asked whether the in-model path reads it.

### And not allocating it at all raises the length ceiling

Freeing the unused save-set (above) returned the memory but still ALLOCATED
it first -- and that allocation is where long runs died.  `L=375` with the
extra-MSA stack aborted at `stack.cuh:88`, which is `cudaMalloc(&psave,
np * 4 * depth)`: **3.62 GB at N=384, for a buffer the in-model path never
reads.**  It failed asking for memory it was about to throw away.

`alloc()` no longer makes it.  `ensureSaveSet()` does, on first use, for the
standalone harness -- which is the only caller with nobody to hand it one.
The FFI always calls `setSaveBuffers` (XLA's) or `useOwnSaveSet()` before the
first forward, so it never reaches that path.

**L=375 + extra now runs**, where it was a hard abort -- and it is faster
than that length managed WITHOUT the extra stack:

| L=375 | ms a step |
|---|---|
| extra off (what it could do before) | 3808.7 |
| **extra on (previously an abort)** | **3590.3** |

**-5.7%.**  The unlock delivers the extra-MSA stack's speedup at exactly the
lengths that could not have it, which is where a design is slowest and the
saving is worth most.  Standalone is unchanged (13.20 GB, grad/pair
0.978698): it allocates on demand exactly as before, just later.

**The length ceiling was a bug, not a budget.**

**IT COSTS 0.8% AT SHORT LENGTHS, and that is reproducible, not noise.**
L=224 goes 1071.6 -> 1079.3 / 1080.5 / 1080.2 over three separate runs.  The
earlier free-when-handed-another-buffer change was neutral (1071.6 ->
1072.3); it is the DEFERRAL that costs, and the mechanism is not identified
-- `ensureSaveSet()` is an early return on every path that matters, so the
likeliest explanation is allocator layout, the tiers now landing differently
in a heap that no longer has a 1.23 GB hole punched in it first.

Worth it: -1.6 GB everywhere, and at L=375 the trade is the other way
entirely (a hard abort becomes 3590.3 ms, 5.7% faster than that length could
previously manage at all).  Recorded because "no slowdown" was what the
previous commit said, and it is only true of the first half.

## The pair save-set in bf16: smaller AND faster

`SAVE_BF16` has sat in this file as a rejected probe, turned down on ORACLE
deviation: f32 against the oracle is 8.3e-02 on the pair gradient and bf16 is
1.56e-01, 3x worse.  That is the wrong question for a design run.  Asked
against a design STEP -- which is what the thing is for -- the median step-1
metric moves **8.1e-04, 27x LESS than the native stack already moves them
against XLA (2.2e-02)**.

So the save-set is bf16 now.  `pin[]`, `psave`, `p1save` and the FFI's
`psave` buffer are all narrow; the block's `forward()` takes a
`__nv_bfloat16*` pair; and the three kernels that read it (`copyNarrowK`,
`addOpmNarrowK`, `addConstNarrowK`) are templated on the source type, since
they are also used mid-block where the source is f32.

**It is smaller and faster at the same time**, which the rejection did not
anticipate:

| standalone, N=256 d48 | before | after |
|---|---|---|
| VRAM | 13.20 GB | **12.39** |
| fwd | 283.65 ms | **280.36** |
| bwd | 617.87 ms | **614.11** |
| fwd+bwd | 901.51 | **894.47** |
| grad/pair relL2 | 2.064e-01 | 2.094e-01 |

Faster because narrowing the input DELETES work: `toBf16K(p0b, p0)` is gone
entirely -- `p0b` now aliases `p0`, which is already narrow -- and the
residual kernels that read `p0` read half the bytes.  The save itself writes
16.8 MB where it wrote 33.6.

`SAVE_BF16` survives as a probe for the MSA half, which is `S*N*Cm` = 0.02 GB
and has never been worth narrowing.

**In-model it buys SPEED but not peak memory**, and the distinction is worth
recording rather than rounding off:

| PD1 binder L=224 | step | peak VRAM |
|---|---|---|
| before tonight's save-set work | 1071.6 ms | 15.80 GB |
| + free the unused copy | 1072.3 | 14.18 |
| + defer the allocation | 1080.2 | 14.18 |
| **+ bf16 save-set** | **1063.9** | 14.18 |

The narrowing is plainly active -- that is where the 1.5% came from -- but
the high-water mark does not move, because in-model the save-set is XLA's
buffer and XLA's allocator does not give its arena back.  The peak is set by
something else.  So: **-0.81 GB standalone, and in-model a speedup with the
memory saving invisible to `nvidia-smi`.**  A longer length, where the
save-set is 3.62 GB rather than 1.23, is where it should show -- untested.

Gates after all of it: trimul 0.999961, triattn 0.999971, pair_transition
0.999208, opm 0.999985, msa_row 0.998855, block 0.994950, stack-48 0.978325
(band 0.9783-0.9788).

### At L=375 the bf16 save-set does show, and it shows as 2.2 GB

The caveat above -- "a longer length is where it should show, untested" -- is
now tested, against a rebuild of the previous commit in a worktree so the only
difference is the save-set's width.  PD1 binder, `--contig A:260`, the
extra-MSA stack on:

| L=375 + extra | ms a step (16) | ms a step (64) | **peak VRAM** | free at alloc | tiers granted |
|---|---|---|---|---|---|
| f32 save-set | 3589.4 | 3641.3 | **20817 MiB** | 2.50 GB | 0 / 0 / 0 |
| **bf16 save-set** | **3538.6** | **3587.6** | **18603 MiB** | 6.79 GB | flash 0, acts 6, t 2 |

**-2214 MiB of peak, -1.4% of the step.**  So the L=224 reading was not the
general case: there the save-set is 1.23 GB and the peak is set by something
else, and here it is 3.62 and the peak is set by this.

The saving is BIGGER than the 1.81 GB the narrowing removes, and the free
memory at `alloc()` moves by 4.29 GB rather than 1.81.  That is consistent
with XLA holding the save-set twice -- it is an output of the forward custom
call and an input to the backward's, and nothing donates it -- so halving it
is worth two copies, not one.

## The keep tiers are re-planned once the process is warm

`alloc()` runs inside the FIRST forward, and `cudaMemGetInfo` at that moment
has not seen JAX grow into the backward.  So it holds back
`np*4*depth/2 + totalb/8` -- **4.77 GB at N=384** -- against growth it cannot
measure, and spends what is left cheapest-tier-first.  At L=375 that produced
the last row of the table above: 2.02 GB of budget, the flash tier refused
(it may only be granted whole, and it is 3.85 GB), and 1.75 GB spent on
`acts` and `trimul t` at 9.3 and 8.5 ms/GB -- **the two WORST tiers, because
they were the two that fit.**

By the third step there is nothing left to guess.  XLA's BFC allocator has
reached its high-water mark and never returns a byte, so free memory IS what
is spare.  `regrant()` reads it again and **re-plans**, which is the part that
matters: it hands the partial grants back first, so the budget it spends is
1.75 GB returned plus what was never spoken for, and the whole-only tier can
be afforded out of memory its own cheaper rivals were holding.

It runs at the top of a SAVE forward and nowhere else.  The kept buffers are
written by the forward and read by the backward's replay, so moving a depth
between "kept" and "recomputed" in between would have the replay read a slot
the forward never wrote.

**WHAT MAY NOT BE HANDED BACK.**  A tier `alloc()` granted in FULL had the
module's own buffer freed under it, so `use()` has no fallback left for it;
those tiers are locked at what they have.  In practice they are `stats` and
`bias`, 0.05 GB between them.

At L=375 it takes the trade the blind pass could not see:

```
blind:   flash 0/48, acts 6/48, t 2/48        (6.79 GB free, 4.77 held back)
replan:  flash 48/48, acts 2/48, t 0/48       (5.93 GB free after handing 1.75 GB back, 1.48 held back)
```

| L=375 + extra | ms a step (16) | (64) | peak VRAM |
|---|---|---|---|
| blind grant | 3538.6 | 3587.6 | 18603 MiB |
| **re-planned** | **3422.4** | **3476.9** | 21035 MiB |

**-3.3%**, and it is honest about the cost: it SPENDS 2.4 GB of headroom to
get it, which is the whole idea -- that headroom was being held against an
estimate that two complete steps have since made unnecessary.  It leaves
1.99 GB free against a 1.48 GB margin.

**L=224 is untouched**: 1064.8 / 1067.9 ms and 14517 MiB peak, against 1063.9
and 14517 committed.  Every tier is already granted whole there, so the
re-plan finds nothing to do and says nothing.

Together with the save-set, L=375 goes **3590.3 -> 3422.4, -4.7%.**

### Gating it

The re-plan only fires under memory pressure, which on this card means
N=384 -- the same "exercised exactly once, in the hardest configuration"
problem the partial grants had.  `REGRANT_AFTER=0` makes it fire on the first
call, so `KEEP_BUDGET_MB` can starve the blind grant and the re-plan can be
gated on a card with room to spare:

| N=256 depth 48 | grad/pair |
|---|---|
| control (no re-plan) | 0.978515 |
| `KEEP_BUDGET_MB=120 REGRANT_AFTER=0` (0/0/0 -> 48/48/48) | 0.978218 |
| `KEEP_BUDGET_MB=120` alone, the other control | 0.978932 |
| `KEEP_BUDGET_MB=1000 REGRANT_BUDGET_MB=2200` (hands 0.90 GB back, flash 0->48, acts 7->11) | 0.978477 |

All four inside the 0.9783-0.9788 band, including the hand-back path, which
is the one that frees memory a later depth might still have wanted.

`NO_REGRANT=1` switches it off.  It never runs under `STACK_GRAPH_UNSAFE`: a
captured graph bakes these pointers in, and moving a tier's storage under one
is how that path's existing divergence would get a sibling.

## The two stacks share one block's pair scratch

A design step builds TWO native stacks -- the 4-block extra-MSA one and the
48-block main one -- and each allocated its own block scratch.  `BLK_MEM` now
reads the card BEFORE the stack allocates anything, which is the only way to
tell what the model holds from what this does, and at N=384 the answer was:

```
before any stack allocation, 8.88 GB of 23.68 is already in use
  trimul_out 662.7  trimul_in 595.6  triattn_start 820.0
  triattn_end 1048.6  pair_trans 453.0  block 304.1     -> 3.16 GB   (extra)
before any stack allocation, 13.46 GB of 23.68 is already in use
  ... the same again, plus opm/msa_row/msa_col/msa_trans -> 3.32 GB   (main)
```

**6.48 GB for one block's working memory, held twice.**  And it is scratch by
definition: everything that has to survive a call is in the save-set or a keep
tier, and the replay recomputes whatever a tier did not get.  The two stacks
never run at the same time -- extra forward, main forward, main backward,
extra backward are four separate custom calls -- so one set of buffers serves
both.

The second stack now adopts the first's five PAIR modules and pair-shaped
buffers, which depend on N alone and are 95% of the bytes.  Everything
MSA-shaped stays its own.

| | second stack's scratch |
|---|---|
| N=224 | 1.25 GB -> **0.10** |
| N=384 | 3.32 GB -> **0.22** |

**A FIRST VERSION SHARED EVERYTHING AND THE GATE READ 0.12.**  It inflated S
to a safe maximum and copied the whole `Block`, which cannot work: a module
captures S when it allocates, so a block told S=8 cannot then be run at S=1.
Only the S-independent half is shareable, and the `donor` argument to
`Block::alloc` is where that line is drawn.

### The hazard, made real, and what it caught

The risky moment is between a stack's forward and its backward -- the OTHER
stack runs in between and writes all over the buffers.  `SHARE_PROBE=1` does
exactly that in the standalone gate: it builds a second stack on the same
scratch, runs its forward between the forward and the backward, and asks
whether the gradient still lands.

It read **grad/pair 0.911 against a control of 0.978** -- and the cause was
not the sharing.  The second stack had been allowed to create its own stream,
and `setStream` had pointed the cuBLAS handles that BOTH stacks now own at a
stream the first was not running on.  **Sharing the buffers means sharing the
handles.**  Put the probe on one stream, as the FFI always is, and it reads
0.978255 against controls of 0.978183 and 0.978409.

That is the third time a private stream has cost a day in this directory, so
`Stack::setStream` now drains the previous stream whenever the stream changes
under a shared block.  In the FFI it never does -- every call gets XLA's
stream -- but "in practice there is one stream" is not a guarantee.

| N=256 depth 48 | grad/pair | grad/msa |
|---|---|---|
| sharing on | 0.978151 | 0.961874 |
| sharing on + `SHARE_PROBE` | 0.978175 | 0.961706 |
| sharing off | 0.978163 | 0.961829 |
| sharing off + `SHARE_PROBE` | 0.978121 | 0.961673 |
| the re-plan's hand-back path | 0.978114 | 0.961709 |

All five within 6e-05 of each other.  (The band this file has been quoting,
0.9783-0.9788, is from an older binary; everything tonight sits at
0.9781-0.9785, and the spread between two runs of the SAME configuration is
as large as the spread between configurations.)

### What it buys

| | ms a step (16) | (64) | peak VRAM |
|---|---|---|---|
| L=224, sharing off | 1069.9 | | 14517 MiB |
| **L=224, sharing on** | 1071.7 | | **13717 MiB** |
| L=224, off (repeat) | 1075.4 | | 14517 |
| **L=224, on (repeat)** | 1076.0 | | **13717** |
| L=375, before | 3422.4 | 3476.9 | 21035 |
| **L=375, sharing on** | **3399.4** | **3449.7** | 21123 |

**-800 MiB at L=224 for +0.1%, which is free** -- and the paired A/B is why
that is the number: a single run against this afternoon's would have read
+1.0%, which is drift between runs, not the change.  Nothing is granted
differently at L=224, so there is no speed to gain there; the memory is the
whole point.

At L=375 the 3.10 GB goes straight into the keep: the blind grant can now
afford the flash tier outright, and the re-plan takes `acts` from 2 depths to
**12**.  -0.7% on top of the re-plan's own -3.3%.

**L=375 over the night: 3590.3 -> 3399.4 ms, -5.3%.**

The one thing given up: a tier granted in full no longer frees the module
buffer it makes dead (190 MB at N=256, 427 at N=384), because the other stack
still uses that buffer as its own fallback.  Spending 427 MB to save 3100 is
the right way round, and `NE_SHARE_BLOCK=0` restores the old behaviour.

### Two memory knobs that do nothing, and one number that explains why

With the re-plan in place, `XLA_PYTHON_CLIENT_ALLOCATOR=platform` and
`XLA_FLAGS=--xla_gpu_autotune_level=0` were both priced at L=375, on the
hypothesis that XLA's arena holds more than its executable needs (its
`memory_analysis` says the whole design step is 3.30 GB of temp and 0.30 of
output, against 8.88 GB in use before the stack allocates anything).

**Neither moves a byte.**  All three runs report the same 8.81 GB free at the
re-plan and the same 12 of 48 `acts` depths, and both knobs are SLOWER
(3442.1 and 3428.4 against 3396.3).  Peak VRAM is identical too -- which is
the useful part: **since the re-plan, peak is set by US, not by XLA.**  It
spends down to its margin whatever is there, so "did peak move" has stopped
being a measurement of anything, and the number to read is how many depths
the tiers got.

### Where L=375 actually goes: 95% of it is these kernels

`NE_TIME` reports per-call AVERAGES (`dev / n`), and both stacks report into
the same pair of timers.  A step at L=375 is four forwards (the detached
recycle pass and the differentiated one, each running the extra-MSA stack and
then the main one) and two backwards:

    4 x 352.3 + 2 x 903.8 = 3216.8 ms of a 3399.3 ms step -- 94.6%

So at the top of the length range there is no XLA left to port: the embedder,
the template stack, the structure module, the heads, the losses and the
optimizer are ~180 ms between them.  Everything further has to come out of
the kernels, the replay, or the number of passes.

### Truncating the backward: the trace says no

`GRAD_TRACE` has sat in `backward_` with a note proposing that if the
gradient decays, backprop through only the last K blocks would be nearly free
and would cut the backward AND its replay by `(depth-K)/depth`.  It had never
been run.  At N=256 depth 48, rms of the cotangent arriving at each block:

| block | 47 | 40 | 32 | 24 | 16 | 8 | 2 | 0 |
|---|---|---|---|---|---|---|---|---|
| pair | 1.000 | 0.677 | 0.543 | 0.469 | 0.388 | 0.337 | 0.321 | 0.325 |
| msa | 0.998 | 1.112 | 1.392 | 1.651 | 1.814 | 2.587 | 9.594 | 8.554 |

**The pair gradient decays to a third and then PLATEAUS; the msa gradient
GROWS 8.5x.**  And the msa gradient at block 0 is the one that reaches the
sequence, because the sequence is upstream of block 0 -- so the cheapest
blocks to drop carry the largest part of the thing being optimised.

Truncation is not a minor approximation here.  Recorded so the idea is not
re-proposed: it is the obvious big win, it is written down in the code as a
suggestion, and the data refuses it.

## KEEP_MSA: the tier the enumeration missed

`BLK_PROF` separates the backward's `d:` rows (the real backward) from the
rest (the replay).  At N=256 depth 48 the replay is **149.9 ms of 622.1 --
24.1%** -- and the breakdown says where:

| | replay | the same module in a full forward | bought out |
|---|---|---|---|
| triattn_end | 39.7 ms | 62.6 | 37% |
| triattn_start | 32.8 | 55.6 | 41% |
| trimul_out | 19.2 | 51.9 | 63% |
| trimul_in | 19.2 | 51.6 | 63% |
| opm | 14.2 | 14.7 | **3%** |
| pair_trans | 12.8 | 31.4 | 59% |
| msa_row | 8.8 | 8.9 | **1%** |
| msa_col | 2.4 | 2.5 | **4%** |
| msa_trans | 0.8 | 1.7 | 53% |

**The three MSA modules are replayed in full, because nothing of theirs was
ever offered to the budget** -- `msarow.cuh` had no `keepSlots()` at all.  And
the reason is visible in the enumeration that built the tier list: every
candidate it priced was PAIR-shaped, where everything costs N^2.  The MSA
modules are `S*N`-shaped and this model runs S=2.

What their backwards actually read -- grepped, not assumed -- is 6.0 MB a
depth at N=256:

    msa_row    comb, qkvgH, biasH, avH, lseF, Dv, mu, rstd, pmu, prstd
    msa_col    comb, av, mu, rstd
    msa_trans  h, mu, rstd
    the links  m1b, m2b

Not `x`, not `pn`, not `nbb`, not `ag`, not `out`: those are forward-only.
Listing them anyway would have nearly doubled the tier, and `pn` alone is
N^2*CZ -- msa_row LayerNorms the PAIR to build its bias -- which would have
taken the tier from 34 ms/GB to about 8.

    KEEP_MSA   0.33 GB a depth-set at N=256   ranked 3rd, behind stats and bias

With it granted the replay skips the whole MSA track: `msa_row`, `msa_col`
and `msa_trans` go to **n=0** in the backward profile.

| N=256 depth 48, paired on one binary | bwd | fwd+bwd |
|---|---|---|
| `NO_KEEP_TIER=2` (off) | 618.93 ms | 910.78 |
| **on** | **606.27** | **902.15** |

**-2.0% of the backward.**  In-model at L=224, two runs each way:

| | ms a step | peak |
|---|---|---|
| off | 1074.6 / 1075.6 | 13717 MiB |
| **on** | **1059.2 / 1063.1** | 13985 |

**-1.3% for +268 MiB.**

At L=375 it is a wash and the two counts disagree on the sign (3399.4 ->
3424.5 at iters=16, 3476.9 -> 3447.9 at iters=64), because there the tier
competes: it is granted whole and `acts` drops from 12 depths to 10.  Taken
as the larger sample, -0.8%.

### What the same profile says is NOT worth taking

The two triangle attentions are half the replay (72.5 ms) and they are the
expensive half.  The thing the replay rebuilds is one CONCATENATED
projection -- `triattn.cuh:25` is `QKVG = 4 * C`, `TOT = QKVG + H` -- so it is
**67.6 MB a depth per module, 6.5 GB over 48 depths for the pair**, which is
~11 ms/GB: above `acts` but below `flash`, and it would displace better
tiers at exactly the lengths where the budget is tight.

(An earlier reading of this table in-session put it at ~30 ms/GB by sizing
q/k/v as three separate half-width tensors.  They are one 516-wide
concatenation.  The original enumeration was right to leave it off.)

The outer product mean is replayed in full too (14.2 of 14.7 ms) and cannot
be skipped cheaply: it feeds `p1`, which is f32 because the pair chain's next
link is an f32 add, and `BLK_ONEPF` aliases p1..p6 onto ONE buffer so the
value is gone by the time the backward wants it.  Keeping it is 33.6 + 16.8
MB a depth for 14.2 ms -- ~6 ms/GB, below everything on the board.  The pair
transition is worse still: its backward reads `h`, which is `M*4C`, 67 MB a
depth, for 12.8 ms -- **3 ms/GB**, the worst candidate measured.

**NOTE: `NO_KEEP_TIER=<n>` INDICES HAVE SHIFTED.**  KEEP_MSA is 2, so flash
is 3, acts 4 and trimul-t 5.

## Against XLA, measured back to back, four lengths

Every ratio this file has quoted came from an XLA number taken on a different
day to the native one it was divided by.  These are the same card, the same
session, the same PD1 binder, alternating:

| | XLA | native | ratio | XLA peak | native peak |
|---|---|---|---|---|---|
| L=224 | 1625.0 ms | 1065.2 | 1.53x | 4489 MiB | 13985 |
| L=375 | 5865.2 | 3447.9 | **1.70x** | 16683 | 21141 |
| L=435 | 8218.7 | 5085.4 | 1.62x | 16683 | 21161 |
| L=515 | **RESOURCE_EXHAUSTED** | **8329.8** | -- | -- | 21173 |

**THE RATIO IS NOT MONOTONIC IN LENGTH.**  1.53 -> 1.70 -> 1.62.  It peaks
around L=375 and falls back, because from L=375 onward the native peak is
pinned at ~21.1 GB -- the re-plan spends down to its margin and there is
nothing more to spend -- while the replay keeps growing as N^2, so the tiers
cover a thinner slice of it at every further length.

**XLA REMATERIALISES TOO, and this file has implied otherwise.**
`alphafold3/af2/runner.py:104` has `use_remat=True` as the default, so XLA's
48-block `layer_stack` is wrapped in `hk.remat` exactly as our backward
replays.  Its 4489 MiB peak at L=224 against our 13985 is the proof: it is
not storing activations we recompute.  Both replay.  What we have that it
does not is a BUDGET -- the keep tiers -- and that is what the extra memory
buys.  "We beat it while doing more arithmetic" was the wrong story.

**L=515 is the result that matters more than any of the ratios.**  XLA dies
asking for 10.60 GiB; the native path runs the step in 8.3 s.  Two lengths
that were not available at all now are (L=375 earlier tonight, L=515 here).

### Is 2.0x reachable

Not by kernel work on this card, and the budget says so rather than an
opinion.  At the best length the gap is 18%.  flashBwd is 34% of the backward
and runs at 10.3 TFLOP/s on its shape -- against XLA's Triton at 4.2 and
localfold's hand-tuned FORWARD at 6.4 -- with the ceiling set by FD=32, and
its mma.sync rewrite priced at 1.7%.  combineDx is 70% of its mma roof, the
residual adds 94% of bandwidth, lnBwd/dGateK/dOutGate 82-100%.  Five
register- and fragment-level rewrites were tried this session and all five
lost.

The one structural item left is the replay, and it is capped by the card, not
by ideas: 24.1% of the backward at N=256 with every tier granted, and the
tiers are already spending every byte that is free.  Tonight took the
per-depth footprint down (6.48 GB of duplicated block scratch -> 3.4) and the
budget immediately spent the difference on more depths.

**So 1.7x looks close to the A10 ceiling.**  The step that would change it is
a card where the whole keep menu fits at once and the replay goes to near
zero -- which is an 80 GB measurement nobody here has permission to take.

## The relu-mask tier: built, measured, defaulted OFF

This file has carried an end-to-end price for the pair transition's relu mask
for weeks:

    replay skips gemm1+relu      -9.5 ms
    maskDx reads 8.4 not 67.1 MB -6
    writing the mask each forward +7.4
                                 ----  -8.1, or 0.8% of a step

Built it.  `packReluMaskK` turns a warp's 32 signs into one word with a single
`__ballot_sync`; `hbit` is M*I/8 bytes; `maskDxK` is templated on whether the
sign comes from bits or from `h`; the pack is gated on BOTH the depth being
covered and `save_full`, so the detached recycle pass never pays it.  The
gradient is right: grad/pair 0.977947 with it against 0.978203 without, both
inside the band.

**And the price was wrong in the middle line.**

| N=256 depth 48 | off | on |
|---|---|---|
| `tr.f gemm1+relu` (replay) | 9.1 ms | **0.0** |
| `tr.b maskDx` | 19.8 | **18.7** |
| `tr.f packMask` | -- | **+10.9** |
| fwd | 280.91 | 293.09 |
| bwd | 605.95 | 599.45 |
| **fwd+bwd** | **886.86** | **892.54** |

**`maskDx` gave back 1.1 ms, not 6.**  Removing 44% of its traffic bought
5.6% of its time, because it is MMA-bound -- 20.8 TFLOP/s, the same band as
combineDx and scatterDx -- and this file's own note said so two sections
earlier while still pricing the saving as if it were bandwidth-bound.

And the pack costs 10.9 ms, not 7.4: it reads 67.1 MB and writes 8.4 at
332 GB/s, 68% of roof.  A lane reads one bf16, so a warp's access is 64 B of
a 128 B sector; vectorising it would reach perhaps 8 ms and the sum would
still be positive.

**The irreducible part is that the mask cannot be made without reading `h`
once, and reading `h` once costs about what the GEMM it saves costs.**  The
one way out is the GEMM epilogue, and `CUBLASLT_EPILOGUE_RELU_AUX_BIAS` is
measured at the top of `transition.cuh` as 5.6x slower on this GEMM (0.199 ->
1.098 ms).  So: `NE_KEEP_RELU=1` to switch it on, off by default, and the
estimate is now a measurement.

### And the argument that was never being passed

`stack.cuh`'s own forward called

    b.forward(..., /*tail*/true, /*haveFlash*/false, /*haveStats*/false,
              /*haveBias*/false, /*keepGates*/false);

which is NINE positional arguments, and the ninth is **`haveActs`**.
`keepGates` kept its default of `true`, so the tri-muls have been writing
their gate sigmoids in this pass all along and the 0.482 -> 0.423 ms a call
the comment claims was never collected.

It is not a bug in effect -- it is now REQUIRED, because `gl`/`gr` are
KEEP_ACTS and this forward is what fills that tier, so passing `false`
unconditionally would hand the replay a tier of garbage.  The right value is
`have(KEEP_ACTS, i)`: write the gates on the depths the tier covers, skip
them on the depths the replay will rebuild anyway.  Now passed explicitly.

**And it is worth about 0.3 ms.**  With `KEEP_BUDGET_MB=2500` the acts tier
gets 3 of 48 depths, so 45 depths take the skip, and the forward measures
280.62 ms against 280.91 with all 48 covered.  That is 0.006 ms a depth
against the 0.118 the module-level figure implies -- **20x smaller in the
stack than in isolation**, which is the same lesson as `t`'s 10.6-vs-13.7 ms:
a module-level number is not a stack-level number, in either direction.

## Two structural ideas, both bounded, both small

### CUDA graphs: 0.6%

`STACK_GRAPH_UNSAFE` has sat here with an open divergence (a 40-iteration
design goes wrong, a single shot does not) and a note saying "the standalone
win is real and measured" without saying how big.  Measured:

| N=256 depth 48, 4 reps | fwd | bwd | fwd+bwd |
|---|---|---|---|
| stream path | 279.96 | 603.12 | 883.08 |
| graph path | 277.54 | 600.36 | **877.91** |

**-0.6%.**  The prize for chasing a known wrong-gradient-over-a-loop bug is
half a percent, so the bug stays unchased and the flag stays off.  (The
pointer hazards one would look for first are already handled: `pout`, `mout`,
`sp`, `sm`, `spm`, `smm` and `sctp`/`sctm` are all Stack-owned staging
buffers, and the FFI copies to XLA's pointers outside the capture.)

### Overlapping the replay with the backward: 3.1%

Depth i-1's replay reads the save-set and the weights; depth i's backward
touches neither.  They are independent, and `flashBwd` -- 33% of the backward
-- holds only 3 blocks an SM.  So: run the replay on a second stream.  What
stops it is that both want the same block scratch, which would have to be
double-buffered (1.5 GB at N=256, 3.4 at N=384).

`PROBE_OVERLAP` prices it before paying: it issues the replay on a second
stream with NO ordering, so the math is wrong and the instruction count and
traffic are identical.

    serial          bwd 604.42 ms
    PROBE_OVERLAP   bwd 585.52      -3.1%

**-18.9 ms, and that is the UPPER bound** -- no events, no ordering, a
correct version pays for both.  So the GPU is already saturated through the
backward: `flashBwd`'s 3 blocks an SM is a SHARED MEMORY limit, not idle SMs,
and a concurrent kernel has nothing much to fill.  That is the same answer
the graph path gives from the other side.

Not built.  At N=384 it is 3.4 GB, and at the lengths where that memory
exists at all the keep tiers want it more -- `acts` at 9.3 ms/GB over 38
ungranted depths beats 12.6 ms/GB of overlap that also has to be correct.

## The f32 residual chain was 9.5% of a fwd+bwd, and it is bf16 now

The forward's kernel table had never been read this session -- all the work
went into the backward -- and it has the second-largest row in it:

| forward, N=256 d48 | ms | % |
|---|---|---|
| `ta.f flash` | 42.9 | 18.5% |
| **`blk residual`** | **42.1** | **18.1%** |
| `tm.f projGate` | 37.1 | 16.0% |
| `ta.f gemm qkvg` | 18.6 | 8.0% |

`blk residual` is 42.1 ms in the forward AND 42.1 in the backward's replay --
**84 ms of an 884 ms fwd+bwd, 9.5%** -- and it is at 94% of this card's
bandwidth roof, so the only lever is BYTES.  Each of the five links reads
`p` in f32 (33.6 MB at N=256), writes it back (33.6) and writes the bf16
narrow the modules actually read (16.8), against a delta of 16.8: **100.8 MB,
two thirds of which is the f32 chain nothing reads.**

`PROBE_BF16CHAIN` priced it first -- same instruction count, a bf16 chain's
traffic, wrong math -- at -29.9 ms.  Built properly it is more, because the
save-set narrowing at the top of each block becomes a plain copy:

| N=256 depth 48 | f32 chain | **bf16 chain** |
|---|---|---|
| fwd | 293.44 ms | **270.88** |
| bwd | 609.10 | **588.17** |
| **fwd+bwd** | **902.54** | **859.05** |

**-4.8%**, and in-model, paired over two rounds each way:

| | off | on |
|---|---|---|
| L=224 | 1054.6 / 1062.1 ms | **1003.8 / 1007.6** |
| L=375 | 3447.9 (iters=64) | **3281.4** |

**-5.0% and -4.9%.**

### Why the oracle would have rejected this

It costs the oracle plenty: `out/pair` relL2 1.669e-02 -> 3.357e-02 (2x), and
grad/pair 0.9782 -> 0.9755.  Read that way it is the worst approximation in
this directory.

Read against a DESIGN STEP -- 8 seeds, step-1 metrics, L=224 PD1 binder,
with the native-vs-XLA deviation as the control, because a stack that is
0.978 against its own oracle has no exact trajectory to deviate from:

| | median | p90 |
|---|---|---|
| native vs XLA (already there) | 1.87e-02 | 8.36e-02 |
| **bf16 chain vs native** | **2.25e-03** | 1.74e-02 |

**8.3x smaller than the deviation the port already carries**, and ~10x
smaller on every individual metric -- con 1.05e-03 against 1.10e-02, i_con
9.55e-04 against 7.66e-03, rg 3.91e-02 against 3.08e-01.

The chain BETWEEN blocks was already bf16 (the save-set is), so what this
changes is 5 roundings a block where there was 1 -- not a new kind of error.
`NE_BF16CHAIN=0` switches it off.

**This is the third time the same re-framing has paid**: the save-set
(rejected on a 3x oracle deviation, worth 0.81 GB and -1.5%), the extra-MSA
default (a 10-step trajectory gate failed at 4-15%, a controlled 1-step gate
passed at 2.6x under baseline), and now the residual chain.  *The oracle
answers "is this the same arithmetic".  It does not answer "is this the same
tool".*

### Where the gate band moved to

With the chain on, the stack gate reads grad/pair **0.9754-0.9755** across
six configurations (default, `NE_BF16CHAIN=0`, `SHARE_PROBE`,
`NO_KEEP_TIER=2`, the re-plan's hand-back path, `NE_SHARE_BLOCK=0`), and
`NE_BF16CHAIN=0` recovers 0.978099.  That is the band to compare against from
here.

## Borrowed from localfold's `opus-55-opt`: an occupancy hint that asked for the wrong number

That branch has a commit worth reading in full -- *"transitionUpK held to 128
registers - two blocks an SM, as it was written for: -11% to -13%,
byte-identical"*.  Its finding: a kernel whose SHARED MEMORY allowed two
blocks an SM had grown to 160 registers, so the REGISTERS silently held it to
one, and `__launch_bounds__(256, 2)` fixed it for free.

Asked of every big kernel here, with
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` on the real kernels:

| | regs | threads | smem | blocks | registers allow | shared allows |
|---|---|---|---|---|---|---|
| `projGateK` | 74 | 256 | 25216 | 3 | **3** | 4 |
| `combineDxK` | 93 | 256 | 43264 | 2 | 2 | 2 |
| `scatterDxK` | 80 | 256 | 25600 | 3 | 3 | 3 |
| `maskDxK` | 94 | 256 | 25600 | 2 | **2** | 3 |
| `flashBwdK` | 96 | 128 | 31232 | 3 | 5 | 3 |

Two register-bound kernels, and `projGateK` is the one that matters: 37.1 ms
of the forward and the worst roofline fraction in this directory (51%).

**AND THE NOTE THAT USED TO SIT ON IT ASKED FOR THE WRONG NUMBER.**  It said
"TM_MINB=2 on both kernels moved bwd 721.0 -> 718.1 and fwd 286.2 -> 288.8, a
wash" and concluded projGateK does not want the hint.  But minBlocks=2 is a
FLOOR, and the kernel already cleared it at 3 -- that experiment could only
ever have measured noise.  4 is the number its shared memory allows, and it
needs 64 registers.

`__launch_bounds__(32 * WF, 4)` gets there, with spills (28 B stores, 76 B
loads) -- which localfold's rule says to avoid -- so it had to be measured
rather than assumed.  Three alternating pairs, standalone N=256 depth 48:

| | fwd | mean |
|---|---|---|
| no hint | 279.89 / 279.74 / 280.70 | 280.1 |
| **minBlocks 4** | **261.20 / 262.80 / 264.45** | **262.8** |

**-6.2% on the forward**, backward unchanged, and the three pairs never
overlap.  In-model, alternated the same way:

| L=224 | run 1 | run 2 | mean |
|---|---|---|---|
| no hint | 1031.8 ms | 1039.9 | 1035.9 |
| **minBlocks 4** | **1007.2** | **1012.0** | **1009.6** |

**-2.5%.**  (A first reading said in-model showed nothing -- it compared
against a number from a different run session instead of pairing.  Same
mistake, same shape, as the shared-scratch "+1.0%" that was really +0.1%.
Pair the A/B or do not run it.)

`maskDxK` reaches 3 blocks at 80 registers with NO spills and **loses**
(858.25 -> 862.89).  So occupancy is not what that one is short of either --
the fourth kernel here to give that answer, after flashBwd, combineDx and
projGate's own earlier experiments.  `TR_MASK_MINB=3` re-measures it;
`TM_PROJ_MINB=1` drops the projGate hint.

## A CUPTI profile, borrowed from localfold, and what it says

`opus-55-opt` has `cuda/af3/src/profile.cuh`: CUPTI's activity API, `dlopen`'d
rather than linked, recording every kernel's start and end ON THE DEVICE --
plus `LOCALFOLD_PROF_GAPS=<n>`, which sorts the stretches between kernels and
prints the longest with the kernels either side.

**This box has no `ncu` and no `nsys`, which is why half the questions in this
file were answered indirectly.  It does have `libcupti.so.12`.**  Ported as
`cupti_prof.cuh` with the Beer-ware notice retained; `NE_CUPTI=1`, and
`NE_CUPTI_GAPS=<n>` for the idle stretches.

Why it is better than `BLK_PROF`, which is what every per-kernel number here
has come from: `BLK_PROF` brackets each step with events and SYNCHRONISES to
read them, so it serialises what the real run overlaps -- its own note says
the total it reports is larger than the real one -- and it only sees what
someone wrapped.  `BLK_PROF=1` accounts 603.4 ms of a 606 ms backward while
`BLK_PROF=2` accounts 511.6, and nobody could say what the 92 ms between them
was.  CUPTI needs no brackets, perturbs nothing and misses nothing.

### The device is 96% busy

    GPU kernel time 814.3 ms over a 851.6 ms span (96% busy)
    idle 37.3 ms in 8066 gaps (19.7 ms of them under 0.1 ms each); the longest:
       3.45 ms at +  3.6 ms  after toBf16K  before toBf16K
       0.15 ms x many        after tattn::dGateK  before fb::flashBwdK

One 3.45 ms gap at startup and then dust: 8066 gaps averaging 4 microseconds.
**That is the direct measurement of what the CUDA-graph bound (0.6%) and the
replay-overlap bound (3.1%) each said indirectly** -- there is no idle to
recover, and a concurrent kernel has nothing to fill.

### What the kernel table shows that BLK_PROF could not

| | ms | calls |
|---|---|---|
| `fb::flashBwdK` | 175.41 | 144 |
| `flashGridHalf` (localfold's flash forward) | 43.60 | 96 |
| `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_relu_f2f_tn` | 43.04 | 432 |
| `tmul::combineDxK` | 40.59 | 96 |
| `tmul::projGateK<true>` | 39.88 | 96 |
| `ampere_fp16_s1688gemm_fp16_128x128_ldg8_relu_f2f_tn` | 36.63 | 192 |
| `tattn::scatterDxK` | 30.59 | 96 |
| `addNarrowBK` | 25.33 | 240 |
| `addGateNarrowBK` | 23.20 | 192 |
| `tattn::dGateK` | 20.05 | 96 |
| `trans::maskDxK` | 19.79 | 48 |
| `addOpmNarrowK` | 19.28 | 96 |
| `tattn::transposePairVecK` | 16.67 | 240 |

Three things this makes visible for the first time:

* **The cuBLASLt GEMMs are 156.9 ms, 19% of all kernel time**, spread over
  eight algorithms.  `BLK_PROF` charged them to whichever module called them,
  so they had never been added up.
* **The residual chain is still 67.8 ms (8.3%)** even in bf16 --
  `addNarrowBK` + `addGateNarrowBK` + `addOpmNarrowK`.
* **`transposePairVecK` is 16.67 ms over 240 calls, 2%, moving data and
  computing nothing.**

The gate is unchanged under the profiler (grad/pair 0.975401), which it has
to be: CUPTI records, it does not intervene.

### The ratio after the day's work, both sides re-measured

The table further up (1.53x / 1.70x) is from this morning and the native side
has moved twice since.  Alternating, both arms fresh:

| | XLA | native | ratio |
|---|---|---|---|
| L=224 | 1618.4 / 1626.6 ms | **1004.5 / 1008.3** | **1.61x** |
| L=375 | 5898.0 | **3291.5** | **1.79x** |

XLA has not moved (1625.0 / 5865.2 this morning, inside the run-to-run
spread).  The native side went 1065.2 -> 1006.4 at L=224 and 3447.9 ->
3291.5 at L=375 across the bf16 residual chain and the projGate occupancy
hint.

**2.0x would need another -19% at L=224 or -10% at L=375.**  What is left
unattacked is the cuBLASLt GEMM block -- 156.9 ms, 19% of kernel time over
eight algorithms, which only became visible as a block when the CUPTI profile
landed.  Everything else is bounded: the device is 96% busy, flashBwd runs at
10.3 TFLOP/s on its shape, the replay's remaining candidates price at 3-11
ms/GB against tiers already competing for the same bytes, CUDA graphs are
0.6% and overlapping the replay is 3.1%.

A NOTE ON THE STALE-.so GUARD, which earned its keep again here: the native
arm of the first attempt at this measurement produced nothing, because
`cupti_prof.cuh` was newer than the library and the guard refused to run.
That is the second time today it has caught a measurement that would
otherwise have reported a stale binary's numbers as current.

## The cuBLASLt GEMMs are healthy, and the mma.sync rewrite is not dead after all

The CUPTI profile showed the GEMMs as a 156.9 ms block (19% of kernel time)
for the first time, so they were the obvious next target.  `LT_TUNE_LOG=1`
gives every shape the tuner planned, and the rate falls out:

| M | Nn | K | ms | TFLOP/s |
|---|---|---|---|---|
| 65536 | 512 | 128 | 0.185 | **46.4** |
| 65536 | 128 | 512 | 0.201 | **42.7** |
| 65536 | 128 | 128 | 0.079 | 27.2 |
| 65536 | 8 | 128 | 0.039 | 3.4 |
| 65536 | 4 | 128 | 0.038 | 1.8 |
| 65536 | 128 | 8 | 0.042 | 3.2 |
| 512 | 1024 | 256 | 0.011 | 24.4 |

**The big ones run at 42-47 TFLOP/s** -- 37% of this card's peak, and far
above anything hand-written in this directory (our best is combineDx at ~20).
The skinny ones look terrible on FLOP/s and are not: `Nn=4` moves 17.3 MB in
0.038 ms, **455 GB/s, 93% of roof**; `Nn=128 K=8` is 424 GB/s, 87%.  They are
the triangle attention's pair-to-bias projections, which are bandwidth shapes
with a GEMM's name.

So there is no bad algorithm pick to find, and the tuner (which TIMES its
candidates rather than trusting the heuristic order) is doing its job.

### Which leaves flashBwd, and a standing NO that was priced on the wrong axis

`flashBwd` is 175.4 ms, 21.5% of all kernel time, at **10.3 TFLOP/s against
those GEMMs' 46**.  This file says in three places that the kernel IS its five
`store_matrix_sync -> shared -> reload` round trips, and that `mma.sync` with
register-resident fragments is how they would go.  It then prices that
rewrite at **1.7% of the stack's backward** -- but that number came from a
swizzle-and-occupancy experiment (the 4th block is worth 5.3%, the padding it
needs costs 19.6%), which is a different axis.  **The round trips themselves
had never been priced as a set.**

`PROBE_NOSTORE` drops the four remaining `store_matrix_sync` calls -- the
loads, the barriers and every mma stay exactly where they are, and the
arithmetic is wrong because the consumers read stale shared memory:

| | flashBwdK | total kernel time |
|---|---|---|
| as shipped | 174.19 ms | 811.8 |
| **four stores dropped** | **149.63** | 785.9 |

**-24.56 ms, -14.1% of the kernel, from the stores ALONE.**  And that is a
LOWER bound on what the rewrite retires, because `mma.sync` chaining also
removes the matching loads -- which this file counts at 40 against 20 mma a
warp a key tile.

So the honest standing is: **the prize is at least 14% of flashBwd and
plausibly 25-30%, i.e. 3-6% of the whole stack, not 1.7%.**  The reasons not
to have written it -- days on the most delicate kernel here, against a
history of two wrong-gradient incidents in it -- are unchanged.  What has
changed is that the number on the other side of that trade is now measured,
and it is three times what the file has been quoting.

(One round trip was already retired on exactly this reasoning: `FB_REGSM`,
the softmax, -9.5% and the default since.  The other four lost when tried
with WMMA because their outputs go to global atomics or to registers the
accumulator cannot reach while staying alive -- which is the constraint
`mma.sync` does not have.)

### Correction: "plausibly 25-30%" was an extrapolation, and it does not measure

The commit above priced the round-trip set from `PROBE_NOSTORE` (-14.1%, and
clean: 96 registers and 0 spills in both arms, identical to the baseline) and
then said the full prize was "plausibly 25-30%" because `mma.sync` chaining
also retires the `a`-operand loads from `Ps`.  **That second number was an
extrapolation from a load count, not a measurement, and two attempts to
measure it both failed.**

`PROBE_NOPS` skips the three `load_matrix_sync(a, Ps + ...)`:

| attempt | registers | spill | flashBwdK |
|---|---|---|---|
| baseline | 96 | 0 | 174.19 ms |
| `PROBE_NOSTORE` | 96 | 0 | **149.63** |
| v1: skip the load, leave the fragment undefined | 128 | 4 B | 166.46 |
| v1, with `PROBE_NOSTORE` too | **202** | 0 | **182.97 -- slower than baseline** |
| v2: `fill_fragment` instead of the load | 96 | 4 B | **232.40 -- 33% slower** |
| v2, with `PROBE_NOSTORE` too | 158 | 0 | 141.13 |

v1 is explained: an undefined fragment stays live across the loop, 96 -> 202
registers, 1 block an SM instead of 3 -- which is why removing instructions
measured SLOWER.  v2 fixes the register count and is still incoherent: 96
registers, one instruction fewer, and a third slower.

**So the load side is not measurable by removal here.**  Taking the load out
changes what the scheduler can do with everything around it, and that
dominates whatever the load itself costs.  The honest standing is:

* the stores are **-14.1% of flashBwd, measured clean**;
* the loads are **unpriced**, and the "25-30%" in the commit above should be
  read as a guess that failed its own test.

The rewrite's case rests on 14.1% plus an unknown, against days in the
kernel with two wrong-gradient incidents in its history.  That is a weaker
case than the previous section claimed, and the probe failures are kept here
so the next attempt starts from them rather than repeating them.

**Both probes default to 0 and the shipped kernel is unchanged** (grad/pair
0.975432, fwd+bwd 843.86 ms).

## The fused centre norm: opt-in, and it should stay that way

The tri-mul's centre LayerNorm runs as three passes (`centerStatsK`,
`centerWriteK`, and a transposing write) and a FUSED single-pass kernel
exists beside it.  The comment on the branch named a flag `TM_CENTER3` that
does not exist and read as though the fused kernel were the default.  It is
not: the flag is `TM_CENTERF` and it opts IN.

Measured, two alternating pairs at N=256 depth 48:

| | fwd+bwd |
|---|---|
| three-pass (default) | 842.28 / 849.15 ms |
| `TM_CENTERF=1` | 903.46 / 910.68 |

**+7%**, and the kernel table says why: `centerFusedK` is **85.34 ms over 192
calls** against the three-pass's 23.06 (stats 7.24 + write 15.82).  That is
0.44 ms a call for a 33.6 MB pass -- **76 GB/s, 16% of roof** -- where the
three-pass kernels run at 84-87%.  The fused kernel loads a tile, reduces
over it, and the SM has nothing to do while it does.

**Fewer passes over the data is not the same as less time**, and this is the
clearest example of it in the directory: the fused form moves 33.6 MB where
the split form moves 50.4, and takes 3.7x as long.

It also makes `KEEP_STATS` -- the best tier on the board at 136 ms/GB --
worth nothing, because it recomputes the statistics from a tile it loads
anyway.

localfold's `opus-55-opt` reached the same verdict from the other side, in
`cbf4c08b`: *"note the folded centre norm measured in triangleOutK (-1% in
AF3, +3.7% in ESMFold2; not taken)"*.

## The pair transpose as two arguments, priced

`transposePairVecK` is 16.67 ms over 240 calls -- 2% of all kernel time
moving bytes and computing nothing -- and it is already at 100% of bandwidth
roof, so the only lever is NOT DOING IT.  It exists because triangle
attention's ending node is its starting node run on a transposed pair.

**It should be expressible as two arguments.**  The LayerNorm and the qkvg
projection are both ROW-WISE, so

    (LN(z^T) W)[row (i,j)]  ==  (LN(z) W)[row (j,i)]

-- a pure row permutation -- and `flashGridHalf` indexes its input as
`base + rl * rowStride + i * posStride`.  Swapping those two strides IS that
permutation.  The kernel has had `STRIDED` and the four stride parameters all
along and every caller in this directory passed zeros.

The bias cannot follow (it is `bias + (h*n + i) * biasStride + j`, one stride,
so a transposed read is not expressible) -- but the bias is `[H, N, N]` half,
**0.5 MB against the pair's 16.8**, and it is produced by `nbbToChan`, a
permutation kernel that could write it transposed for nothing.

So the one thing that can kill the idea is whether the flash kernel reading a
`[N, N, W]` block down its columns costs more than the transpose it replaces.
`TA_SWAPPROBE` measures exactly that: the strided access pattern on the data
we already transposed, so the arithmetic is wrong and the instruction count
identical.

| `ta.f flash`, 96 calls | ms |
|---|---|
| contiguous | 43.72 |
| **transposed access** | **52.06** |

**+8.34 ms**, and the trade works out:

* the probe strides BOTH the read and the write, but only the ENDING node
  would be -- 48 of the 96 calls -- so the real cost is about **+4.2 ms**;
* the saving is all 192 forward-path transposes (input and output, forward
  and replay).  The replay pays no flash cost at all, because the flash tier
  is granted there and `ta.f flash` is already n=0;
* **net about -9 ms, 1.1% of kernel time**, plus up to 3.3 ms more if the
  backward's 48 transposes can follow.

**Priced and not taken.**  1.1% is real, and it is a delicate change --
the bias orientation, the mask orientation, the output layout and the
backward's LayerNorm all have to move together, in the module with the most
intricate indexing here, where a mistake is a wrong gradient rather than a
crash.  The plumbing is left in place (`flashGridHalfRun` and
`flashGridHalfAt` now take the four strides, defaulting to the old
behaviour), so the next attempt starts from a probe and a wired path rather
than from this paragraph.

## Four integer divisions an element: the OPM's residual add

`addOpmNarrowK` was 19.28 ms over 96 calls in the CUPTI table -- 50.4 MB a
call at **251 GB/s, 51% of roof**, where its neighbours in the same residual
chain run at 87-94%.  It is the only kernel here that was doing arithmetic
the others are not, and the profile is what made it visible: `BLK_PROF`
charged it to "blk residual" along with two kernels that are at roof.

What it was doing per element, over 8.4M elements a call:

    size_t ij = i / CZ_;                       // runtime divide
    int r = ij / N, c = ij % N;                // two more
    for (int t = 0; t < S; ++t)                // and the mask loads
      nz += mask[t * N + r] * mask[t * N + c];
    ... bias[i % CZ_]                          // a fourth

`nz` depends on the mask and on `(r, c)` alone.  So it was recomputed for all
**128 channels of every pair element**, identically at all 48 depths, and
reaching `(r, c)` from a flat index cost three runtime integer divisions with
a fourth for the bias -- none of which the compiler can strength-reduce,
because `CZ_` and `N` are arguments.

`opmNzK` builds `nz` once over N^2 elements (65536 threads against the 8.4M
that each redid it) and `CZ` is a template parameter now, so the channel
index is a shift and a mask:

| | ms | rate |
|---|---|---|
| `addOpmNarrowK` before | 19.28 | 251 GB/s, 51% |
| **after** | **12.32** | ~400 GB/s |

**-36% of the kernel.**  `opmNzK` does not reach the top thirty.  Paired
fwd+bwd over three rounds: 846.97 / 849.64 / 852.88 -> **844.27 / 842.98 /
846.14**, -5.3 ms.

**The forward is bit-identical** (`out/pair` 0.999469725 to the digit) and
the gradient is in band (0.975436), which it has to be: same `nz`, same two
bf16 roundings, same order.  Nothing was approximated -- work was hoisted out
of a loop it never belonged in.

The same lens was then pointed at every other kernel here: the remaining
per-element divisions are by `C = 128`, a compile-time constant the compiler
turns into a shift, or they are in kernels already at bandwidth roof where
the ALU is not the limit.  This was the only one.

## A roofline sweep of every kernel, and two estimates that were wrong

With the CUPTI table it is finally possible to put a rate against every
kernel rather than against the few someone bracketed.  Doing that at N=256
depth 48 (M = 65536, C = I = 128, roof 489 GB/s):

| | ms | calls | traffic a call | rate | of roof |
|---|---|---|---|---|---|
| `tmul::lnBwdK` | 13.03 | 96 | 67.6 MB | 498 GB/s | **at roof** |
| `tmul::lnFwdK` | 6.84 | 96 | 34.1 | 478 | 98% |
| `addNarrowBK` | 25.32 | 240 | 50.4 | 476 | 97% |
| `addGateNarrowBK` | 23.23 | 192 | 58.8 | 486 | 99% |
| `centerLnBwdK` | 17.34 | 96 | 84.0 (**two passes**) | 465 | 95% |
| `centerStatsK` | 7.23 | 96 | 34.1 (**two passes**) | 453 | 93% |
| `transposePairVecK` | 16.66 | 240 | 33.6 | 487 | 100% |
| `transposeRowToChanK` | 7.55 | 96 | 33.6 | 427 | 87% |
| `halfToBf16K` | 7.59 | 96 | 33.6 | 425 | 87% |
| `dGateK` | 20.02 | 96 | 84.0 | 402 | 82% |
| `gateFromAvK` | 11.99 | 96 | 50.3 | 403 | 82% |
| `addOpmNarrowK` | 12.28 | 96 | 50.4 | ~400 | 82% (was 51%) |

and the compute-bound ones, against the cuBLASLt GEMMs' 42-47 TFLOP/s:

| | TFLOP/s |
|---|---|
| the big GEMMs | **42-47** |
| `tmul::projGateK` | 25.8 |
| `trans::maskDxK` | 20.8 |
| `tmul::combineDxK` | ~20 |
| `flashGridHalf` (forward) | 18.9 |
| `fb::flashBwdK` | **10.3** |

**TWO CANDIDATES LOOKED BELOW ROOF AND BOTH WERE MY OWN ARITHMETIC.**
`centerStatsK` read as 47% and `centerLnBwdK` as 58% until the traffic was
counted properly -- both make TWO passes over their input (the variance is
`mean((x-mu)^2)`, which this file is emphatic about for conditioning
reasons), and leaving the second pass out halves the traffic and halves the
apparent rate.  Counted right they are at 93% and 95%.

Recorded because the method is the point: a roofline fraction is only as good
as the byte count behind it, and guessing the byte count from a kernel's name
produced two false positives in a row.  The one real find of the sweep --
`addOpmNarrowK` at 51% -- was real because the kernel was doing *arithmetic*
the others were not, which the rate exposed and the source confirmed.

**So the bandwidth kernels are between 82% and 100% of roof, and there is
nothing left to collect there.**  The compute kernels are ranked by how much
fusion they carry: `projGateK` at 25.8 TFLOP/s is a GEMM plus five gated
outputs, and splitting it into a cuBLAS GEMM at 46 plus a gating pass prices
at 0.52 ms against its 0.415 -- the fusion is already the right call.

### Where the day ended

| | start of day | now |
|---|---|---|
| L=224 step | 1065.2 ms | **994.2 / 1002.1** |
| L=375 step | 3447.9 | **3259.4** |
| standalone fwd+bwd, N=256 d48 | 618.93 bwd | 580.25 |

against XLA measured fresh on the same card (1622.5 and 5898.0):

**1.63x at L=224, 1.81x at L=375.**

## Retraction: PROBE_NOSTORE's 14.1% is not a prize

Earlier today this file said the `mma.sync` rewrite was "worth at least 14% of
flashBwd, three times what this file has been quoting", on the strength of
`PROBE_NOSTORE` -- four `store_matrix_sync` calls dropped, 174.19 -> 149.63 ms,
with 96 registers and 0 spills in both arms.  The measurement is clean.  **The
interpretation was wrong, and the reason is worth more than the number was.**

A probe that REMOVES work only prices a prize if what is left corresponds to a
configuration you could ship.  `PROBE_NOSTORE` removed the stores and KEPT the
coalesced reads those stores exist to enable -- the consumers went on reading
shared memory, just stale shared memory.  No real kernel can do that.

What the four stores actually feed, and what each alternative measured:

| store | what it stages | the alternative | measured |
|---|---|---|---|
| dV | coalesced global atomics | accumulator-layout atomics | `FB_REGDS` family: much worse |
| dQ | registers in the epilogue | `FB_REGDQ` | **+13%**, 96 -> 222 registers |
| dK | coalesced global atomics | as dV | as dV |
| dP | `dS` for the next matmul | `FB_REGDS` (scattered dnbb atomics) | **+34.7%** |
| dP | " | `FB_REGDS2` (dS to Ps, second coalesced pass) | **a wash** |

So the one store `mma.sync` chaining could retire -- dP, feeding `dS` into
`dQ = dS K` -- has had its chaining measured twice already, under the names
`FB_REGDS` and `FB_REGDS2`, and it loses or washes.  **mma.sync changes none
of that**: the two transposed operands (`dV = P^T dO`, `dK = dS^T Q`) go
through shared under mma.sync as well, which is exactly what
`lf_flash.cuh`'s forward does with `ldsm4t` -- `ldmatrix.trans` out of shared.

**The rewrite is dead, and now for the right reason.**  Not "1.7% on an
occupancy basis" (which was the wrong axis) and not "14% of round trips"
(which was an unachievable state), but: every one of the five round trips has
had its register-resident alternative measured in this kernel, and the only
one that won is already the default (`FB_REGSM`, -9.5%).

**THE LESSON, which is the general one:** a removal probe needs TWO checks,
not one.  The first is that it did not perturb what it was not meant to --
registers, spills, occupancy -- and `PROBE_NOSTORE` passed that cleanly.  The
second is that the thing it leaves behind is reachable.  It was not, and
nothing in the register dump could have told me so; only reading what the
consumers do with the data could.

## The first full in-model kernel profile

`NE_CUPTI` works in the FFI now, not just the standalone driver, and because
CUPTI records every kernel on the device it profiles XLA's share of a design
step as well as ours -- the ~18% this file has only ever sized by subtraction.
The window is counted in SAVE forwards (`NE_CUPTI_WARM`, `NE_CUPTI_SPAN`), and
both stacks increment the same counter, so a span of 4 is one step.

PD1 binder, L=224, one step:

    GPU kernel time 953.0 ms over a 1003.6 ms span (95% busy)

| | ms | % | calls |
|---|---|---|---|
| `fb::flashBwdK` | 143.53 | 15.1 | 152 |
| `flashGridHalf` | 71.86 | 7.5 | 208 |
| `tmul::projGateK` | 66.96 | 7.0 | 208 |
| `ampere_bf16...relu_f2f_stages_32x1_tn` | 54.30 | 5.7 | 696 |
| `ampere_fp16...relu_f2f_tn` | 47.33 | 5.0 | 312 |
| `addNarrowBK` | 34.01 | 3.6 | 416 |
| `tmul::combineDxK` | 33.87 | 3.6 | 104 |
| ... | | | |
| XLA's own (cutlass, cudnn sdpa, `*_fusion_*`) | ~150-170 | **~16%** | |

**The subtraction was right.**  XLA's share measured directly is ~16%, against
the 16.6% this file arrived at by timing our half and subtracting.

### And there is no idle to recover in-model either

    idle 51.1 ms in 14955 gaps (29.0 ms of them under 0.1 ms each); the longest:
       3.30 ms at +   3.4 ms  after toBf16K  before opmNzK
       2.98 ms at + 756.4 ms  after input_concatenate_fusion  before loop_add_fusion_81
       1.68 ms at + 751.2 ms  after memcpy32_post  before wrapped_multiply
       0.54 ms at + 752.7 ms  after loop_multiply_fusion  before wrapped_compare

Of the 51.1 ms: **3.3 ms is the window opening** (an artifact of where the
profile starts), **29.0 ms is dust** -- 14955 gaps averaging 1.9 us, which is
launch overhead and which CUDA graphs were already measured at 0.6% against --
and **~6 ms is one cluster at t = 751-757 ms where every gap sits between XLA
kernels**: the optimiser and loss end of the step, running many tiny fusions
the host cannot feed.

So the standalone answer holds in the model: **95-96% busy, and the 5% is
launch dust plus XLA's own host-bound tail.**  Nothing on our side of the
boundary is waiting.

In-model is also where `projGateK`'s weight shows: 7.0% of a whole step, the
third-largest kernel, because it runs in the detached recycle pass, the
differentiated pass AND the replay -- 208 calls against `combineDxK`'s 104.
That is the kernel localfold's occupancy check was worth -2.5% on.

### CORRECTION, and the answer to "what is still XLA"

The section above put XLA's share at "~150-170 ms, ~16%".  That was read off a
table truncated at 40 rows with the tail GUESSED.  `NE_CUPTI_TOP=400` prints
all of it -- 399 distinct kernels in one step -- and the real split is:

| | ms | % | launches |
|---|---|---|---|
| ours, hand-written CUDA | 667.0 | **69.7%** | 7794 |
| ours, cuBLASLt (we dispatch it) | 210.5 | **22.0%** | 2536 |
| XLA, unambiguous | 58.6 | **6.1%** | 3148 |
| cutlass (cuBLASLt also dispatches these) | 20.8 | 2.2% | 1394 |

**~92% of a design step is ours.  XLA is 6.1%, or 8.3% if every cutlass
kernel is charged to it.**  Not 16%.

What XLA still runs, by name:

* **the template pair stack's triangle attention, on cuDNN flash** --
  `sdpa_flash_fprop` 5.07 ms (12 calls), `flash_bprop` 4.61 (4),
  `fmha_dbias` 1.72, `matmul_dq` 1.12: **12.5 ms, 1.3%.**  This is the only
  MODEL piece left on XLA, and it is the one already priced as not worth
  porting -- C=64 and FD=16 would need both of the most delicate kernels here
  re-templated on width and head dim for ~0.7%.
* the structure module and the heads: `gemm_fusion_dot_*` ~3.8 ms and most of
  the cutlass rows.
* the embedder, relpos and recycling: fusion kernels.
* **the loss, the optimiser and the sequence transforms: 285 distinct kernels
  each under 0.5 ms, 31.8 ms between them.**

That last line is the surprise.  **Over half of XLA's remaining time is 285
kernels under half a millisecond each, and none of it is the model** -- it is
Adam, the loss terms, the softmax/gumbel sequence transforms and the
bookkeeping.  No kernel there is worth writing; what would move it is XLA
emitting fewer ops, not faster ones.

So: the Evoformer is ported, the extra-MSA stack is ported, and what remains
on XLA is one small attention (1.3%), the structure module and heads (~2%),
and a long tail of glue (~3.3%) that a CUDA kernel cannot help.

## Two speedups tried and reverted, and what each taught

### The detached pass's backward-only state: +1.7 ms

The recycle forward (`save_full` false) has no backward, so the state that
exists only for one is dead there: `flashGridHalfAt` is called with
`avOut`/`lseOut` unconditionally, and those are M*C half plus N*H*N f32 --
18 MB a call, **1.7 GB over a 48-deep pass**.  Passing null in that pass, and
`keepGates` false with it, gates clean (grad/pair 0.975488).

**And it is slower.**  Paired in-model: 996.6 / 1003.2 against 998.4 /
1004.7, **+1.7 ms both times.**  Removing 1.7 GB of writes costs time,
because `avOut` and `lseOut` are tested with `if (avOut)` INSIDE the kernel's
inner loop -- so nulling them buys a per-element branch and sells writes that
were already hidden behind the MMA work.

**Writes hidden behind compute are free; a branch to avoid them is not.**
That is the same wall the overlap probe (3.1%) and the graph bound (0.6%)
each found from their own side.

### dKf/dVf looked dead and are not

`flashBwdK` writes dK and dV through `dcomb` as bf16 PAIRS when `dcomb` is
non-null, and triattn always passes it -- so the f32 `dKf`/`dVf` looked like
the `dcomb == nullptr` fallback, dead on the default path.  They are also
memset every backward: **67.2 MB a call, 6.45 GB a fwd+bwd**, which CUPTI
never showed because it records memsets under a different activity kind than
kernels.  It looked like 1.1% of free work.

**They are live.**  `scatterDxK` reads dQ and the gate out of `dcomb` with
`fromComb`, and dK and dV out of the f32 buffers with **`fromF32(dK, ...)`**.
Removing them is an illegal memory access, which is how it was caught.

The reading error is worth naming: the grep was for `dK[` and `dV[`, and
every real use is a function ARGUMENT, not a subscript.  Searching for how a
buffer is indexed misses every use that passes it along.

**The good news is the failure mode.**  A dead-buffer mistake in this
directory crashes rather than returning a quiet wrong gradient, because the
consumer dereferences null immediately.  That is not true of the keep tiers,
where the same class of mistake (KEEP_MSA's missing `av`) showed up as
grad/pair 0.9716 instead.

## The traffic CUPTI does not show: 26.3 ms of zeroing

CUPTI records memsets and memcpys under DIFFERENT activity kinds than
kernels, so none of them appear in the profile that found everything else
here.  Totalled by hand at N=256, a fwd+bwd moves ~8.5 GB that way, and one
item dominates: the triangle attention's f32 `dKf`/`dVf` accumulators are
zeroed at the top of every backward, 67.1 MB a call, **6.54 GB a fwd+bwd**.

`PROBE_NOZERO` skips the zeroing -- wrong arithmetic, identical kernels --
over three alternating pairs:

| | fwd+bwd |
|---|---|
| zeroed | 835.48 / 842.59 / 841.53 |
| **not zeroed** | **811.30 / 813.54 / 816.09** |

**-26.3 ms, 3.1%,** and invisible to every profile taken today.

### Making the accumulators bf16: the gradient is fine and it is 19% slower

Halving them halves the zeroing, the atomic traffic and what `scatterDxK`
reads back.  `flash_bwd.cuh` already carries a note rejecting a bf16x2
atomic -- but it rejected one into `dcomb`, which is 516 wide, and blamed the
stride (1.946 -> 3.527 ms).  A compact `[N*N][W]` bf16 buffer has the f32
buffers' locality exactly.

Built it.  **The gradient is untouched** -- grad/pair 0.975473 / 0.975459 /
0.975419, squarely in band, so 2-4 deep bf16 accumulation costs nothing
numerically.  **And fwd+bwd goes 837 -> 992 ms, +19%.**

So the stride was never the problem: `atomicAdd(__nv_bfloat162*)` has no
hardware path on Ampere where f32 `atomicAdd` does, and under 2-4 way
contention that costs far more than halving the bytes saves.  The earlier
note's conclusion was right and its reason was wrong.

### And a vectorised zeroing kernel wins standalone and loses in the model

6.54 GB in 26.3 ms is 249 GB/s.  `zeroFast` (float4 stores, `common.cuh`)
does the same job:

| | standalone fwd+bwd | in-model, L=224 |
|---|---|---|
| `cudaMemsetAsync` | 842.93 / 842.80 / 848.08 | **996.2 / 999.9** |
| `zeroFast` | **838.16 / 840.21 / 844.50** | 999.9 / 1002.6 |

**-3.6 ms standalone and +3.4 ms in the model**, each consistent across its
own pairs.  A memset can be serviced without occupying SMs; a zeroing kernel
competes for them, and in the model it competes against XLA's interleaved
work as well.  **The model is what ships**, so the default is the memset and
`TA_ZEROK=1` selects the kernel.

That is the fourth time today standalone and in-model have disagreed, and
the second where standalone had the wrong sign.  Nothing here is believed
until it is measured in the model, paired.
