# Native Evoformer kernels

AF2's Evoformer trunk, forward and backward, as CUDA rather than XLA. It is
~99% of a design step, so replacing it roughly halves the wall clock of a
BindCraft design step: **1.97x** at 224 tokens, **1.86x** at 128, measured one
gradient step at a time against stock JAX on an A10 with dropout on.

The speedup is below the trunk's own 3.4-6.3x because the heads, the losses
and the structure module stay in JAX, where a new loss term is a Python edit
rather than a CUDA rebuild.

## Building

`install.sh` does this for you. By hand:

```bash
source .venv/bin/activate     # jax.ffi's headers come from the environment
bash native/build.sh
```

Needs the CUDA toolkit (`nvcc`) and a GPU of compute capability 8.0 or newer
(A100, A10/A40/3090, L4/L40S, H100; Blackwell and later via PTX).

**Not Colab's T4, and not a V100.** The whole numeric chain here is bf16 --
the save-set, the nine residual steps, every GEMM -- and bf16 arithmetic and
bf16 tensor cores start at Ampere. Turing and Volta have fp16 tensor cores and
no bf16 at all. Running there would mean a second numeric chain in fp16, whose
quarter-sized exponent range over 48 blocks is a real overflow risk, or fp32,
which gives up the tensor cores that make this fast. So those cards run the
Evoformer on JAX and the run says so.

The build is one fat binary over every supported architecture, takes a couple
of minutes and produces `libnative_evoformer.so` here.

**The build is optional.** Without `nvcc`, without the headers, or on an older
card, `build.sh` says so and exits 0, and BindCraft runs the Evoformer on JAX:
the same model, about twice as slow. Nothing fails.

## Using it

`evoformer_backend` picks the trunk: `auto` (the default) takes these kernels
where the card and build allow and the JAX trunk otherwise, `native` insists
and errors if they cannot run here, `jax` asks for the JAX trunk.

To run on JAX:

```bash
python bindcraft.py mycampaign.json --jax
python bindcraft.py mycampaign.json evoformer_backend=jax    # the same thing
```

## What it is not

The masks AF2 applies for dropout come here from a counter-based hash of
(seed, depth, module, index), not from JAX's threefry. A native run and a JAX
run are therefore comparable **in distribution** and never step by step, even
from the same seed. Everything else is the same model: the rate-0 path gates
against a JAX oracle at cos 0.9998 forward and 0.983 on the gradient.

Lengths are padded to a multiple of 32 and the pad is masked out. The pad is
inert -- a control that puts values 100x larger under the mask does not move
the real output one bit.

## Where this came from

These files are vendored from `ColabDesign2/dev/native_block`, which is where
they are developed and where their gates live. `SOURCE_COMMIT` records the
ColabDesign2 commit this copy was taken from, and `native/sync.sh` re-copies
and tells you whether anything changed. **A fix made in ColabDesign2 does not
reach here on its own** -- run the sync.
