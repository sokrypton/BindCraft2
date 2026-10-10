#!/usr/bin/env bash
# Build the native Evoformer kernels.
#
# One fat binary over the cards BindCraft runs on, so the same .so serves an
# A10, an A100 and an H100.  ~2 minutes, ~8 MB.
#
# install.sh calls this.  It is safe to call by hand after editing a kernel,
# and safe to skip entirely: without the library BindCraft runs the Evoformer
# on JAX, which is the same model about twice as slow.
set -euo pipefail
cd "$(dirname "$0")"

OUT=libnative_evoformer.so

if ! command -v nvcc > /dev/null; then
  echo "native: no nvcc on PATH, skipping the kernel build."
  echo "native: BindCraft will run the Evoformer on JAX (about 2x slower)."
  echo "native: install the CUDA toolkit and re-run native/build.sh to enable it."
  exit 0
fi

# jax.ffi's headers describe the custom-call ABI the library registers against.
INC=$(python -c 'import jax.ffi; print(jax.ffi.include_dir())' 2>/dev/null || true)
if [ -z "$INC" ]; then
  echo "native: cannot find jax.ffi headers -- is the BindCraft environment active?"
  echo "native: skipping; BindCraft will run the Evoformer on JAX."
  exit 0
fi

# sm_80 A100, sm_86 A10/A40/3090, sm_89 L4/L40S, sm_90 H100.
#
# The last line emits PTX as well as cubin, so a card NEWER than anything here
# -- Blackwell, and whatever follows -- JITs from that PTX on first launch
# instead of failing with 'no kernel image is available'.  It costs a one-off
# JIT at startup and nothing after.
#
# A card OLDER than sm_80 cannot be served at all: this is bf16 tensor-core
# and cp.async code, so Turing (T4, 2080) and Volta (V100) are out.  native.py
# checks the capability before installing and falls back to the JAX trunk, so
# such a machine runs correctly, just at JAX speed.
echo "native: building $OUT (this takes a couple of minutes)"
nvcc -O3 -std=c++17 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_90,code=compute_90 \
  --compiler-options -fPIC -shared -I. -I"$INC" \
  ffi.cu -lcublas -lcublasLt -o "$OUT"

echo "native: built $(pwd)/$OUT"
