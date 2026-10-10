"""Install the native CUDA Evoformer into BindCraft2's AF2 model.

The boundary is the TRUNK, and that is a measured choice rather than a
convenient one: the Evoformer is ~99% of a design step's runtime (a JAX
trunk-only step at L=128 costs 1372 ms against 1382 ms for the whole chain),
and the losses cost 2.25 s of compile and 0.18 ms a step. So the trunk goes
native and everything else -- heads, losses, the schedule -- stays in JAX,
where a new loss term is a Python edit rather than a CUDA rebuild.

WEIGHTS ARE NOT ARGUMENTS. Design freezes them, so they are loaded once from
the manifest weights.py writes and kept on the device. Passing 85 parameters
x 48 depths through the FFI would be 4080 buffers to express something that
never changes during a run.

install() is a no-op if the library is missing, so an environment without the
build still runs stock.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

_LIB = os.environ.get(
    'BC2_NATIVE_LIB',
    '/home/ubuntu/ColabDesign2/dev/native_block/libnative_evoformer.so')
_WEIGHTS = os.environ.get(
    'BC2_NATIVE_WEIGHTS',
    '/home/ubuntu/ColabDesign2/dev/native_block/stack_weights')

_registered = False

# haiku's param tree uses '/' inside module names and the native manifest keys
# on the SAME paths the checkpoint does, so the two differ only in how the
# stacked leading axis is spelled.
_MAIN = 'alphafold/alphafold_iteration/evoformer/evoformer_iteration'


def weights_from_params(params, out_dir=None, depth=None):
    """Write a native weights manifest from a LIVE haiku param tree.

    WHY THIS EXISTS: the stack loads its weights from a directory that
    weights.py writes from a checkpoint file.  That is a second copy of the
    weights and a separate step per checkpoint -- and BindCraft2 already holds
    the params in memory, possibly ones the caller has modified.  Deriving the
    manifest from THOSE means the native path runs whatever the model is
    actually using, including a checkpoint this repo has never seen.

    Cached on the content of the arrays, not on a name: two models with the
    same name and different weights must not share a directory.
    """
    import hashlib
    import tempfile

    mods = {k: v for k, v in params.items() if k.startswith(_MAIN)}
    if not mods:
        raise KeyError(f'no {_MAIN}* in these params')
    if depth is None:
        depth = min(int(np.asarray(a).shape[0])
                    for d in mods.values() for a in d.values())

    h = hashlib.sha1()
    for mod in sorted(mods):
        for pname in sorted(mods[mod]):
            h.update(mod.encode())
            h.update(pname.encode())
            h.update(np.ascontiguousarray(
                np.asarray(mods[mod][pname], np.float32)).tobytes()[:4096])
    tag = h.hexdigest()[:16]
    out = out_dir or os.path.join(tempfile.gettempdir(), f'bc2_native_{tag}')
    if os.path.exists(os.path.join(out, 'manifest.txt')):
        return out                      # already materialised this exact tree
    os.makedirs(out, exist_ok=True)
    total = 0
    with open(os.path.join(out, 'manifest.txt'), 'w') as fh:
        for mod in sorted(mods):
            path = mod[len(_MAIN) + 1:] if len(mod) > len(_MAIN) else ''
            for pname in sorted(mods[mod]):
                w = np.asarray(mods[mod][pname], np.float32)
                for b in range(depth):
                    name = f'param/block{b}/{path}/{pname}'
                    arr = np.ascontiguousarray(w[b], dtype=np.float32)
                    fn = name.replace('/', '__') + '.f32'
                    arr.tofile(os.path.join(out, fn))
                    fh.write(f'{name} {fn} {arr.ndim} '
                             f'{" ".join(str(x) for x in arr.shape)}\n')
                    total += arr.size
    print(f'native: materialised {total/1e6:.1f}M floats x {depth} blocks '
          f'from live params -> {out}')
    return out


def available() -> bool:
    return os.path.exists(_LIB) and os.path.isdir(_WEIGHTS)


def _register():
    global _registered
    if _registered:
        return
    import ctypes
    lib = ctypes.CDLL(_LIB)
    # THE EXPORTED SYMBOL IS THE C++ HANDLER'S NAME, not a snake_case
    # convention: XLA_FFI_DEFINE_HANDLER_SYMBOL(NativeEvoformerFwd, ...)
    # exports exactly `NativeEvoformerFwd`.  Checked with nm, not guessed.
    for target, symbol in (('native_evoformer_fwd', 'NativeEvoformerFwd'),
                           ('native_evoformer_bwd', 'NativeEvoformerBwd')):
        jax.ffi.register_ffi_target(
            target, jax.ffi.pycapsule(getattr(lib, symbol)), platform='CUDA')
    _registered = True


def _make_stack(depth, weights_dir):
    """Build the custom_vjp for ONE (depth, weights) pair.

    depth and weights_dir are CLOSED OVER, not passed: custom_vjp requires
    every argument to be a JAX type, and a weights path is a Python string.
    A closure is also the honest shape -- they are fixed for the life of an
    installed model, not per-call values.
    """

    def raw(pair, msa, pair_mask, msa_mask):
        out = jax.ffi.ffi_call(
            'native_evoformer_fwd',
            (jax.ShapeDtypeStruct(pair.shape, jnp.float32),
             jax.ShapeDtypeStruct(msa.shape, jnp.float32),
             jax.ShapeDtypeStruct((depth,) + pair.shape, jnp.bfloat16),
             jax.ShapeDtypeStruct((depth,) + msa.shape, jnp.float32)),
            vmap_method='sequential')
        return out(pair, msa, pair_mask, msa_mask,
                   depth=np.int64(depth), save=np.int64(1),
                   pair_only=np.int64(0), own_save=np.int64(0),
                   opm_first=np.int64(1), weights_dir=weights_dir)

    @jax.custom_vjp
    def stack(pair, msa, pair_mask, msa_mask):
        p, m, _, _ = raw(pair, msa, pair_mask, msa_mask)
        return p, m

    def fwd(pair, msa, pair_mask, msa_mask):
        p, m, psave, msave = raw(pair, msa, pair_mask, msa_mask)
        # THE SAVE-SET CROSSES AS A REAL VALUE, not as state held inside the
        # stack.  JAX dispatches asynchronously, so a second forward can
        # overtake the first backward -- a wrong gradient rather than an error.
        return (p, m), (pair_mask, msa_mask, psave, msave)

    def bwd(res, g):
        pair_mask, msa_mask, psave, msave = res
        gp, gm = g
        out = jax.ffi.ffi_call(
            'native_evoformer_bwd',
            (jax.ShapeDtypeStruct(gp.shape, jnp.float32),
             jax.ShapeDtypeStruct(gm.shape, jnp.float32)),
            vmap_method='sequential')
        return out(gp, gm, pair_mask, msa_mask, psave, msave,
                   depth=np.int64(depth), pair_only=np.int64(0),
                   own_save=np.int64(0), opm_first=np.int64(1),
                   weights_dir=weights_dir)

    stack.defvjp(fwd, bwd)
    return stack


def install(depth: int = 48, weights_dir: str = None, params=None):
    """Point modules_multimer's hook at the native stack.

    `params` is a live haiku param tree -- pass the one the model is using and
    the weights are derived from it, so no checkpoint conversion step is
    needed and a checkpoint this repo has never seen works unchanged.
    """
    if params is not None and weights_dir is None:
        weights_dir = weights_from_params(params, depth=depth)
    if not os.path.exists(_LIB):
        print('native: library missing, staying on JAX')
        return False
    if weights_dir is None and not os.path.isdir(_WEIGHTS):
        print('native: library or weights missing, staying on JAX')
        return False
    _register()
    wd = weights_dir or _WEIGHTS
    from .alphafold.model import modules_multimer

    cache = {}

    def run(acts, masks, n_block):
        # LENGTH MUST BE 32-ALIGNED.  Every gate in this port ran at 128, 224,
        # 256, 375 or 384, so an unaligned length has never been exercised --
        # and L=100 does not fail cleanly, it reaches `misaligned address`
        # inside cuBLASLt.  Refuse it here, where the message can say why.
        # BindCraft2 pads to length buckets (default 32) so a real run is
        # already aligned; this bites only at length_bucket_size=1.
        n = int(acts['pair'].shape[0])
        if n % 32:
            raise ValueError(
                f'native evoformer: length {n} is not a multiple of 32. '
                'Use length_bucket_size=32 (the default) or pad the input; '
                'the CUDA kernels have only ever been gated on 32-aligned '
                'lengths and an unaligned one faults inside cuBLASLt.')
        pair = acts['pair'].astype(jnp.float32)
        msa = acts['msa'].astype(jnp.float32)
        fn = cache.get(n_block)
        if fn is None:
            fn = cache[n_block] = _make_stack(int(n_block), wd)
        p, m = fn(pair, msa,
                  masks['pair'].astype(jnp.float32),
                  masks['msa'].astype(jnp.float32))
        return {'pair': p.astype(acts['pair'].dtype),
                'msa': m.astype(acts['msa'].dtype)}

    modules_multimer.NATIVE_EVOFORMER = run
    return True


def uninstall():
    from .alphafold.model import modules_multimer
    modules_multimer.NATIVE_EVOFORMER = None
