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


_MODEL_RUNNERS = {}


def _make_runner(weights_dir, depth=48, dropout=False):
    """A hook callable bound to ONE weights directory."""
    cache = {}

    def run(acts, masks, n_block, seed=None):
        n = int(acts['pair'].shape[0])
        if n % 32:
            raise ValueError(
                f'native evoformer: length {n} is not a multiple of 32. '
                'Use length_bucket_size=32 (the default) or pad; the kernels '
                'have only ever been gated on 32-aligned lengths and an '
                'unaligned one faults inside cuBLASLt.')
        fn = cache.get(n_block)
        if fn is None:
            fn = cache[n_block] = _make_stack(int(n_block), weights_dir,
                                              dropout)
        if seed is None:
            seed = jnp.zeros((2,), jnp.uint32)   # [seed, enabled]
        p, m = fn(acts['pair'].astype(jnp.float32),
                  acts['msa'].astype(jnp.float32),
                  masks['pair'].astype(jnp.float32),
                  masks['msa'].astype(jnp.float32),
                  seed.astype(jnp.uint32).reshape((2,)))
        return {'pair': p.astype(acts['pair'].dtype),
                'msa': m.astype(acts['msa'].dtype)}

    return run


def _make_stack(depth, weights_dir, dropout=False):
    """Build the custom_vjp for ONE (depth, weights) pair.

    depth and weights_dir are CLOSED OVER, not passed: custom_vjp requires
    every argument to be a JAX type, and a weights path is a Python string.
    A closure is also the honest shape -- they are fixed for the life of an
    installed model, not per-call values.
    """

    # PER RUNNER, never module-level.  One campaign installs twice -- the
    # design pool with dropout, the validation pool without -- into the same
    # registry, and the stacks are built lazily on first trace, i.e. after
    # both installs.  A global would hand the design pool whichever flag the
    # LAST install carried, and the stack would quietly compute the rate-0
    # model while every log line still said dropout was on.
    dropout = np.int64(1 if dropout else 0)

    def raw(pair, msa, pair_mask, msa_mask, seed):
        out = jax.ffi.ffi_call(
            'native_evoformer_fwd',
            (jax.ShapeDtypeStruct(pair.shape, jnp.float32),
             jax.ShapeDtypeStruct(msa.shape, jnp.float32),
             jax.ShapeDtypeStruct((depth,) + pair.shape, jnp.bfloat16),
             jax.ShapeDtypeStruct((depth,) + msa.shape, jnp.float32)),
            vmap_method='sequential')
        return out(pair, msa, pair_mask, msa_mask, seed,
                   depth=np.int64(depth), save=np.int64(1),
                   pair_only=np.int64(0), own_save=np.int64(0),
                   opm_first=np.int64(1), dropout=dropout,
                   weights_dir=weights_dir)

    @jax.custom_vjp
    def stack(pair, msa, pair_mask, msa_mask, seed):
        p, m, _, _ = raw(pair, msa, pair_mask, msa_mask, seed)
        return p, m

    def fwd(pair, msa, pair_mask, msa_mask, seed):
        p, m, psave, msave = raw(pair, msa, pair_mask, msa_mask, seed)
        # THE SAVE-SET CROSSES AS A REAL VALUE, not as state held inside the
        # stack.  JAX dispatches asynchronously, so a second forward can
        # overtake the first backward -- a wrong gradient rather than an error.
        # THE SEED CROSSES TOO.  The backward regenerates the masks rather
        # than storing them, so it must be handed the same seed the forward
        # drew -- a different one silently scales the delta's cotangent by a
        # mask the forward never applied.
        return (p, m), (pair_mask, msa_mask, psave, msave, seed)

    def bwd(res, g):
        pair_mask, msa_mask, psave, msave, seed = res
        gp, gm = g
        out = jax.ffi.ffi_call(
            'native_evoformer_bwd',
            (jax.ShapeDtypeStruct(gp.shape, jnp.float32),
             jax.ShapeDtypeStruct(gm.shape, jnp.float32)),
            vmap_method='sequential')
        dpair, dmsa = out(gp, gm, pair_mask, msa_mask, psave, msave, seed,
                          depth=np.int64(depth), pair_only=np.int64(0),
                          own_save=np.int64(0), opm_first=np.int64(1),
                          dropout=dropout, weights_dir=weights_dir)
        # ONE COTANGENT PER PRIMAL ARGUMENT.  The primal takes four
        # (pair, msa, pair_mask, msa_mask); the masks are not differentiated,
        # so they get explicit zeros rather than being omitted -- custom_vjp
        # checks the pytree structure, not just the differentiable subset.
        # the seed is an integer input, so its cotangent is float0 -- the
        # dtype JAX uses for "this argument has no tangent space at all",
        # which zeros_like on a uint32 array does NOT produce
        return (dpair, dmsa, jnp.zeros_like(pair_mask), jnp.zeros_like(msa_mask),
                np.zeros(seed.shape, dtype=jax.dtypes.float0))

    stack.defvjp(fwd, bwd)
    return stack


def use_model(name: str):
    """Point the hook at a PARTICULAR model's weights.

    bc2 samples a model per step out of its pool (_sample_design_model), and a
    single installed weights directory would serve model_1's weights for every
    one of them -- silently, across a whole campaign.  af2.py calls this before
    building the compiled function for a model, and the closure captured at
    trace time is the right one because that function is cached per model.
    """
    from .alphafold.model import modules_multimer
    run = _MODEL_RUNNERS.get(name)
    if run is None:
        raise KeyError(
            f'native: no weights installed for model {name!r}; '
            f'have {sorted(_MODEL_RUNNERS)}. install_models() takes the whole '
            'pool so each model gets its own directory.')
    modules_multimer.NATIVE_EVOFORMER = run


def install_models(model_parameters: dict, depth: int = 48,
                   dropout: bool = False):
    """Install EVERY model in the pool, one weights directory each."""
    if dropout:
        print('native: dropout ALLOCATED.  The masks come from a counter-based hash '
              'of (seed, depth, module, index), NOT from threefry, so they do '
              'not match what stock JAX would have drawn -- a native run and '
              'a stock run are comparable in distribution, never step by '
              'step.  Both the seed and the on/off flag are runtime inputs, '
              'so a stage that turns dropout off costs no recompile.')
    if not available():
        print('native: library missing, staying on JAX')
        return False
    _register()
    for name, params in model_parameters.items():
        wd = weights_from_params(params, depth=depth)
        _MODEL_RUNNERS[name] = _make_runner(wd, depth, bool(dropout))
    first = next(iter(model_parameters))
    use_model(first)
    print(f'native: installed {len(_MODEL_RUNNERS)} model(s): '
          f'{sorted(_MODEL_RUNNERS)}')
    return True


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
