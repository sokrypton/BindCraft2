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

# WHERE THE KERNELS LIVE.  Vendored, in native/ at the repository root, so a
# clone of this repository is all you need -- `install.sh` builds them and
# `native/sync.sh` re-copies from ColabDesign2, where they are developed.
#
# BC2_NATIVE_LIB overrides.  Otherwise: the vendored build first, then a
# ColabDesign2 checkout beside this one or under $HOME, which is what a
# developer working on the kernels has.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))        # the bindcraft checkout
_LIBNAME = 'libnative_evoformer.so'


def _candidates():
    here = [os.path.join(_REPO, 'native', _LIBNAME)]
    roots = []
    if os.environ.get('COLABDESIGN2'):
        roots.append(os.environ['COLABDESIGN2'])
    roots += [os.path.join(os.path.dirname(_REPO), 'ColabDesign2'),
              os.path.expanduser('~/ColabDesign2')]
    return here + [os.path.join(r, 'dev', 'native_block', _LIBNAME)
                   for r in roots]


def _find_lib():
    override = os.environ.get('BC2_NATIVE_LIB')
    if override:
        return override
    for candidate in _candidates():
        if os.path.exists(candidate):
            return candidate
    return _candidates()[0]        # the vendored path, for the error message


_LIB = _find_lib()
# ONLY the legacy install() path reads weights from disk.  install_models()
# materialises them from the live param tree, which is what a campaign uses,
# so a weights directory is NOT a requirement for the native path to work --
# requiring one here made native unavailable on every machine without the
# development harness.
_WEIGHTS = os.environ.get(
    'BC2_NATIVE_WEIGHTS',
    os.path.join(os.path.dirname(_REPO), 'ColabDesign2', 'dev', 'native_block',
                 'stack_weights'))


# THE KERNELS DO NOT RUN ON EVERY NVIDIA GPU.  They are bf16 tensor-core and
# cp.async code, so compute capability 8.0 is the floor: Ampere (A100, A10,
# A40, 3090), Ada (L4, L40S, 4090) and Hopper (H100, H200) run them; Turing
# (T4, 2080) and Volta (V100) do not, and neither does anything older.
#
# This matters because the native path is the DEFAULT.  A launch on a card the
# binary has no image for fails with a CUDA error in the middle of a campaign,
# which is a much worse outcome than running on JAX -- so the card is checked
# here, before anything is installed, and a card below the floor simply gets
# the JAX trunk with a line saying why.
_MIN_CC = (8, 0)


def _device_cc():
    """(major, minor) of the default JAX GPU, or None if there isn't one."""
    try:
        import jax
        d = jax.devices()[0]
        if d.platform != 'gpu':
            return None
        cc = getattr(d, 'compute_capability', None)
        if cc is None:
            return None
        cc = str(cc)
        return (tuple(int(x) for x in cc.split('.'))[:2] if '.' in cc
                else (int(cc[:-1]), int(cc[-1])))
    except Exception:
        return None                 # a probe must not be the failure


def why_unavailable() -> str:
    """One line saying what is missing, for a run that asked for native."""
    if not os.path.exists(_LIB):
        return (f'no kernel library at {_LIB} -- build it with '
                '`bash native/build.sh` (needs nvcc and a compute-8.0+ GPU)')
    cc = _device_cc()
    if cc is not None and cc < _MIN_CC:
        return (f'this GPU is sm_{cc[0]}{cc[1]} and the kernels need '
                f'sm_{_MIN_CC[0]}{_MIN_CC[1]} or newer (Ampere and later)')
    return ''


_registered = False

# haiku's param tree uses '/' inside module names and the native manifest keys
# on the SAME paths the checkpoint does, so the two differ only in how the
# stacked leading axis is spelled.
_MAIN = 'alphafold/alphafold_iteration/evoformer/evoformer_iteration'
# The EXTRA-MSA stack is the same EvoformerIteration with its own 4 blocks of
# weights.  Its pair track is identical to the main stack's in name and shape;
# the only module it does not share is msa_column_GLOBAL_attention, which the
# pair-only path never runs.
_EXTRA = 'alphafold/alphafold_iteration/evoformer/extra_msa_stack'
# The TEMPLATE pair stack: two 64-channel iterations, its own weights under
# the template embedding.  Templates are the largest remaining piece of the
# design step's XLA compile once the extra-MSA stack is native.
_TPL = ('alphafold/alphafold_iteration/evoformer/template_embedding/'
        'single_template_embedding/template_embedding_iteration')


def weights_from_params(params, out_dir=None, depth=None, prefix=_MAIN):
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

    mods = {k: v for k, v in params.items() if k.startswith(prefix)}
    if not mods:
        raise KeyError(f'no {prefix}* in these params')
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
            path = mod[len(prefix) + 1:] if len(mod) > len(prefix) else ''
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
    return why_unavailable() == ''


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
                           ('native_evoformer_bwd', 'NativeEvoformerBwd'),
                           ('template_pair_stack_fwd', 'TemplatePairStackFwd'),
                           ('template_pair_stack_bwd', 'TemplatePairStackBwd')):
        jax.ffi.register_ffi_target(
            target, jax.ffi.pycapsule(getattr(lib, symbol)), platform='CUDA')
    _registered = True


_MODEL_RUNNERS = {}
_EXTRA_RUNNERS = {}
_TPL_RUNNERS = {}


def supported_length(n: int) -> bool:
    """Can the kernels take this complex length?

    Asked BEFORE the hook draws its seed.  Drawing consumes haiku's SafeKey,
    and a key that has been used cannot be handed to the JAX fallback -- it
    raises 'Random key has been used previously'.  So support is decided
    first and the key is touched only on the path that will use it.
    """
    return True   # every length: run() pads to 32 and masks the pad out


class Unsupported(Exception):
    """This length cannot go through the kernels; the caller should use JAX.

    Raised at TRACE time, not per step, so the fallback costs nothing at
    runtime -- the hook picks a path once per compiled function.
    """


def _make_runner(weights_dir, depth=48, dropout=False):
    """A hook callable bound to ONE weights directory."""
    cache = {}

    def run(acts, masks, n_block, seed=None):
        # PAD TO 32 AND MASK THE PAD OUT.  The kernels take 32-aligned
        # lengths only, and a campaign's length is 32-aligned by luck:
        # bindcraft buckets the DESIGNED chain, not the complex, so a PD-L1
        # run arrives as 211 whatever binder length is asked for.  Refusing
        # meant the native path could not run a real campaign at all.
        #
        # Measured, 64 real residues embedded in a masked 96:
        #   forward   the pad is INERT -- maskgate's same-N control puts
        #             values 100x larger under the mask and the real corner
        #             does not move one bit
        #   gradient  no NaNs, corner cosine 0.998 against the unpadded run
        # The residual (3.5e-03 on the loss, 8.8e-03 on the forward) is a
        # different N taking a different reduction order on a bf16 chain, not
        # a leak, and it is smaller than the port's own distance from JAX
        # (grad/pair cos 0.983 at rate 0).
        #
        # An earlier version of this comment said a masked pad NaNs the
        # gradient.  That was the harness: the probe used sum(p**2), whose
        # cotangent is ~1e4 here and overflows f16 in the triangle attention
        # backward.  It NaNs at every length with an all-ones mask too, so it
        # was never about padding.  It is a real robustness limit of the
        # backward, just an unrelated one.
        n = int(acts['pair'].shape[0])
        pad = (-n) % 32
        fn = cache.get(n_block)
        if fn is None:
            fn = cache[n_block] = _make_stack(int(n_block), weights_dir,
                                              dropout)
        if seed is None:
            seed = jnp.zeros((2,), jnp.uint32)   # [seed, enabled]
        pair = acts['pair'].astype(jnp.float32)
        msa = acts['msa'].astype(jnp.float32)
        pair_mask = masks['pair'].astype(jnp.float32)
        msa_mask = masks['msa'].astype(jnp.float32)
        if pad:
            pair = jnp.pad(pair, ((0, pad), (0, pad), (0, 0)))
            msa = jnp.pad(msa, ((0, 0), (0, pad), (0, 0)))
            pair_mask = jnp.pad(pair_mask, ((0, pad), (0, pad)))
            msa_mask = jnp.pad(msa_mask, ((0, 0), (0, pad)))
        p, m = fn(pair, msa, pair_mask, msa_mask,
                  seed.astype(jnp.uint32).reshape((2,)))
        if pad:
            p, m = p[:n, :n], m[:, :n]
        return {'pair': p.astype(acts['pair'].dtype),
                'msa': m.astype(acts['msa'].dtype)}

    return run


def _make_tpl_runner(weights_dir, depth=2, dropout=False):
    """AF2's template pair stack, as two native 64-channel iterations.

    Same shape of win as the extra-MSA stack and for the same reason: two
    blocks or forty-eight, layer_stack scans them so XLA compiles ONE block
    body, and 70% of that cost is the reverse pass.
    """
    drop = np.int64(1 if dropout else 0)

    def raw(act, mask, seed):
        n = act.shape[0]
        out = jax.ffi.ffi_call(
            'template_pair_stack_fwd',
            (jax.ShapeDtypeStruct(act.shape, jnp.float32),
             # the save-set is the INPUT: the backward replays from it rather
             # than keeping each module's state, see tplstack.cuh
             jax.ShapeDtypeStruct((n * n * 64,), jnp.bfloat16)),
            vmap_method='sequential')
        return out(act, mask, seed, depth=np.int64(depth), dropout=drop,
                   weights_dir=weights_dir)

    @jax.custom_vjp
    def stack(act, mask, seed):
        return raw(act, mask, seed)[0]

    def fwd(act, mask, seed):
        o, save = raw(act, mask, seed)
        return o, (mask, save, seed)

    def bwd(res, g):
        mask, save, seed = res
        out = jax.ffi.ffi_call(
            'template_pair_stack_bwd',
            jax.ShapeDtypeStruct(g.shape, jnp.float32),
            vmap_method='sequential')
        dact = out(g, mask, save, seed, depth=np.int64(depth), dropout=drop,
                   weights_dir=weights_dir)
        return (dact, jnp.zeros_like(mask),
                np.zeros(seed.shape, dtype=jax.dtypes.float0))

    stack.defvjp(fwd, bwd)

    def run(act, mask, n_block, seed=None):
        if int(n_block) != depth:
            raise ValueError(f'template stack built for depth {depth}, '
                             f'asked for {int(n_block)}')
        if seed is None:
            seed = jnp.zeros((2,), jnp.uint32)
        return stack(act.astype(jnp.float32), mask.astype(jnp.float32),
                     seed.astype(jnp.uint32).reshape((2,))).astype(act.dtype)

    return run


def _make_extra_runner(weights_dir, depth=4, dropout=False):
    """The extra-MSA stack, pair-only.

    Four blocks of the same triangle ops as the main trunk, and in XLA they
    cost about what the 48-block trunk would: layer_stack scans them, so the
    compiler sees ONE block body either way, forward and reverse.  Taking them
    native removes that body from the graph.

    The caller discards this stack's msa output (modules_multimer keeps only
    `extra_evoformer_output['pair']`), and its msa side cannot reach its pair
    because bindcraft's extra_msa_mask is all zeros -- so a dummy msa goes in
    and whatever comes back is handed straight back unused.
    """
    cache = {}

    def run(acts, masks, n_block, seed=None, extra_msa_mask=None):
        # THE PRECONDITION IS CHECKED, NOT ASSUMED.  This path is only valid
        # while the extra MSA is entirely masked: then its msa side cannot
        # reach its pair and the outer product mean collapses to a constant.
        # With a real extra alignment the constant is WRONG and the fold is
        # silently wrong with it, so the one fact everything rests on is
        # established from the features before anything is traced.  The same
        # check guards the ColabDesign2 hook (install_hook.require_dead_extra_msa).
        # A TRACER CANNOT BE READ.  Inside the model this is a traced array,
        # so the check can only run when a concrete one is handed in -- which
        # is why ColabDesign2 checks the numpy features before tracing
        # instead.  For bindcraft the guarantee is structural rather than
        # dynamic: alphafold_input_features writes
        # `'extra_msa_mask': jnp.zeros(...)` unconditionally, so the mask is
        # dead by construction for every design and prediction this package
        # builds.  This catches a caller that ever changes that.
        if extra_msa_mask is not None and not isinstance(
                extra_msa_mask, jax.core.Tracer):
            m = np.asarray(extra_msa_mask)
            if m.any():
                raise ValueError(
                    f'native extra-MSA stack: extra_msa_mask has '
                    f'{int((m != 0).sum())} unmasked entries, so its MSA side '
                    'reaches the pair and this path is invalid. It is only '
                    'legal for a design run, whose extra MSA is all zeros.')
        n = int(acts['pair'].shape[0])
        pad = (-n) % 32
        fn = cache.get(n_block)
        if fn is None:
            fn = cache[n_block] = _make_stack(int(n_block), weights_dir,
                                              dropout, pair_only=True)
        if seed is None:
            seed = jnp.zeros((2,), jnp.uint32)
        pair = acts['pair'].astype(jnp.float32)
        pair_mask = masks['pair'].astype(jnp.float32)
        if pad:
            pair = jnp.pad(pair, ((0, pad), (0, pad), (0, 0)))
            pair_mask = jnp.pad(pair_mask, ((0, pad), (0, pad)))
        # the stack's msa inputs are read but cannot affect the pair here
        dummy_msa = jnp.zeros((1, pair.shape[0], 256), jnp.float32)
        dummy_mask = jnp.zeros((1, pair.shape[0]), jnp.float32)
        p, _ = fn(pair, dummy_msa, pair_mask, dummy_mask,
                  seed.astype(jnp.uint32).reshape((2,)))
        if pad:
            p = p[:n, :n]
        return {'pair': p.astype(acts['pair'].dtype), 'msa': acts['msa']}

    return run


def _make_stack(depth, weights_dir, dropout=False, pair_only=False):
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
    # PAIR-ONLY is how the extra-MSA stack runs: its msa side cannot reach
    # its pair (extra_msa_mask is all zeros in design), so the four MSA
    # modules are skipped and the outer product mean collapses to a
    # constant.  See the comment in block.cuh.
    pair_only = np.int64(1 if pair_only else 0)

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
                   pair_only=pair_only, own_save=np.int64(0),
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
                          depth=np.int64(depth), pair_only=pair_only,
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
    modules_multimer.NATIVE_EXTRA_MSA_STACK = _EXTRA_RUNNERS.get(name)
    modules_multimer.NATIVE_TEMPLATE_STACK = _TPL_RUNNERS.get(name)


def install_models(model_parameters: dict, depth: int = 48,
                   dropout: bool = False):
    """Install EVERY model in the pool, one weights directory each."""
    # THE AVAILABILITY CHECK COMES FIRST.  On a card the kernels cannot serve
    # -- a Colab T4, say -- the only useful line is the one saying so, and
    # printing a paragraph about dropout masks before it just buries it.
    if not available():
        print(f'native: {why_unavailable()}; staying on JAX')
        return False
    if dropout:
        print('native: dropout ALLOCATED.  The masks come from a counter-based hash '
              'of (seed, depth, module, index), NOT from threefry, so they do '
              'not match what stock JAX would have drawn -- a native run and '
              'a stock run are comparable in distribution, never step by '
              'step.  Both the seed and the on/off flag are runtime inputs, '
              'so a stage that turns dropout off costs no recompile.')
    _register()
    for name, params in model_parameters.items():
        wd = weights_from_params(params, depth=depth)
        _MODEL_RUNNERS[name] = _make_runner(wd, depth, bool(dropout))
        # and the extra-MSA stack, which has its own 4 blocks under a
        # different prefix.  A model without it (a monomer preset) simply
        # does not get one and that stack stays on JAX.
        # ON by default; BC2_NATIVE_EXTRA=0 turns it off.
        #
        # GATED in ColabDesign2's harness (dev/native_block/gate_extra.py),
        # which asks the only question that can be asked of a design path:
        # does replacing the extra stack move the loss MORE than replacing
        # the main stack already does?  L=256, 4 iterations:
        #
        #   xla         loss 5.02381   con 3.92102   pae 0.73679
        #   main        loss 0.32554   con 0.06413   pae 0.36135
        #   main+extra  loss 0.35745   con 0.04338   pae 0.36176
        #
        # The main stack swap moves the loss 4.70; adding the extra stack
        # moves it a further 0.032 -- 147x less, and every other metric the
        # same way.  A design trajectory is chaotic (an end-to-end gradient
        # is not even comparable: the same path in two processes reads
        # cosine -0.80), so a relative bar like this is the honest test.
        if os.environ.get('BC2_NATIVE_EXTRA') != '0':
            try:
                ed = weights_from_params(params, depth=None, prefix=_EXTRA)
                _EXTRA_RUNNERS[name] = _make_extra_runner(ed, 4, bool(dropout))
            except KeyError:
                pass
        # and the template pair stack, two blocks under its own prefix
        if os.environ.get('BC2_NATIVE_TEMPLATE') == '1':
            try:
                td = weights_from_params(params, depth=None, prefix=_TPL)
                _TPL_RUNNERS[name] = _make_tpl_runner(td, 2, bool(dropout))
            except KeyError:
                pass
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
