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

import functools
import os
import sys

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


def _refuse_if_stale(lib: str) -> None:
    """A STALE .so IS WORSE THAN A MISSING ONE.

    ColabDesign2's backend.py says so and refuses; this loader did not, and
    picked the first file that existed.  That is not hypothetical: a
    thirteen-hour-old library served a whole afternoon of measurements
    while the fixes under test sat in the sources next to it, and the only
    symptom was that the numbers did not move.

    Sources beside the library are the evidence.  A vendored .so shipped
    without them cannot be checked and is left alone.
    """
    src_dir = os.path.dirname(os.path.abspath(lib))
    try:
        built = os.path.getmtime(lib)
        newer = [f for f in os.listdir(src_dir)
                 if f.endswith(('.cu', '.cuh'))
                 and os.path.getmtime(os.path.join(src_dir, f)) > built]
    except OSError:
        return
    if not newer:
        return
    raise RuntimeError(
        f'{lib} is older than {len(newer)} of its sources '
        f'({", ".join(sorted(newer)[:4])}{"..." if len(newer) > 4 else ""}).\n'
        f'It would run code that is not the code in the tree.  Rebuild it:\n'
        f'  cd {src_dir} && nvcc -O3 -std=c++17 \\\n'
        f'    -gencode arch=compute_80,code=sm_80 \\\n'
        f'    -gencode arch=compute_86,code=sm_86 \\\n'
        f'    -gencode arch=compute_90,code=sm_90 \\\n'
        f'    --compiler-options -fPIC -shared -I. \\\n'
        f"    -I$({sys.executable} -c 'import jax.ffi;print(jax.ffi.include_dir())') \\\n"
        f'    ffi.cu -lcublas -lcublasLt -o {os.path.basename(lib)}\n'
        f'BC2_NATIVE_LIB_STALE_OK=1 runs it anyway.')


_LIB = _find_lib()
if os.path.exists(_LIB) and not os.environ.get('BC2_NATIVE_LIB_STALE_OK'):
    _refuse_if_stale(_LIB)
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
_LIBH = None   # the loaded CDLL, for the plain C entry points beside the FFI

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


# The template embedding AS A WHOLE, the stack plus the four modules around
# it.  `single_template_embedding/` holds three of them; output_linear is one
# scope up, under template_embedding/ itself -- SingleTemplateEmbedding returns
# before it.
_TPE = 'alphafold/alphafold_iteration/evoformer/template_embedding'
_TPE_FLAT = {
    f'{_TPE}/single_template_embedding/query_embedding_norm':
        'query_embedding_norm',
    f'{_TPE}/single_template_embedding/template_pair_embedding_8':
        'template_pair_embedding_8',
    f'{_TPE}/single_template_embedding/output_layer_norm':
        'output_layer_norm',
    f'{_TPE}/output_linear': 'output_linear',
}


def tplembed_weights_from_params(params, out_dir=None):
    """Everything the native template embedding needs, in one directory.

    The stack's two blocks come out of layer_stack's leading axis exactly as
    weights_from_params writes them; the four modules around it carry no block
    index.  OUTPUT_LAYER_NORM IS NOT OPTIONAL: SingleTemplateEmbedding ends
    with it, after the stack and before the relu, and leaving it out puts the
    activation on the wrong scale for output_linear's weights -- rms 1.75
    against a reference 0.95, which is what the gate caught.
    """
    import hashlib
    import tempfile

    mods = {k: v for k, v in params.items() if k.startswith(_TPL)}
    if not mods:
        raise KeyError(f'no {_TPL}* in these params')
    depth = min(int(np.asarray(a).shape[0])
                for d in mods.values() for a in d.values())

    flat = {}
    for mod in sorted(mods):
        path = mod[len(_TPL) + 1:] if len(mod) > len(_TPL) else ''
        for pname in sorted(mods[mod]):
            w = np.asarray(mods[mod][pname], np.float32)
            for b in range(depth):
                key = f'param/block{b}/{path}/{pname}' if path else \
                      f'param/block{b}/{pname}'
                flat[key] = np.ascontiguousarray(w[b], dtype=np.float32)
    for scope, leaf in _TPE_FLAT.items():
        if scope not in params:
            raise KeyError(f'template embedding: no {scope} in these params')
        for pname, v in params[scope].items():
            flat[f'param/{leaf}/{pname}'] = np.ascontiguousarray(
                np.asarray(v, np.float32))

    h = hashlib.sha1()
    for k in sorted(flat):
        h.update(k.encode())
        h.update(flat[k].tobytes()[:4096])
    out = out_dir or os.path.join(tempfile.gettempdir(),
                                  f'bc2_tpe_{h.hexdigest()[:16]}')
    if os.path.exists(os.path.join(out, 'manifest.txt')):
        return out
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'manifest.txt'), 'w') as fh:
        for name in sorted(flat):
            arr = flat[name]
            fn = name.replace('/', '__') + '.f32'
            arr.tofile(os.path.join(out, fn))
            fh.write(f'{name} {fn} {arr.ndim} '
                     f'{" ".join(str(x) for x in arr.shape)}\n')
    print(f'native: template embedding weights ({len(flat)} arrays, '
          f'{depth} blocks) -> {out}')
    return out


_SM = 'alphafold/alphafold_iteration/structure_module'
_HEADS = ('distogram_head', 'predicted_aligned_error_head',
          'predicted_lddt_head', 'experimentally_resolved_head')
_IT = 'alphafold/alphafold_iteration'
# AF2's IPA builds its three scalar projections with use_bias=True so that the
# graph is the same for monomer and multimer, but no checkpoint carries those
# biases -- they were trained at zero.  Supplied here rather than special-cased
# in CUDA.
_IPA_ZERO_BIAS = ('fold_iteration/invariant_point_attention/q_scalar_projection',
                  'fold_iteration/invariant_point_attention/k_scalar_projection',
                  'fold_iteration/invariant_point_attention/v_scalar_projection')


def post_weights_from_params(params, out_dir=None):
    """The post-trunk weights -- structure module and the four heads.

    Unlike the Evoformer's these carry NO block index: AF2's structure module
    is recurrent and its eight iterations share one set of fold_iteration
    weights.  The residue-constant tables travel with the dump, from the same
    residue_constants the model itself uses, so there is no second copy of
    AF2's geometry on the CUDA side.
    """
    import hashlib
    import tempfile

    flat = {}
    for mod, d in params.items():
        if mod == _SM or mod.startswith(_SM + '/'):
            path = mod[len(_SM) + 1:]
        elif mod.startswith(_IT + '/') and mod[len(_IT) + 1:].split('/')[0] in _HEADS:
            path = mod[len(_IT) + 1:]
        else:
            continue
        for pname, v in d.items():
            key = f'param/{path}/{pname}' if path else f'param/{pname}'
            flat[key] = np.asarray(v, np.float32)
    if not flat:
        raise KeyError('no structure_module or head parameters in these params')

    for m in _IPA_ZERO_BIAS:
        wk, bk = f'param/{m}/weights', f'param/{m}/bias'
        if wk in flat and bk not in flat:
            # [C, H, QK] -- the bias spans every dim but the input one
            flat[bk] = np.zeros(int(np.prod(flat[wk].shape[1:])), np.float32)

    from .alphafold.common import residue_constants as rc
    flat['const/default_frame'] = np.asarray(
        rc.restype_rigid_group_default_frame, np.float32)
    flat['const/atom14_to_group'] = np.asarray(
        rc.restype_atom14_to_rigid_group, np.float32)
    flat['const/atom14_lit_pos'] = np.asarray(
        rc.restype_atom14_rigid_group_positions, np.float32)
    flat['const/atom14_mask'] = np.asarray(rc.restype_atom14_mask, np.float32)

    h = hashlib.sha1()
    for k in sorted(flat):
        h.update(k.encode())
        h.update(np.ascontiguousarray(flat[k]).tobytes()[:4096])
    out = out_dir or os.path.join(tempfile.gettempdir(),
                                  f'bc2_post_{h.hexdigest()[:16]}')
    if os.path.exists(os.path.join(out, 'manifest.txt')):
        return out
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'manifest.txt'), 'w') as fh:
        for name in sorted(flat):
            arr = np.ascontiguousarray(flat[name], dtype=np.float32)
            fn = name.replace('/', '__') + '.f32'
            arr.tofile(os.path.join(out, fn))
            fh.write(f'{name} {fn} {arr.ndim} '
                     f'{" ".join(str(x) for x in arr.shape)}\n')
    return out


def _make_post_runner(weights_dir, depth=8):
    """Structure module + all four heads as one custom call.

    NO PADDING, unlike the three Evoformer-family runners.  Those fail inside
    cuBLASLt at an unaligned length; this path goes through the IPA and the
    atom builder instead, and gates clean at N=211 -- a campaign's own length
    -- so padding it would cost an allocation to buy nothing.

    THE BACKWARD REPLAYS.  Nothing crosses from the forward but the inputs,
    which is what makes both calls re-entrant under JAX's async dispatch; see
    the note in ffi.cu.
    """
    dep = np.int64(depth)

    def raw(single, pair, pair_mask, aatype, seed):
        n = single.shape[0]
        f = jnp.float32
        return jax.ffi.ffi_call(
            'native_post_trunk_fwd',
            (jax.ShapeDtypeStruct((n, 14, 3), f),
             jax.ShapeDtypeStruct((n, n, 64), f),
             jax.ShapeDtypeStruct((n, n, 64), f),
             jax.ShapeDtypeStruct((n, 50), f),
             jax.ShapeDtypeStruct((n, 37), f)),
            vmap_method='sequential')(
                single, pair, pair_mask, aatype, seed,
                depth=dep, weights_dir=weights_dir)

    @jax.custom_vjp
    def post(single, pair, pair_mask, aatype, seed):
        return raw(single, pair, pair_mask, aatype, seed)

    def fwd(single, pair, pair_mask, aatype, seed):
        # THE SEED GOES IN THE RESIDUAL.  The backward replays the forward,
        # and a replay that redrew its dropout masks would hand back the
        # gradient of a model that was never run -- with nothing wrong in any
        # output to show for it.
        return (raw(single, pair, pair_mask, aatype, seed),
                (single, pair, pair_mask, aatype, seed))

    def bwd(res, g):
        single, pair, pair_mask, aatype, seed = res
        n = single.shape[0]
        dsingle, dpair = jax.ffi.ffi_call(
            'native_post_trunk_bwd',
            (jax.ShapeDtypeStruct((n, 384), jnp.float32),
             jax.ShapeDtypeStruct((n, n, 128), jnp.float32)),
            vmap_method='sequential')(
                single, pair, pair_mask, aatype,
                *[x.astype(jnp.float32) for x in g], seed,
                depth=dep, weights_dir=weights_dir)
        return (dsingle, dpair, jnp.zeros_like(pair_mask),
                jnp.zeros_like(aatype),
                np.zeros(seed.shape, dtype=jax.dtypes.float0))

    post.defvjp(fwd, bwd)

    def run(single, pair, pair_mask, aatype, seed=None):
        if seed is None:
            seed = jnp.zeros((2,), jnp.uint32)
        out = post(single.astype(jnp.float32), pair.astype(jnp.float32),
                   pair_mask.astype(jnp.float32), aatype.astype(jnp.float32),
                   seed.astype(jnp.uint32).reshape((2,)))
        return dict(zip(('atom14', 'distogram', 'pae', 'plddt', 'expres'), out))

    return run


_EMB = 'alphafold/alphafold_iteration/evoformer'
# single_activations is NOT the embedder's -- AF2 applies it to the
# EVOFORMER's msa[0], after the trunk -- but it sits under the same scope and
# the native embedder is what holds it (embed.cuh::singleFromTrunk).  The
# per-piece path never needed it, because JAX did that projection; the
# one-call design step does, and without it the pointer is null.
_EMB_LEAVES = ('preprocess_1d', 'preprocess_msa', 'left_single',
               'right_single', 'position_activations', 'single_activations',
               'template_single_embedding', 'template_projection',
               'prev_pos_linear', 'prev_msa_first_row_norm', 'prev_pair_norm')


def embed_weights_from_params(params, out_dir=None):
    """The embedder's five arrays, from a live haiku param tree.

    THE SCOPE IS THE TRAP: haiku scopes a module created inside a METHOD
    under that method's name and the checkpoint keeps it, so
    position_activations lives at `evoformer/~_relative_encoding/
    position_activations` while its four siblings sit directly under
    `evoformer/`.  Matching on the LEAF rather than the full path is what
    makes both land.
    """
    import hashlib
    import tempfile

    flat = {}
    for mod, d in params.items():
        if not mod.startswith(_EMB):
            continue
        leaf = mod.split('/')[-1]
        if leaf not in _EMB_LEAVES:
            continue
        for pname, v in d.items():
            flat[f'param/{leaf}/{pname}'] = np.asarray(v, np.float32)
    if len(flat) != 2 * len(_EMB_LEAVES):
        raise KeyError(f'embedder: found {len(flat)} arrays, '
                       f'expected {2 * len(_EMB_LEAVES)}')
    h = hashlib.sha1()
    for k in sorted(flat):
        h.update(k.encode())
        h.update(np.ascontiguousarray(flat[k]).tobytes()[:4096])
    out = out_dir or os.path.join(tempfile.gettempdir(),
                                  f'bc2_emb_{h.hexdigest()[:16]}')
    if os.path.exists(os.path.join(out, 'manifest.txt')):
        return out
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'manifest.txt'), 'w') as fh:
        for name in sorted(flat):
            arr = np.ascontiguousarray(flat[name], dtype=np.float32)
            fn = name.replace('/', '__') + '.f32'
            arr.tofile(os.path.join(out, fn))
            fh.write(f'{name} {fn} {arr.ndim} '
                     f'{" ".join(str(x) for x in arr.shape)}\n')
    return out


def _make_embed_runner(weights_dir):
    """The four input Linears and the relative-position gather, in one call.

    NO REPLAY AND NO SAVE SET -- the only piece of this chain that needs
    neither.  The backward is a pure function of the cotangents, target_feat
    and the weights, so the residual carries one tensor.
    """

    def raw(tf, mf, ri, ai, ei, si):
        n, s = tf.shape[0], mf.shape[0]
        f = jnp.float32
        return jax.ffi.ffi_call(
            'native_embed_fwd',
            (jax.ShapeDtypeStruct((s, n, 256), f),
             jax.ShapeDtypeStruct((n, n, 128), f)),
            vmap_method='sequential')(
                tf, mf, ri, ai, ei, si, weights_dir=weights_dir)

    @jax.custom_vjp
    def emb(tf, mf, ri, ai, ei, si):
        return raw(tf, mf, ri, ai, ei, si)

    def fwd(tf, mf, ri, ai, ei, si):
        return raw(tf, mf, ri, ai, ei, si), (tf, ri, ai, ei, si)

    def bwd(res, g):
        tf, ri, ai, ei, si = res
        dmsa, dpair = [x.astype(jnp.float32) for x in g]
        s, n = dmsa.shape[0], dmsa.shape[1]
        dtf, dmf = jax.ffi.ffi_call(
            'native_embed_bwd',
            (jax.ShapeDtypeStruct((n, 21), jnp.float32),
             jax.ShapeDtypeStruct((s, n, 49), jnp.float32)),
            vmap_method='sequential')(
                dmsa, dpair, tf, weights_dir=weights_dir)
        # the relative-position term is a GATHER from the four id features,
        # so they carry no gradient; they are arrays, so they cannot be
        # nondiff_argnums and a None here would be an error rather than a
        # shortcut
        return (dtf, dmf, jnp.zeros_like(ri), jnp.zeros_like(ai),
                jnp.zeros_like(ei), jnp.zeros_like(si))

    emb.defvjp(fwd, bwd)

    def run(tf, mf, ri, ai, ei, si):
        return emb(tf.astype(jnp.float32), mf.astype(jnp.float32),
                   ri.astype(jnp.float32), ai.astype(jnp.float32),
                   ei.astype(jnp.float32), si.astype(jnp.float32))

    return run


def template_act_const(model_parameters, template_batch, multichain_mask_2d,
                       query_shape, dtype, cache={}):
    """The sum of SingleTemplateEmbedding's eight CONSTANT feature projections.

    Those eight are built from the template alone -- its aatype, atom
    positions and atom mask -- and designed residues are masked to aatype 21
    whatever they currently are, so the sum cannot change while a trajectory
    runs.  Computing it once and handing it back removes the backbone affine,
    the rigid inverse, the pseudo-beta, the distogram, the one-hots and eight
    Linears from EVERY STEP, not just from the compile.

    Harvested through the model's own module, with __const_only__, rather
    than restated here: that is what keeps the eight Linears in their haiku
    scope and so bound to their own weights.  Verified bit-identical against
    the shipped path, max|d| 0.000e+00.

    Cached on the template arrays' contents.  They are invariant WITHIN a
    trajectory, not across campaigns -- induced fit, a multitarget swap or a
    redesign changes them -- so the key is the data, never the call site.
    """
    import hashlib

    import haiku as hk
    from .alphafold.model import modules_multimer as mm
    from .alphafold.model import config as af_config
    from .alphafold.model import prng

    h = hashlib.sha1()
    for k in sorted(template_batch):
        h.update(k.encode())
        h.update(np.ascontiguousarray(np.asarray(template_batch[k])).tobytes())
    h.update(np.ascontiguousarray(np.asarray(multichain_mask_2d)).tobytes())
    h.update(str(query_shape).encode())
    key = h.hexdigest()
    if key in cache:
        return cache[key]

    cfg = af_config.model_config('model_1_multimer_v3').model
    tc = cfg.embeddings_and_evoformer.template
    gc = cfg.global_config
    # strip only as far as template_embedding/: the standalone transform
    # instantiates SingleTemplateEmbedding, which supplies the next level of
    # the name itself.  Stripping that too asks haiku for
    # 'template_pair_embedding_0' where it created
    # 'single_template_embedding/template_pair_embedding_0'.
    scope = 'alphafold/alphafold_iteration/evoformer/template_embedding/'
    params = {k[len(scope):]: v for k, v in model_parameters.items()
              if k.startswith(scope)}

    def fn(q, tb, mc):
        emb = mm.SingleTemplateEmbedding(tc, gc)
        return emb(q, {**tb, '__const_only__': True},
                   jnp.ones(q.shape[:2], q.dtype), mc, False,
                   prng.SafeKey(jax.random.PRNGKey(0)))

    t = hk.transform(fn)
    q = jnp.zeros(query_shape, dtype)      # unused under __const_only__
    out = jax.jit(t.apply)(params, jax.random.PRNGKey(0), q,
                           template_batch, multichain_mask_2d)
    cache[key] = out
    return out


_EMB_RUNNERS = {}
_POST_RUNNERS = {}


def available() -> bool:
    return why_unavailable() == ''


def _register():
    global _registered
    if _registered:
        return
    import ctypes
    # THE CHUNK THRESHOLD SCALES WITH THE CARD TOO, and it has to be in the
    # environment BEFORE the library reads it -- nbChunkOn() caches it in a
    # function-local static on first use, so a later setenv is ignored.  480
    # is the measured A10 value in PADDED residues (the kernels pad to a
    # multiple of 32, so a 457 complex arrives as 480); a card with more room
    # should start chunking later, by the same sqrt as the length limit.
    os.environ.setdefault('NB_CHUNK_N', str(_chunk_threshold()))
    # SAME DISCIPLINE FOR THE STRIDE: strideK() caches it in a function-local
    # static, so it has to be in the environment before the library loads.
    os.environ.setdefault('NB_STRIDE_N', str(_stride_threshold()))
    # SAY IT OUT LOUD.  The stride is a trade of time for memory and it is
    # read in two places (here and stack.cuh); a silent one is exactly the
    # kind of setting that gets blamed on something else.
    _k = os.environ.get('NB_STRIDE')
    if _k:
        print(f'native: stride remat FORCED to NB_STRIDE={_k} at every length '
              f'-- {_save_slots(48, 0)} of 48 save slots', file=sys.stderr)
    global _LIBH
    lib = _LIBH = ctypes.CDLL(_LIB)
    # THE EXPORTED SYMBOL IS THE C++ HANDLER'S NAME, not a snake_case
    # convention: XLA_FFI_DEFINE_HANDLER_SYMBOL(NativeEvoformerFwd, ...)
    # exports exactly `NativeEvoformerFwd`.  Checked with nm, not guessed.
    for target, symbol in (('native_evoformer_fwd', 'NativeEvoformerFwd'),
                           ('native_evoformer_bwd', 'NativeEvoformerBwd'),
                           ('template_pair_stack_fwd', 'TemplatePairStackFwd'),
                           ('template_pair_stack_bwd', 'TemplatePairStackBwd'),
                           ('native_post_trunk_fwd', 'StructureHeadsFwd'),
                           ('native_post_trunk_bwd', 'StructureHeadsBwd'),
                           ('native_embed_fwd', 'EmbedderFwd'),
                           ('native_embed_bwd', 'EmbedderBwd'),
                           ('native_tplembed_fwd', 'TemplateEmbeddingFwd'),
                           ('native_tplembed_bwd', 'TemplateEmbeddingBwd')):
        jax.ffi.register_ffi_target(
            target, jax.ffi.pycapsule(getattr(lib, symbol)), platform='CUDA')
    _registered = True


_MODEL_RUNNERS = {}
_EXTRA_RUNNERS = {}
_TPL_RUNNERS = {}
_TPE_RUNNERS = {}


def supported_length(n: int) -> bool:
    """Can the kernels take this complex length?

    Asked BEFORE the hook draws its seed.  Drawing consumes haiku's SafeKey,
    and a key that has been used cannot be handed to the JAX fallback -- it
    raises 'Random key has been used previously'.  So support is decided
    first and the key is touched only on the path that will use it.

    Length is also where the kernels STOP.  They hold a save-set per block
    for the backward replay where XLA rematerialises, so the native path
    needs about 15 GB at 427 residues where JAX peaks at 1.3 GB -- the same
    design choice that makes it 1.6-1.8x faster per step.  Measured on a
    23.7 GB A10.  THE LIMIT DEPENDS ON THE LOSS SET, which is why it is 480
    and not the 544 a lighter one measures.  plddt + compactness + entropy are
    all binder-only, so no gradient crosses the interface and the backward is
    half the size; under them 537 runs.  Add interface_contacts -- which is
    what a binder campaign actually optimises -- and the complex's distogram
    comes into the backward, the step roughly doubles, and 487 dies in
    ipa.cuh:558.  Measured with it: 457 runs at 5988 ms, 487 did not.

    487-512 RUN NOW, because the three big pair modules chunk themselves
    above 457 residues (nbChunkOn in common.cuh).  487 was a 16320 ms JAX
    failover and is 7078 ms native; 512 is 7279.  The ceiling is 512 because
    522 and 532 die in triangle attention's own whole-module buffers, which
    chunking does not reach.

    A limit that is too HIGH is a crash in the middle of a campaign, which is
    the thing this exists to prevent; too low is a slower run.  So it is set
    from the heavier, realistic case.  Raise it with BC2_NATIVE_MAX_N if your
    losses are light or your card is bigger -- native/length_ladder.py is the
    thing to re-measure with.

    Dying mid-campaign is the worst outcome available, so above the limit
    this declines and JAX runs the span: slower, but it finishes.  The limit
    is a measured length, not a memory query, because the decision is made at
    TRACE time when XLA has not yet allocated and a free-VRAM reading would
    not mean anything.  BC2_NATIVE_MAX_N overrides it -- raise it on a bigger
    card, where the ceiling is proportionally further out.
    """
    limit = _length_limit()
    if n <= limit:
        return True
    if n not in _DECLINED_LENGTHS:
        _DECLINED_LENGTHS.add(n)
        print(f'native: {n} residues is past the kernels\' measured limit of '
              f'{limit}; running this span on JAX instead (slower, but it '
              f'does not run out of memory mid-step). BC2_NATIVE_MAX_N raises '
              f'the limit.', flush=True)
    return False


# --------------------------------------------------------------- the limit
# 512 WAS MEASURED ON ONE CARD, and a hardcoded 512 is wrong for every other
# one.  The native path's footprint is dominated by per-block save sets that
# are N*N*C -- QUADRATIC in length -- so the ceiling moves with the square
# root of the memory budget, and an 80 GB card would otherwise decline to JAX
# at 513 residues and pay 2.6x for nothing.
#
# WHY TOTAL AND NOT FREE.  This is asked at TRACE time, before XLA has
# allocated anything, so a free-VRAM reading describes a moment that has not
# happened yet.  `bytes_limit` is the device budget XLA will work inside --
# stable, known up front, and proportional to the card.
#
# WHAT IS NOT MODELLED: the part of the footprint that is NOT quadratic --
# weights, the cuBLASLt workspace, XLA's own pool.  Ignoring it UNDERSTATES
# the headroom on a card bigger than the A10 (safe) and OVERSTATES it on a
# smaller one (not), so cards below the measured point take an extra margin.
#
# ONLY THE A10 POINT IS MEASURED.  Everything else is this extrapolation.
# native/length_ladder.py is how to replace a guess with a number, and
# BC2_NATIVE_MAX_N overrides it outright in either direction.
_A10_BYTES = 17763631104       # jax bytes_limit on the 23.7 GB A10
# 608 since the trace-time prealloc.  The history of this constant is the
# history of what was actually binding:
#   512  before stride remat
#   544  stride remat shrank the save-set, but 552 still failed
#   608  the stacks stopped racing XLA for the first allocation
# 552, 567, 577, 587, 600 and 608 all run, in order, with no gaps; 617, 632
# and 647 die in tplblock/tplstack, the template pair stack, which is the
# next mandatory allocation still made lazily.  See
# dev/native_block/ALLOC_ORDER.md for why it cannot simply be preallocated
# too (it was tried, and it cost 577 and 587).
_A10_MAX_N = 608
# Padded N above which the save-set goes stride-remat.  It is the OLD
# ceiling on purpose: below it memory is not the binding constraint and the
# stride is 21% of a step for nothing.
_A10_STRIDE_N = 512


def _scaled(a10_value: int) -> int:
    """An A10 measurement moved to this card by the sqrt of its budget."""
    budget = _device_budget()
    if not budget:
        return a10_value
    scale = (budget / _A10_BYTES) ** 0.5
    if scale < 1.0:
        scale *= 0.9                   # the unmodelled fixed cost bites here
    return max(128, min(2048, int(a10_value * scale) // 32 * 32))


def _device_budget() -> int:
    """Bytes XLA will work inside on this device, or 0 if it cannot be read."""
    try:
        import jax
        return int(jax.local_devices()[0].memory_stats()['bytes_limit'])
    except Exception:
        return 0


@functools.lru_cache(maxsize=1)
def _length_limit() -> int:
    env = os.environ.get('BC2_NATIVE_MAX_N')
    if env:
        return int(env)
    return _scaled(_A10_MAX_N)


@functools.lru_cache(maxsize=1)
def _stride_threshold() -> int:
    """Padded N above which the save-set keeps every 4th block, not every one."""
    return _scaled(_A10_STRIDE_N)


@functools.lru_cache(maxsize=1)
def _chunk_threshold() -> int:
    """Padded N above which the pair modules chunk themselves."""
    return _scaled(480)


_DECLINED_LENGTHS: set = set()


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
        # PAD TO 32, as the trunk and the extra-MSA stack do.  The kernels
        # are only gated on 32-aligned lengths and an unaligned one fails
        # inside cuBLASLt -- which is not hypothetical: without this a PD-L1
        # campaign died at its real length of 211 with
        #     trimul_body.cuh:1014 cublas: 14
        # while every measurement here passed, because they were all run at
        # 224.  A benchmark length that happens to be aligned will never find
        # this; a campaign finds it immediately.
        n = int(act.shape[0])
        pad = (-n) % 32
        a = act.astype(jnp.float32)
        m = mask.astype(jnp.float32)
        if pad:
            a = jnp.pad(a, ((0, pad), (0, pad), (0, 0)))
            m = jnp.pad(m, ((0, pad), (0, pad)))
        out = stack(a, m, seed.astype(jnp.uint32).reshape((2,)))
        if pad:
            out = out[:n, :n]
        return out.astype(act.dtype)

    return run


def _make_tplembed_runner(weights_dir, depth=2, dropout=False):
    """The WHOLE template embedding, not just its stack.

    Takes the query pair activation and the precomputed constant eight, and
    returns the embedding output_linear produces.  That is query_embedding_norm,
    template_pair_embedding_8, the two stack blocks, output_layer_norm, the
    relu and output_linear -- everything SingleTemplateEmbedding and
    TemplateEmbedding do between them, in one custom call.

    ONE TEMPLATE ONLY.  TemplateEmbedding scans over templates and divides by
    their count; BindCraft emits template_mask jnp.ones((1,)), so the scan and
    the mean are identities this does not reproduce.  Asserted by the caller,
    not here, because here the count is no longer visible.

    act_const takes no cotangent -- it is a constant by construction, which is
    the entire reason it is hoisted out of the step.
    """
    drop = np.int64(1 if dropout else 0)
    CQ, CT = 128, 64

    def raw(q, ac, pm, seed):
        n = q.shape[0]
        return jax.ffi.ffi_call(
            'native_tplembed_fwd',
            (jax.ShapeDtypeStruct((n, n, CQ), jnp.float32),
             jax.ShapeDtypeStruct((n * n * CT * 5 * depth,), jnp.bfloat16)),
            vmap_method='sequential')(
                q, ac, pm, seed, depth=np.int64(depth), dropout=drop,
                weights_dir=weights_dir)

    @jax.custom_vjp
    def embed(q, ac, pm, seed):
        return raw(q, ac, pm, seed)[0]

    def fwd(q, ac, pm, seed):
        o, save = raw(q, ac, pm, seed)
        # q AND ac ARE KEPT.  The backward replays the forward from them --
        # everything it would otherwise read is member state in the C++
        # object, and a design step runs forwards for every state and every
        # recycle before any backward, so that state is never the right one.
        return o, (q, ac, pm, save, seed)

    def bwd(res, g):
        q, ac, pm, save, seed = res
        n = g.shape[0]
        dq = jax.ffi.ffi_call(
            'native_tplembed_bwd',
            jax.ShapeDtypeStruct((n, n, CQ), jnp.float32),
            vmap_method='sequential')(
                g.astype(jnp.float32), q, ac, pm, save, seed,
                depth=np.int64(depth), dropout=drop,
                weights_dir=weights_dir)
        return (dq, jnp.zeros((n, n, CT), jnp.float32), jnp.zeros_like(pm),
                np.zeros(seed.shape, dtype=jax.dtypes.float0))

    embed.defvjp(fwd, bwd)

    def run(query, act_const, pair_mask, n_block=depth, seed=None):
        if int(n_block) != depth:
            raise ValueError(f'template embedding built for depth {depth}, '
                             f'asked for {int(n_block)}')
        if seed is None:
            seed = jnp.zeros((2,), jnp.uint32)
        # the same 32-alignment the stack needs, for the same cuBLASLt reason
        n = int(query.shape[0])
        pad = (-n) % 32
        q = query.astype(jnp.float32)
        ac = act_const.astype(jnp.float32)
        pm = pair_mask.astype(jnp.float32)
        if pad:
            q = jnp.pad(q, ((0, pad), (0, pad), (0, 0)))
            ac = jnp.pad(ac, ((0, pad), (0, pad), (0, 0)))
            pm = jnp.pad(pm, ((0, pad), (0, pad)))
        out = embed(q, ac, pm, seed.astype(jnp.uint32).reshape((2,)))
        if pad:
            out = out[:n, :n]
        return out.astype(query.dtype)

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


def _stride_k(depth: int, n: int) -> int:
    """How many blocks the save-set skips.  Mirrors Stack::strideK."""
    env = os.environ.get('NB_STRIDE')
    if env:
        k = int(env)
    else:
        k = 4 if n > int(os.environ.get('NB_STRIDE_N') or 512) else 1
    return max(1, min(k, depth))


def _save_slots(depth: int, n: int) -> int:
    """How many blocks' worth of input the save-set holds.

    THIS IS A DUPLICATE of Stack::saveSlots in dev/native_block/stack.cuh and
    the two must agree exactly.  The save-set is an FFI buffer: XLA allocates
    it from the shape declared below and the C++ slices it.  If this says
    `depth` and the C++ says fewer, the tail is merely wasted; if this says
    fewer than the C++ slices, the backward reads past the end of someone
    else's buffer.  Neither is a compile error and neither raises -- the
    first costs memory, the second is a wrong gradient.

    `n` is `pair.shape[0]`, which is the same number the FFI reads out of
    `pair.dimensions()[0]`; the policy keys on it so that both sides reach
    the same answer without either being told.

    Stride remat keeps every k-th block's input and rebuilds the rest of
    each segment during the backward, into a scratch of k-1 slots: the
    save-set goes from `depth` slots to `ceil(depth/k) + k - 1` -- 15 of 48
    at k = 4 -- at the price of a second forward over every non-checkpoint
    block.  Measured on the A10: a 215 step costs 21% more with it on, and
    537, 545 and 587 residues run with it where 512 was the ceiling.
    """
    k = _stride_k(depth, n)
    return depth if k <= 1 else (depth + k - 1) // k + k - 1


_PREALLOCED: set = set()


def _prealloc_stack(n, s_, depth, weights_dir, pair_only, dropout):
    """Build this stack's buffers NOW, while the card is still empty.

    THE ORDER OF ALLOCATION IS THE CEILING, not the amount.  A design step
    runs two predictions -- binder_alone and the complex -- and whichever
    traces into execution first allocates first.  Measured on the A10 with
    BLK_MEM: at 552 residues the small stacks go first, XLA then grows to
    17.83 GB in use, the complex stack takes its 4.30 GB out of the 5.86
    remaining and the structure module dies with 1.47 GB.  At 587 -- LONGER
    -- the complex stack goes first, takes 4.79 GB with 13.64 GB free, and
    the whole step fits.  Nothing about 552 is bigger; the order decided it.

    This runs at TRACE time, when XLA has allocated nothing, so the stacks
    are built before there is anything to lose a race to.  It allocates the
    exact buffers the step will then run on -- `native_stack_prealloc` goes
    through the same `get()` cache the FFI call uses -- so unlike the
    trace-time ballast this replaces, there is nothing held and nothing to
    release at the right moment.

    A failure here is not fatal: the stack is built on first call as it
    always was, in the order that used to decide things.
    """
    if os.environ.get('NB_NO_PREALLOC'):
        return
    key = (int(n), int(s_), int(depth), weights_dir, bool(pair_only),
           bool(dropout))
    if key in _PREALLOCED:
        return
    _PREALLOCED.add(key)
    _register()
    fn = getattr(_LIBH, 'native_stack_prealloc', None)
    if fn is None:            # an old library: say so, do not pretend
        print('native: this libnative_evoformer.so has no '
              'native_stack_prealloc; stacks will be built in call order',
              file=sys.stderr)
        return
    import ctypes
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int]
    fn(int(n), int(s_), int(depth), str(weights_dir).encode(),
       1 if pair_only else 0, 1, 1 if dropout else 0)


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
        # TRACE TIME, not call time -- see _prealloc_stack.
        _prealloc_stack(pair.shape[0], msa.shape[0], depth, weights_dir,
                        bool(pair_only), bool(dropout))
        out = jax.ffi.ffi_call(
            'native_evoformer_fwd',
            (jax.ShapeDtypeStruct(pair.shape, jnp.float32),
             jax.ShapeDtypeStruct(msa.shape, jnp.float32),
             jax.ShapeDtypeStruct((_save_slots(depth, pair.shape[0]),)
                                  + pair.shape, jnp.bfloat16),
             jax.ShapeDtypeStruct((_save_slots(depth, pair.shape[0]),)
                                  + msa.shape, jnp.float32)),
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


# ONE SWITCH, NOT FIVE.  `backend` (auto|native|jax, in the
# settings file or `--set`) is the user-facing control: it turns the whole
# native path on or off, and nothing below is meant to be part of a normal
# run.  BC2_NATIVE_DISABLE is a DEBUGGING escape hatch -- a comma separated
# list of pieces to leave on JAX while the rest stay native -- for bisecting
# a bad campaign, not a performance dial.  There is no speed reason to use
# it: all-on is the fastest configuration measured (native/flag_sweep.py).
_PIECES = ('extra', 'template', 'tplembed', 'post', 'embed')
# the five that used to be separate.  They are REFUSED rather than ignored:
# someone with BC2_NATIVE_POST=0 in a script should learn that it moved, not
# silently get the opposite of what they asked for.
_RETIRED = {'BC2_NATIVE_EXTRA': 'extra', 'BC2_NATIVE_TEMPLATE': 'template',
            'BC2_NATIVE_TPLEMBED': 'tplembed', 'BC2_NATIVE_POST': 'post',
            'BC2_NATIVE_EMBED': 'embed'}


def _disabled_pieces():
    for old, piece in _RETIRED.items():
        if os.environ.get(old) is not None:
            raise ValueError(
                f'{old} no longer exists.  To turn the whole native path '
                f'off use backend=jax (or BC2_NATIVE=0); to leave '
                f'just this piece on JAX for debugging use '
                f'BC2_NATIVE_DISABLE={piece}.')
    raw = os.environ.get('BC2_NATIVE_DISABLE', '')
    names = {n.strip().lower() for n in raw.split(',') if n.strip()}
    unknown = names - set(_PIECES)
    if unknown:
        raise ValueError(
            f'BC2_NATIVE_DISABLE: unknown piece(s) {sorted(unknown)}; '
            f'known are {list(_PIECES)}')
    return names


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
    # BOTH STAY INSTALLED.  The whole-embedding runner returns before
    # SingleTemplateEmbedding ever runs, so there is no risk of the stack
    # executing twice -- and nulling the stack runner here was a real bug:
    # the embedding declines whenever its preconditions do not hold (more
    # than one template, no harvested constant), and the state that declines
    # then fell back to the XLA stack while the other used the native one.
    # Measured AFTER this change: it did NOT account for the 14x design
    # gradient difference that prompted it, so do not read it as the cause.
    # It is still the right behaviour -- there is no reason to disable a
    # correct runner -- but the 14x is something else.
    modules_multimer.NATIVE_TEMPLATE_EMBED = _TPE_RUNNERS.get(name)
    modules_multimer.NATIVE_TEMPLATE_STACK = _TPL_RUNNERS.get(name)
    modules_multimer.NATIVE_POST_TRUNK = _POST_RUNNERS.get(name)
    modules_multimer.NATIVE_EMBEDDER = _EMB_RUNNERS.get(name)


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
    # read once, and REFUSE a retired flag here rather than per model
    _off = _disabled_pieces()
    if _off:
        print(f'native: DEBUG -- leaving {sorted(_off)} on JAX '
              f'(BC2_NATIVE_DISABLE).  all-on is the fastest configuration; '
              f'this is for bisecting, not for speed.')
    for name, params in model_parameters.items():
        wd = weights_from_params(params, depth=depth)
        _MODEL_RUNNERS[name] = _make_runner(wd, depth, bool(dropout))
        # and the extra-MSA stack, which has its own 4 blocks under a
        # different prefix.  A model without it (a monomer preset) simply
        # does not get one and that stack stays on JAX.
        # On with the native path; BC2_NATIVE_DISABLE=extra for debugging.
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
        if 'extra' not in _off:
            try:
                ed = weights_from_params(params, depth=None, prefix=_EXTRA)
                _EXTRA_RUNNERS[name] = _make_extra_runner(ed, 4, bool(dropout))
            except KeyError:
                pass
        # and the template pair stack, two blocks under its own prefix.
        # On with the native path; BC2_NATIVE_DISABLE=template for debugging.
        #
        # Judged on the bar gate_extra.py sets and the extra-MSA stack was
        # enabled on: does replacing this move the loss MORE than replacing
        # the main stack already does?  L=224, dropout off, one process each:
        #
        #   all XLA                        loss 3.987596
        #   native trunk, templates JAX    loss 4.006162 / 4.007646 / 4.007384
        #   native trunk + native templates     4.015828
        #
        # The same configuration in two processes differs by 2.6e-04, so that
        # is the floor.  The main stack swap moves the loss 1.85e-02; adding
        # the template stack moves it a further 8.7e-03 -- about half, and it
        # also has what the extra-MSA stack never had: a module-level gate
        # against a JAX oracle on the real checkpoint (tplblock, fwd
        # 0.999952 / bwd 0.997982).
        if 'template' not in _off:
            try:
                td = weights_from_params(params, depth=None, prefix=_TPL)
                _TPL_RUNNERS[name] = _make_tpl_runner(td, 2, bool(dropout))
            except KeyError:
                pass
        # the WHOLE template embedding rather than just its two blocks.  On
        # with the native path; BC2_NATIVE_DISABLE=tplembed for debugging.
        # It supersedes the
        # stack runner above WITHOUT disabling it: this one returns before
        # SingleTemplateEmbedding runs, so the stack cannot execute twice,
        # and leaving the stack runner installed keeps any state that
        # declines the embedding -- more than one template, no harvested
        # constant -- on the native stack rather than dropping it to XLA.
        #
        # ON BY REQUEST, NOT BY THE MEASUREMENT.  Compile goes
        # 6.4 -> 4.2 s and the step does not move.  Paired hPDL1 campaigns
        # could not separate it from the JAX path: nothing was accepted in
        # ANY arm, including pure JAX, so three trajectories had no
        # resolution; on stage depth it was 2-1 to off.  See the reference
        # table for the numbers.
        if 'tplembed' not in _off:
            try:
                ted = tplembed_weights_from_params(params)
                _TPE_RUNNERS[name] = _make_tplembed_runner(ted, 2,
                                                           bool(dropout))
                print(f'native: template EMBEDDING INSTALLED for {name} '
                      f'-> {ted}')
            except KeyError as e:
                print(f'native: template embedding NOT installed for '
                      f'{name}: {e}')
        # and the post-trunk step: the structure module and the four heads,
        # and the post-trunk step: the structure module and the four heads in
        # one call.  On with the native path; BC2_NATIVE_DISABLE=post to debug.
        #
        # It carries AF2's two dropout sites (folding_multimer.FoldIteration
        # lines 428 and 449, rate 0.1 elementwise on the act after each
        # residual).  It was guarded until it did: without them the path was
        # not a faster structure module but a different one, and use_dropout
        # is a runtime value so the hook cannot decline at trace time.
        #
        # Judged on the bar the extra-MSA and template stacks were enabled on
        # -- does adding it move the model MORE than replacing the main stack
        # already does?  native/post_trunk_gate.py, N=179, dropout off:
        #
        #   xla        plddt 0.450951  |grad| 10.6216
        #   main       plddt 0.451231  |grad|  5.4268
        #   main+post  plddt 0.451212  |grad|  4.4129
        #   main+post  plddt 0.451212  |grad|  4.4118   <- the floor
        #   main       plddt 0.451235  |grad|  5.4032   <- the floor
        #
        # The main stack swap moves pLDDT 2.8e-04 and the gradient norm
        # 48.9%; adding this moves them a further 1.9e-05 and 18.7%, 15x and
        # 5x less.  The gradient move is REAL -- 18.7% against a 0.025%
        # floor -- just smaller than one the default already makes.  And it
        # takes the design graph's compile from 17.0 s to 7.7 s at 179
        # tokens for 1.1% more time a step.
        if 'post' not in _off:
            # LOUD, not silent.  The other two stacks swallow a KeyError
            # because a preset that lacks them is a normal thing; asking for
            # this one explicitly and silently not getting it is not, and a
            # comparison run against a hook that never installed would read
            # as 'no difference' for the wrong reason.
            try:
                pd = post_weights_from_params(params)
                _POST_RUNNERS[name] = _make_post_runner(pd, 8)
                print(f'native: post-trunk INSTALLED for {name} -> {pd}')
            except KeyError as e:
                print(f'native: post-trunk NOT installed for {name}: {e}')
        # and the embedder: the four input Linears and the relative-position
        # gather.  On with the native path; BC2_NATIVE_DISABLE=embed to debug.
        #
        # Off because it is not yet judged.  The module gate is exact against
        # AF2 (~1e-07 on both outputs and both gradients at N=179, four
        # Linears and a gather being easy to get right), but TWO THINGS
        # CHANGE in the model that the module gate cannot see: the block runs
        # in bfloat16 there and this runs in f32, and the recycle terms that
        # sit between the Linears and the relative encoding are added
        # afterwards in JAX instead of in between.  Both are sound -- the sum
        # is the same and f32 is the more accurate side -- but 'sound' is an
        # argument, and the bar here is a measurement.
        if 'embed' not in _off:
            try:
                ed2 = embed_weights_from_params(params)
                _EMB_RUNNERS[name] = _make_embed_runner(ed2)
                print(f'native: embedder INSTALLED for {name} -> {ed2}')
            except KeyError as e:
                print(f'native: embedder NOT installed for {name}: {e}')
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
