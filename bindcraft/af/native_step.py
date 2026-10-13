"""The whole design step as ONE custom call, with its gradient.

This is the thing the glue port exists for.  An XLA custom call takes
STATIC shapes, so ten calls padded to 32 in JAX leave the graph specialised
per bucket and the glue between them is still XLA's.  One call whose
buffers are sized NMAX and whose LENGTH IS A RUNTIME INPUT collapses every
bucket into one graph -- and a loss is still written in Python, because the
loss is the only thing left on the JAX side.

NOT YET WIRED IN.  design_step.py still runs the JAX pipeline; this module
is the adapter that will replace its body once dev/native_block's
ds_compare.py and ds_grad.py both read green against a real design step.
Nothing imports it yet, deliberately: a half-gated native step that runs is
worse than one that does not, because the campaign still finishes and the
binders are merely worse.

The kernels live in ColabDesign2/dev/native_block (design_step.cuh); this
file only knows the call's signature and the nine-array contract in
boundary.py.
"""
from __future__ import annotations

import functools
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

from .alphafold.common import residue_constants as RC
from .design_step import DesignStepOutputs

F32, I32 = jnp.float32, jnp.int32
DBINS = 64


@functools.lru_cache(maxsize=1)
def _table_arrays():
    """AF2's own constant tables, as NUMPY arrays.

    They travel as INPUTS rather than being baked into the kernel: a second
    copy of residue_constants behind a CUDA rebuild is a second thing to
    keep in step with the first.
    """
    chi_idx = np.zeros((22, 4, 4), np.int32)
    for r, letter in enumerate(RC.restypes):
        name = RC.restype_1to3[letter]
        for c, atoms in enumerate(RC.chi_angles_atoms[name]):
            chi_idx[r, c] = [RC.atom_order[a] for a in atoms]
    # jnp's gather CLAMPS an out-of-range index, so aatype 21 against a
    # 21-row table reads row 20; row 21 here repeats it
    chi_idx[21] = chi_idx[20]
    chi_msk = np.zeros((22, 4), np.float32)
    chi_msk[:20] = np.asarray(RC.chi_angles_mask, np.float32)
    chi_msk[21] = chi_msk[20]
    return (np.asarray(RC.restype_atom14_mask, np.float32),
            np.asarray(RC.restype_atom37_mask, np.float32),
            np.asarray(RC.restype_atom14_to_atom37, np.int32),
            np.asarray(RC.restype_atom37_to_atom14, np.int32),
            chi_idx, chi_msk)


def _tables():
    """...and as device arrays, built fresh on every call.

    CACHING THE jnp ARRAYS LEAKS A TRACER.  The first call happens inside
    `_step`, which is being traced for custom_vjp, so an lru_cache of the
    converted arrays hands the SECOND trace a value that belongs to the
    first -- `UnexpectedTracerError`, raised from ffi_call and pointing at
    the cache rather than at the conversion.  The numpy side is cached;
    the conversion is not.
    """
    return tuple(jnp.asarray(a) for a in _table_arrays())


def _out_shapes(nmax):
    s = jax.ShapeDtypeStruct
    return (s((nmax, 37, 3), F32), s((nmax,), F32), s((nmax, nmax), F32),
            s((), F32), s((), F32), s((nmax,), F32), s((nmax,), F32),
            s((nmax, nmax, DBINS), F32))


# cfg is nondiff, so every member of it must be HASHABLE: the weight
# directories travel as a tuple of pairs, never a dict.  And NOTHING that a
# trajectory varies belongs in here -- the four design scalars are a
# traced INPUT (see `scalars` below), because they are attributes of the
# compiled call otherwise and bc2 ramps them.
@functools.partial(jax.custom_vjp, nondiff_argnums=(13,))
def _step(sequence, atoms, atom_mask, flags, residue_index, asym_id,
          entity_id, interface_asym_id, seq_mask, act_const, length, seed,
          scalars, cfg):
    out, _ = _step_fwd(sequence, atoms, atom_mask, flags, residue_index,
                       asym_id, entity_id, interface_asym_id, seq_mask,
                       act_const, length, seed, scalars, cfg)
    return out


def _call(name, result, args, cfg):
    recycles, msa_rows, dirs = cfg
    d = dict(dirs)
    return jax.ffi.ffi_call(name, result)(
        *args,
        recycles=np.int64(recycles), msa_rows=np.int64(msa_rows),
        trunk_dir=d['trunk'], extra_dir=d['extra'],
        tpl_dir=d['tpl'], post_dir=d['post'], emb_dir=d['emb'])


def _step_fwd(sequence, atoms, atom_mask, flags, residue_index, asym_id,
              entity_id, interface_asym_id, seq_mask, act_const, length,
              seed, scalars, cfg):
    nmax = sequence.shape[0]
    args = (sequence, atoms, atom_mask, flags, residue_index, asym_id,
            entity_id, interface_asym_id, seq_mask, act_const, length, seed)
    out = _call('native_design_step', _out_shapes(nmax),
                args + _tables() + (scalars,), cfg)
    # THE RESIDUAL IS THE INPUT, not the forward's internal state.  The
    # backward re-runs the forward on the device for exactly this reason:
    # a design step runs forwards for every state and every recycle before
    # any backward, so a residual that referred to device-side state would
    # be read after something else had overwritten it.
    return out, (args, scalars)


def _step_bwd(cfg, res, cts):
    args, scalars = res
    nmax = args[0].shape[0]
    # THE COTANGENTS GO IN THE CONTRACT'S ORDER and the scalars last, which
    # is the order ffi.cu binds them.  A reordering here is silent: every
    # one of them is an [N, ...] float32 buffer.
    d_sequence = _call(
        'native_design_step_bwd', jax.ShapeDtypeStruct((nmax, 20), F32),
        args + _tables()
        + tuple(jnp.asarray(c, F32) for c in cts) + (scalars,), cfg)
    zeros = tuple(jnp.zeros_like(a) for a in args[1:]) + (
        jnp.zeros_like(scalars),)
    return (d_sequence,) + zeros


_step.defvjp(_step_fwd, _step_bwd)


_PREALLOCED: set = set()


def _prealloc(nmax, recycles, msa_rows, weight_dirs):
    """Build the fused step now, while the card is still empty.

    A failure here is not fatal: the step is built on first call as before,
    in the order that loses the race.
    """
    if os.environ.get('NB_NO_PREALLOC'):
        return
    key = (nmax, int(recycles), int(msa_rows),
           tuple(sorted(weight_dirs.items())))
    if key in _PREALLOCED:
        return
    _PREALLOCED.add(key)
    from . import native as _bcn
    _bcn._register()
    fn = getattr(_bcn._LIBH, 'native_design_step_prealloc', None)
    if fn is None:
        print('native: this libnative_evoformer.so has no '
              'native_design_step_prealloc; the fused step will be built '
              'in call order', file=sys.stderr)
        return
    import ctypes
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_int] * 3 + [ctypes.c_char_p] * 5
    d = weight_dirs
    fn(int(nmax), int(recycles), int(msa_rows),
       str(d['trunk']).encode(), str(d['extra']).encode(),
       str(d['tpl']).encode(), str(d['post']).encode(),
       str(d['emb']).encode())


TILE = 32


def pad_to_tile(n):
    """The NMAX the kernels can actually tile.

    THE BUFFERS MUST BE 32-ALIGNED EVEN THOUGH `length` IS A RUNTIME VALUE.
    The runtime length is what collapses the per-bucket graphs into one; it
    does NOT free the ALLOCATION from the tiling constraint, and an
    unaligned NMAX dies in cuBLAS -- `trimul_body.cuh:1016 cublas: 15`
    (NOT_SUPPORTED), not a shape error.

    No gate could have caught this: ds_compare.py and ds_grad.py both run
    DS_N=64, and every benchmark length in this repo is 32-aligned, so the
    whole glue port had never seen a length the kernels cannot tile.  The
    first real complex is 215.
    """
    return (n + TILE - 1) // TILE * TILE


def design_step(sequence, atoms, atom_mask, flags, residue_index, asym_id,
                entity_id, interface_asym_id, seq_mask, act_const, length,
                seed, *, softmax_weight, one_hot_weight, temperature,
                logit_scale, recycles, msa_rows, weight_dirs
                ) -> DesignStepOutputs:
    """One call, sequence to the nine arrays boundary.py lists.

    `length` is a VALUE -- a device buffer holding one int32 -- and not a
    shape.  That is the whole point: an attribute or a shape would put the
    graph back to one per 32-residue bucket and the port would buy nothing.
    """
    # BUILD IT AT TRACE TIME, for the same reason the staged path does.
    # Built on first CALL, the fused step finds XLA already holding 22.54
    # of 23.68 GB at complex 487 and dies with 0.77 GB to work in -- it is
    # losing the allocation race, not carrying a bigger footprint, and it
    # loses harder because ONE call allocates everything at once.
    _prealloc(pad_to_tile(sequence.shape[0]), recycles, msa_rows,
              weight_dirs)

    # THE SEQUENCE ARRIVES float16 FROM bc2 AND THE CALL IS float32.
    # custom_vjp requires a cotangent to carry its primal's dtype exactly,
    # so a f32 d_sequence against a f16 primal is an AssertionError that
    # names two ShapedArrays and nothing else.  Casting OUTSIDE _step puts
    # an ordinary convert in the graph, and JAX takes the f32 cotangent
    # back through it to f16 on its own.
    sequence = jnp.asarray(sequence, F32)
    cfg = (recycles, msa_rows, tuple(sorted(weight_dirs.items())))
    scalars = jnp.stack([jnp.asarray(softmax_weight, F32),
                         jnp.asarray(one_hot_weight, F32),
                         jnp.asarray(temperature, F32),
                         jnp.asarray(logit_scale, F32)])
    (pos, plddt, pae, ptm, iptm, iptm_pr, expres, dgram) = _step(
        sequence, atoms, atom_mask, flags, residue_index, asym_id, entity_id,
        interface_asym_id, seq_mask, act_const, length, seed, scalars, cfg)
    # atom_mask comes from the inputs, not the call: it is
    # restype_atom37_mask[aatype] * seq_mask, carries no gradient, and the
    # step already masks the positions with it
    aatype = jnp.argmax(sequence, -1)
    mask37 = (jnp.asarray(RC.restype_atom37_mask, F32)[aatype]
              * seq_mask[:, None])
    return DesignStepOutputs(
        atoms=pos.astype(jnp.float16), atom_mask=mask37 > 0,
        plddt=plddt, pae=pae, ptm=ptm, iptm=iptm, iptm_per_residue=iptm_pr,
        experimentally_resolved_ca=expres, distogram=dgram)
