"""The whole design step as ONE custom call, with its gradient.

This is the thing the glue port exists for.  An XLA custom call takes
STATIC shapes, so ten calls padded to 32 in JAX leave the graph specialised
per bucket and the glue between them is still XLA's.  One call whose
buffers are sized NMAX and whose LENGTH IS A RUNTIME INPUT collapses every
bucket into one graph -- and a loss is still written in Python, because the
loss is the only thing left on the JAX side.

WIRED IN AND ON BY DEFAULT (2026-10-13).  af2.py's gradient path takes this
branch whenever the kernels installed; BC2_NATIVE_DISABLE=step runs the
staged path instead.

It was opt-in for as long as it was, because the acceptance evidence is a
DISTRIBUTION OVER TRAJECTORIES (native/outcome_ladder.py) and not a gate --
a bf16 Evoformer has no reproducible trajectory to be right against.  A
half-gated native step that runs is worse than one that does not, because
the campaign still finishes and the binders are merely worse; this path
spent weeks applying no dropout at all, which is exactly that failure.

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


CZ, CM, TRUNK_DEPTH = 128, 256, 48


def _out_shapes(nmax, msa_rows):
    s = jax.ShapeDtypeStruct
    from . import native as _bcn
    # THE TRUNK, CARRIED TO THE BACKWARD.  Four more outputs: the save set
    # the forward already writes, plus pout/mout, so the backward does not
    # replay 48 blocks it has computed.  The save-set shape must match
    # Stack::saveSlots() EXACTLY -- both sides key on the same depth and N,
    # and a mismatch is not a shape error, it is a backward reading past
    # the end of someone else's buffer.
    slots = _bcn._save_slots(TRUNK_DEPTH, nmax)
    sm = msa_rows + 1          # the trunk carries the template row too
    return (s((nmax, 37, 3), F32), s((nmax,), F32), s((nmax, nmax), F32),
            s((), F32), s((), F32), s((nmax,), F32), s((nmax,), F32),
            s((nmax, nmax, DBINS), F32),
            s((slots, nmax, nmax, CZ), jnp.bfloat16),
            s((slots, sm, nmax, CM), F32),
            s((nmax, nmax, CZ), F32), s((sm, nmax, CM), F32))


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
    recycles, msa_rows, dropout, dirs = cfg
    d = dict(dirs)
    return jax.ffi.ffi_call(name, result)(
        *args,
        recycles=np.int64(recycles), msa_rows=np.int64(msa_rows),
        # CAN this step drop, not does it drop now.  The attribute decides
        # whether the two Evoformer stacks reserve their mask scratches,
        # which happens once when the step is built; WHETHER a mask is drawn
        # is seed[1], a traced value, because bc2 flips dropout between the
        # stages of one trajectory and a static flag would build a second
        # step beside this one -- two graphs is tolerable, two sets of N^2
        # buffers is not.
        dropout=np.int64(dropout),
        trunk_dir=d['trunk'], extra_dir=d['extra'],
        tpl_dir=d['tpl'], post_dir=d['post'], emb_dir=d['emb'])


def _step_fwd(sequence, atoms, atom_mask, flags, residue_index, asym_id,
              entity_id, interface_asym_id, seq_mask, act_const, length,
              seed, scalars, cfg):
    nmax = sequence.shape[0]
    args = (sequence, atoms, atom_mask, flags, residue_index, asym_id,
            entity_id, interface_asym_id, seq_mask, act_const, length, seed)
    recycles, msa_rows, _, _ = cfg
    out = _call('native_design_step', _out_shapes(nmax, msa_rows),
                args + _tables() + (scalars,), cfg)
    # THE RESIDUAL IS THE INPUT, never the forward's device-side state.  A
    # design step runs forwards for every state and every recycle before
    # any backward, so anything the backward read off a C++ member would
    # have been overwritten by another state by then (that is the bug that
    # cost i_pTM 0.79 -> 0.23).  What the backward can be handed instead is
    # BUFFERS: the last four outputs are the trunk's save set and its
    # pout/mout, carried as XLA arrays so the backward skips 48 blocks
    # rather than recomputing them.  They are residual, not results.
    return out[:8], (args, scalars, out[8:])


def _step_bwd(cfg, res, cts):
    args, scalars, carry = res
    nmax = args[0].shape[0]
    # THE COTANGENTS GO IN THE CONTRACT'S ORDER and the scalars last, which
    # is the order ffi.cu binds them.  A reordering here is silent: every
    # one of them is an [N, ...] float32 buffer.
    d_sequence = _call(
        'native_design_step_bwd', jax.ShapeDtypeStruct((nmax, 20), F32),
        args + _tables()
        + tuple(jnp.asarray(c, F32) for c in cts) + (scalars,) + carry, cfg)
    zeros = tuple(jnp.zeros_like(a) for a in args[1:]) + (
        jnp.zeros_like(scalars),)
    return (d_sequence,) + zeros


_step.defvjp(_step_fwd, _step_bwd)


_PREALLOCED: set = set()

# THE BIGGEST LENGTH THIS DESIGN WILL RUN, when one DesignStep is to serve
# every state (NB_ONE_OBJECT).  The native side sizes its buffers on the
# FIRST alloc and setN only ever runs them shorter, so if the 220-residue
# state traces before the 607-residue one the object is built at 224 and the
# long state has nowhere to go.  States trace in sorted-name order, which
# says nothing about length, so the floor has to be stated rather than hoped
# for.  af2.py sets it from `_biggest` before tracing.
_ALLOC_FLOOR = 0


def set_alloc_floor(n: int) -> None:
    global _ALLOC_FLOOR
    _ALLOC_FLOOR = max(_ALLOC_FLOOR, int(n))


def one_object() -> bool:
    """One DesignStep for every state, run at each state's own length,
    instead of one per 32-bucket.  ON by default; NB_ONE_OBJECT=0 opts
    out."""
    return os.environ.get('NB_ONE_OBJECT', '1') not in ('0', '')


def _prealloc(nmax, recycles, msa_rows, dropout, weight_dirs):
    """Build the fused step now, while the card is still empty.

    A failure here is not fatal: the step is built on first call as before,
    in the order that loses the race.
    """
    if os.environ.get('NB_NO_PREALLOC'):
        return
    # WARM EVERY LENGTH THIS DESIGN WILL RUN, not just the biggest.
    #
    # With one object, a state shorter than the allocation re-derives into
    # the buffers already held -- except where a shorter length needs a
    # BIGGER buffer than the long one did, which the chunk policies make
    # real: the pair transition runs unchunked below its threshold and its
    # hidden activation goes from 63 MB to 105.  Asked for on the first
    # call, that is a mid-step cudaMalloc with XLA holding the card, and
    # complex 687 died at `transition.cuh:422` with it.
    #
    # The floor builds the buffers; the second call re-derives to this
    # state's own length, both at TRACE time, so every allocation either
    # length will ever want is made while the card is still empty.
    nmax = int(nmax)
    floor = max(nmax, _ALLOC_FLOOR)
    lengths = [floor] if nmax == floor else [floor, nmax]
    for n in lengths:
        _prealloc_one(n, recycles, msa_rows, dropout, weight_dirs)


def _prealloc_one(nmax, recycles, msa_rows, dropout, weight_dirs):
    key = (nmax, int(recycles), int(msa_rows), int(dropout),
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
    fn.argtypes = [ctypes.c_int] * 4 + [ctypes.c_char_p] * 5
    d = weight_dirs
    fn(int(nmax), int(recycles), int(msa_rows), int(dropout),
       str(d['trunk']).encode(), str(d['extra']).encode(),
       str(d['tpl']).encode(), str(d['post']).encode(),
       str(d['emb']).encode())


def _predict_prealloc_one(nmax, recycles, msa_rows, dropout, weight_dirs):
    """The predict path's own buffers, at trace time.

    predictBuffers() allocates one reused save slot and the trunk's own
    pout/mout -- ~285 MB at complex 608 -- and asked for on the first CALL
    it is asked for with XLA already holding the card.  Same failure, same
    fix, as the design step's prealloc above.
    """
    _prealloc_one(nmax, recycles, msa_rows, dropout, weight_dirs)
    from . import native as _bcn
    _bcn._register()
    import ctypes
    fn = getattr(_bcn._LIBH, 'native_design_step_predict_prealloc', None)
    if fn is None:
        print('native: this libnative_evoformer.so has no '
              'native_design_step_predict_prealloc; the predict buffers '
              'will be built in call order', file=sys.stderr)
        return
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_int] * 4 + [ctypes.c_char_p] * 5
    d = weight_dirs
    fn(int(nmax), int(recycles), int(msa_rows), int(dropout),
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
                logit_scale, recycles, msa_rows, weight_dirs, dropout=True
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
    _prealloc(pad_to_tile(sequence.shape[0]), recycles, msa_rows, dropout,
              weight_dirs)

    # THE SEQUENCE ARRIVES float16 FROM bc2 AND THE CALL IS float32.
    # custom_vjp requires a cotangent to carry its primal's dtype exactly,
    # so a f32 d_sequence against a f16 primal is an AssertionError that
    # names two ShapedArrays and nothing else.  Casting OUTSIDE _step puts
    # an ordinary convert in the graph, and JAX takes the f32 cotangent
    # back through it to f16 on its own.
    sequence = jnp.asarray(sequence, F32)
    cfg = (recycles, msa_rows, int(bool(dropout)),
           tuple(sorted(weight_dirs.items())))
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


# ---------------------------------------------------------------------------
# THE PREDICT-ONLY CALL
#
# bc2's six stages are not six of a kind.  screen / refine / anneal / harden
# take a gradient; `mutate` (semi-greedy), the MPNN stage's reprediction and
# the dozens of *_final filters only ever PREDICT.  A forward is about a
# third of a design step -- 176 ms against 589 at complex 175 -- and the
# other two thirds are the backward, so this is the price of that half of
# the pipeline.
#
# What it actually buys is MEMORY, and that is worth being exact about
# because the first reading of the benchmark was wrong.  `forward(save)` and
# `forward(no save)` timed the same because `want_save` was not wired to
# anything -- see the note on DesignStep::forward.  Now that it is, the
# difference is not the save WRITES (small) but the four carry buffers XLA
# does not allocate: psave alone is slots * NMAX^2 * 128 in bf16, which is
# the single biggest buffer in a design step.  So a predict runs at lengths
# a design step cannot.
#
# No custom_vjp, deliberately.  This call has no backward, and giving it one
# that raised would make a mis-wired caller fail at trace time with a
# confusing message; without one, JAX says the obvious thing -- there is no
# rule for this primitive -- and the caller is meant to be inside a
# stop_gradient or outside a trace anyway.


def predict_step(sequence, atoms, atom_mask, flags, residue_index, asym_id,
                 entity_id, interface_asym_id, seq_mask, act_const, length,
                 seed, *, softmax_weight, one_hot_weight, temperature,
                 logit_scale, recycles, msa_rows, weight_dirs, dropout=False
                 ) -> DesignStepOutputs:
    """One call, sequence to the same nine arrays -- with no gradient.

    The operand contract is the design step's, exactly: the two handlers
    share a Bind list and a DsKey, so a predict and a design at the same
    length share ONE DesignStep and one allocation.  `dropout` defaults
    False here and True there, because a prediction is a measurement.
    """
    nmax = pad_to_tile(sequence.shape[0])
    _predict_prealloc_one(nmax, recycles, msa_rows, int(bool(dropout)),
                          weight_dirs)
    sequence = jnp.asarray(sequence, F32)
    cfg = (recycles, msa_rows, int(bool(dropout)),
           tuple(sorted(weight_dirs.items())))
    scalars = jnp.stack([jnp.asarray(softmax_weight, F32),
                         jnp.asarray(one_hot_weight, F32),
                         jnp.asarray(temperature, F32),
                         jnp.asarray(logit_scale, F32)])
    args = (sequence, atoms, atom_mask, flags, residue_index, asym_id,
            entity_id, interface_asym_id, seq_mask, act_const, length, seed)
    # the first EIGHT of _out_shapes: the last four are the carry, and not
    # producing them is the whole point
    out = _call('native_design_step_predict',
                _out_shapes(sequence.shape[0], msa_rows)[:8],
                args + _tables() + (scalars,), cfg)
    pos, plddt, pae, ptm, iptm, iptm_pr, expres, dgram = out
    aatype = jnp.argmax(sequence, -1)
    mask37 = (jnp.asarray(RC.restype_atom37_mask, F32)[aatype]
              * seq_mask[:, None])
    return DesignStepOutputs(
        atoms=pos.astype(jnp.float16), atom_mask=mask37 > 0,
        plddt=plddt, pae=pae, ptm=ptm, iptm=iptm, iptm_per_residue=iptm_pr,
        experimentally_resolved_ca=expres, distogram=dgram)
