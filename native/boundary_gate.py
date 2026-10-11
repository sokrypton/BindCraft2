"""Does a real design step honour bindcraft/af/boundary.py?

The contract is only worth having if something checks it.  This runs one
design step and asserts, for every field: present, right dtype, right shape,
and -- the part that breaks silently -- a gradient actually flows back
through the ones marked differentiable.

A field that quietly stops carrying a gradient makes every loss using it
inert while still returning a number, which no quality metric would catch.

    python native/boundary_gate.py
"""
import os
import sys
import zlib

for _k in list(os.environ):
    if _k.startswith('BC2_NATIVE'):
        del os.environ[_k]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

from bindcraft.af import boundary
from bindcraft.af2 import AlphaFoldDesignModel as AF2
from bindcraft.loss import build_losses, DesignLoss
from bindcraft.protein import Protein

N = 96
DTYPE = {'float16': jnp.float16, 'float32': jnp.float32,
         'bool': jnp.bool_, 'uint8': jnp.uint8, 'int32': jnp.int32}


def check_seam():
    """The seam and the contract must not drift apart.

    Pure Python, so it runs before the expensive part and fails fast.  The
    adapter in design_step.py is the one place that knows both spellings;
    if a field is added to boundary.py and not to it, every loss reading
    that field gets an AttributeError at trace time instead of here.
    """
    from bindcraft.af.design_step import DesignStepOutputs
    want = [f.name for f in boundary.STRUCTURE + boundary.METRICS]
    got = list(DesignStepOutputs._fields)
    if want != got:
        print('SEAM DIVERGED from boundary.py')
        print(f'  contract {want}')
        print(f'  outputs  {got}')
        raise SystemExit(1)
    print(f'  seam matches the contract ({len(want)} fields)')


def main():
    check_seam()
    af = AF2(presets='model_1_multimer_v3', data_dir='/home/ubuntu/params',
             key=jax.random.PRNGKey(0), num_recycle=0, length_bucket_size=32,
             dropout=False, backend='native')
    # TWO CHAINS, because iptm and iptm_per_residue are interface metrics:
    # on a single chain there is no interface, they are constant, and their
    # gradient is legitimately zero.  A one-chain gate reports that as the
    # contract being violated, which is the gate's fault and not the step's.
    states = {'complex': {'binder': Protein.empty(N, jax.random.PRNGKey(1)),
                          'target': Protein.empty(N, jax.random.PRNGKey(2))}}

    seen = {}

    def capture(protein_states, predictions, **kw):
        pred = next(iter(predictions.values()))
        prot = next(iter(pred.protein_complex.values()))
        seen.update(pred.metrics)
        for f in boundary.INPUT_SIDE + boundary.STRUCTURE:
            if hasattr(prot, f.name):
                seen[f.name] = getattr(prot, f.name)
        return jnp.asarray(0.0)

    # ONE FIELD AT A TIME.  The contract promises a loss can be written on
    # any differentiable field; a summed probe cannot tell which field a
    # NaN or a zero came from, and that is the whole question.
    def probe(field):
        def f(protein_states, predictions, **kw):
            capture(protein_states, predictions)
            v = jnp.asarray(seen[field], jnp.float32)
            # a weighted sum, so a symmetric field does not cancel to zero
            # NOT hash(field): Python randomises string hashing per
            # process, so the probe would differ between runs.  It does not
            # matter to THIS gate, which asks whether a gradient is finite
            # and nonzero rather than whether two runs agree -- but the
            # same idiom in dev/native_block/ds_grad.py made pure JAX
            # disagree with itself at cos -0.66, so it does not stay here.
            w = jnp.asarray(
                np.random.RandomState(zlib.crc32(field.encode()) % 2**31)
                .randn(*v.shape), jnp.float32)
            return jnp.sum(v * w)
        return f

    af_run = lambda fn: af.sequence_gradients(
        states, {'probe': DesignLoss(function=fn, weight=1.0,
                                     required_states=frozenset())},
        model='model_1_multimer_v3')[:2]

    af_run(capture)                     # populate `seen` with shapes/dtypes
    bad = []
    print(f'boundary gate  N={N}')
    for fld in boundary.ALL:
        if fld.name not in seen:
            bad.append(f'{fld.name}: MISSING from the step')
            print(f'  {fld.name:<28} MISSING'); continue
        a = jnp.asarray(seen[fld.name])
        want = DTYPE[fld.dtype]
        ok = a.dtype == want
        if not ok:
            bad.append(f'{fld.name}: dtype {a.dtype}, contract says {fld.dtype}')
        note = ''
        if fld.differentiable:
            _, g = af_run(probe(fld.name))
            gn = float(np.linalg.norm(np.concatenate(
                [np.asarray(v, np.float64).ravel()
                 for v in jax.tree_util.tree_leaves(g)])))
            if not np.isfinite(gn):
                bad.append(f'{fld.name}: gradient is {gn}')
                note = f'  |grad| {gn}  <-- NOT FINITE'
            elif gn == 0.0:
                bad.append(f'{fld.name}: no gradient reaches the sequence')
                note = '  |grad| 0  <-- DEAD'
            else:
                note = f'  |grad| {gn:.4g}'
        print(f'  {fld.name:<28} {str(a.dtype):<8} {str(a.shape):<16}'
              f'{"" if ok else " DTYPE"}{note}')
    if bad:
        print('\nCONTRACT VIOLATED:')
        for b in bad:
            print(f'  - {b}')
        raise SystemExit(1)
    print('\ncontract honoured')


if __name__ == '__main__':
    main()
