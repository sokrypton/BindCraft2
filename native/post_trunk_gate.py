'''Gate the post-trunk FFI -- structure module + four heads -- against AF2.

    python oracle_ipa.py --part post --n 211 --depth 8 --dump pt
    python post_trunk_weights.py --out post_weights
    python post_trunk_gate.py pt/ipa-post8-n211

N=211 is the length a real BindCraft campaign produces, NOT a 32-aligned
benchmark one: the kernels pad internally and the template runner's missing
pad is what crashed a campaign earlier, so every gate here runs unaligned.

It checks the FORWARD and the BACKWARD, and the backward is the point: the
composition is new.  The heads write dpair, then the structure module WRITES
it again rather than accumulating, so the heads' share is staged and added
back; the experimentally_resolved head's cotangent lands on the trunk single
while pLDDT's lands on the fold module's act.  Those two merges are exactly
what a single end-to-end number would hide and what this compares separately.
'''
import sys

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_default_matmul_precision', 'highest')
from native_post_trunk import native_post_trunk  # noqa: E402

OUTS = ('atom', 'dgram', 'pae', 'plddt', 'expres')


_RESULTS = []


def report(name, a, b):
  a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
  cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
  rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
  _RESULTS.append((name, rel))
  print(f'  {name:<14} cos {cos:.9f}  relL2 {rel:.3e}  '
        f'max|d| {np.abs(a - b).max():.3e}  '
        f'(rms {np.sqrt((a**2).mean()):.4f} vs {np.sqrt((b**2).mean()):.4f})')
  return cos, rel


def main():
  d = sys.argv[1]
  wd = sys.argv[2] if len(sys.argv) > 2 else 'post_weights'
  z = np.load(f'{d}/block.npz')
  single = jnp.asarray(z['in/single'])
  pair = jnp.asarray(z['in/pair'])
  pm = jnp.asarray(z['in/pair_mask'])
  aat = jnp.asarray(z['in/aatype'])
  n = single.shape[0]
  print(f'post-trunk gate  N={n}  weights={wd}')

  def run(s, p):
    return native_post_trunk(s, p, pm, aat, depth=8, weights_dir=wd)

  got = jax.block_until_ready(jax.jit(run)(single, pair))
  print('forward')
  for k, v in zip(OUTS, got):
    report(f'out/{k}', v, z[f'out/{k}'])

  ct = [jnp.asarray(z[f'ct/{k}']) for k in OUTS]

  def loss(s, p):
    return sum((o * c).sum() for o, c in zip(run(s, p), ct))

  g = jax.block_until_ready(jax.jit(jax.grad(loss, argnums=(0, 1)))(single,
                                                                    pair))
  print('backward')
  report('grad/single', g[0], z['grad/single'])
  report('grad/pair', g[1], z['grad/pair'])

  # THE INTERLEAVED BACKWARD.  A gate that runs one forward and then its
  # backward cannot see state left in the C++ object -- the state is still
  # the right one.  A design step runs forwards for every state and every
  # recycle BEFORE any backward, and that is what broke the template
  # embedding while every other gate it had said green.  This op replays from
  # single/pair/mask/aatype, which are all passed in, so it should be
  # immune; the point is to MEASURE that rather than read it off the source.
  def loss_interleaved(s, p):
    a = sum((o * c).sum() for o, c in zip(run(s, p), ct))
    b = sum(o.sum() for o in run(s * 0.5 + 0.1, p * 0.5 + 0.1))
    return a + 0.0 * b

  gi = jax.block_until_ready(
      jax.jit(jax.grad(loss_interleaved, argnums=(0, 1)))(single, pair))
  print('backward, with a second forward sequenced in between')
  report('grad/single', gi[0], np.asarray(g[0]))
  report('grad/pair', gi[1], np.asarray(g[1]))


if __name__ == '__main__':
  main()


# ---- THE VERDICT ------------------------------------------------------
#
# Three claims, and they are not the same strength:
#
#   the FORWARD is held tight (1e-3 relative).  dgram and pae come back
#   bit-identical and plddt to 5e-04; there is no reason for it to drift.
#   the BACKWARD is bf16 through the structure module, so it is held only
#   loosely (0.2) -- a REGRESSION CATCHER, not a precision claim, and it
#   says so rather than implying a floor it has not measured.
#   the INTERLEAVED backward must match the plain one (1e-4): that is
#   STATELESSNESS, and it is exact-ish regardless of precision.
_fwd = [(n, r) for n, r in _RESULTS if n.startswith('out/')]
_bwd = [(n, r) for n, r in _RESULTS if n.startswith('grad/')]
_inter = _bwd[2:] if len(_bwd) >= 4 else []
_plain = _bwd[:2]
bad = [n for n, r in _fwd if r > 1e-3]
bad += [n for n, r in _plain if r > 0.2]
bad += [f'{n}/interleaved' for n, r in _inter if r > 1e-4]
print(f'\n  forward worst {max((r for _, r in _fwd), default=0):.2e} (bound 1e-03)'
      f'   backward worst {max((r for _, r in _plain), default=0):.2e} (bound 2e-01, bf16)'
      f'   interleaved worst {max((r for _, r in _inter), default=0):.2e} (bound 1e-04)')
print('  ' + ('the post-trunk matches AF2 and is stateless' if not bad
              else f'ABOVE BOUND: {", ".join(bad)}'))
raise SystemExit(1 if bad else 0)
