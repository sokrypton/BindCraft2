# THE COMPILE FLOOR FOR "ONLY THE LOSS IS JAX".
# Synthetic inputs at the exact shapes that cross the loss boundary today,
# so no tracer capture is involved.  Whatever this compiles to is what a
# user's own loss still costs on a fresh run, however much glue is ported.
import os, sys, time, logging, shutil
for k in list(os.environ):
    if k.startswith('BC2_NATIVE'):
        del os.environ[k]
shutil.rmtree('/tmp/claude-1000/lf_cache', ignore_errors=True)
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/claude-1000/lf_cache'
sys.path.insert(0, '/home/ubuntu/bc2')
import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_log_compiles', True)
ev = []
class G(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if 'Finished XLA compilation' in m:
            try: ev.append(float(m.rsplit(' in ',1)[1].split(' ')[0]))
            except Exception: pass
lg = logging.getLogger('jax'); lg.addHandler(G()); lg.setLevel(logging.DEBUG)
from bindcraft.protein import Protein, StructurePrediction
from bindcraft.loss import build_losses

N = 192
rng = np.random.RandomState(0)
base = Protein.empty(N, jax.random.PRNGKey(1))
losses = build_losses({'weights_plddt_loss': 1.0,
                       'weights_sequence_entropy': 0.1,
                       'weights_compactness': 0.1})
ps = {'complex': {'binder': base}, 'binder_alone': {'binder': base}}

def pack():
    return dict(
        atoms=jnp.asarray(rng.randn(N, 37, 3), jnp.float16),
        sequence=jnp.asarray(rng.rand(N, 20), jnp.float16),
        plddt=jnp.asarray(rng.rand(N), jnp.float32),
        pae=jnp.asarray(rng.rand(N, N), jnp.float32),
        ptm=jnp.asarray(0.5, jnp.float32), iptm=jnp.asarray(0.5, jnp.float32),
        iptm_per_residue=jnp.asarray(rng.rand(N), jnp.float32),
        experimentally_resolved_ca=jnp.asarray(rng.rand(N), jnp.float32),
        distogram=jnp.asarray(rng.rand(N, N, 64), jnp.float32))

def total(d):
    pr = base.replace(atoms=d['atoms'], sequence=d['sequence'])
    mx = {k: v for k, v in d.items() if k not in ('atoms', 'sequence')}
    lp = {s: StructurePrediction(protein_complex={'binder': pr}, metrics=mx)
          for s in ps}
    return sum(float(e.weight) * e.function(ps, lp) for e in losses.values())

d = pack()
t = time.perf_counter()
v, g = jax.block_until_ready(jax.jit(jax.value_and_grad(total))(d))
print(f'  LOSS + GRAD ALONE: wall {time.perf_counter()-t:5.2f} s   '
      f'XLA {sum(ev):5.2f} s   over {len(d)} boundary arrays')
print(f'  loss value {float(v):.4f}, grads for {sorted(g)}')
