# WHAT DOES EACH NATIVE PIECE COST OR SAVE AT RUNTIME, now that they are all
# on?  The per-piece numbers in the notes come from different sessions and
# different configs.  This turns ONE flag off at a time against an all-on
# baseline, same process shape, same length.
import os, sys, time, shutil
ARM = sys.argv[1]
for k in list(os.environ):
    if k.startswith('BC2_NATIVE'):
        del os.environ[k]
if ARM != 'allon':
    os.environ[ARM] = '0'
cache = f'/tmp/claude-1000/fs_{ARM}'
shutil.rmtree(cache, ignore_errors=True)
os.environ['JAX_COMPILATION_CACHE_DIR'] = cache
sys.path.insert(0, '/home/ubuntu/bc2')
import numpy as np, jax
from bindcraft.af2 import AlphaFoldDesignModel as AF2
from bindcraft.protein import Protein
from bindcraft.loss import build_losses
af = AF2(presets='model_1_multimer_v3', data_dir='/home/ubuntu/params',
         key=jax.random.PRNGKey(0), num_recycle=0, length_bucket_size=1,
         dropout=True, evoformer_backend='native')
p = Protein.empty(179, jax.random.PRNGKey(1))
st = {'complex': {'binder': p}, 'binder_alone': {'binder': p}}
ls = build_losses({'weights_plddt_loss': 1.0, 'weights_sequence_entropy': 0.1,
                   'weights_compactness': 0.1})
r = af.sequence_gradients(st, ls, model='model_1_multimer_v3')
jax.block_until_ready(jax.tree_util.tree_leaves(r[-1])[0])
ts = []
for _ in range(6):
    t = time.perf_counter()
    r = af.sequence_gradients(st, ls, model='model_1_multimer_v3')
    jax.block_until_ready(jax.tree_util.tree_leaves(r[-1])[0])
    ts.append(time.perf_counter() - t)
ts = sorted(ts)[1:-1]          # drop best and worst
print(f'  {ARM:<22} step {1e3*np.mean(ts):7.1f} ms  +-{1e3*np.std(ts):4.1f}')
