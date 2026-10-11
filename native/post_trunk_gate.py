# Does replacing the POST-TRUNK step move the model more than replacing the
# main Evoformer stack already does?
#
#   for m in xla main main+post main+post main; do \
#     python native/post_trunk_gate.py $m; done
#
# Measured 2026-10-11, N=179, dropout off, one process an arm:
#
#   xla        plddt 0.450951  ptm 0.145564  pae 26.217312  |grad| 10.6216
#   main       plddt 0.451231  ptm 0.146857  pae 26.217445  |grad|  5.4268
#   main+post  plddt 0.451212  ptm 0.146862  pae 26.217400  |grad|  4.4129
#   main+post  plddt 0.451212  ptm 0.146862  pae 26.217400  |grad|  4.4118
#   main       plddt 0.451235  ptm 0.146857  pae 26.217445  |grad|  5.4032
#
# The repeats are the floor and it is small: main+post twice is identical on
# every forward metric and 0.025% apart on the gradient norm.  Against that,
# swapping the main stack moves pLDDT 2.8e-04 and the gradient norm 48.9%;
# ADDING the post-trunk moves them a further 1.9e-05 and 18.7% -- 15x and 5x
# less than the swap already on by default.  The gradient move is real, not
# noise; it is simply smaller than the one the model already accepts.
#
# That is the bar the extra-MSA and template stacks were judged on, and it is
# the right one here: a design trajectory is chaotic, so an absolute number
# means nothing until you know what the same configuration does twice.  The
# campaign attempt failed exactly there -- two identical runs disagreed by
# more than the arms did.
#
# DROPOUT OFF.  With it on the native masks are a counter hash and JAX's are
# threefry, so the two are comparable in distribution only and no single
# forward can be compared at all.
import os, sys
MODE = sys.argv[1]                      # xla | main | main+post
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['BC2_NATIVE_POST'] = '1' if MODE == 'main+post' else '0'
sys.path.insert(0, '/home/ubuntu/bc2')
import numpy as np, jax
from bindcraft.af2 import AlphaFoldDesignModel as AF2
from bindcraft.protein import Protein
from bindcraft.loss import build_losses
import bindcraft.af.alphafold.model.modules_multimer as mm

af = AF2(presets='model_1_multimer_v3', data_dir='/home/ubuntu/params',
         key=jax.random.PRNGKey(0), num_recycle=0, length_bucket_size=1,
         dropout=False,
         evoformer_backend='jax' if MODE == 'xla' else 'native')
p = Protein.empty(179, jax.random.PRNGKey(1))
st = {'complex': {'binder': p}, 'binder_alone': {'binder': p}}
# single-chain losses: the interface ones want a target chain this toy state
# has none of, and the question here is whether the post-trunk path moves the
# model at all -- the structure module and all four heads run either way
ls = build_losses({'weights_plddt_loss': 1.0, 'weights_sequence_entropy': 0.1,
                   'weights_compactness': 0.1})
r = af.sequence_gradients(st, ls, model='model_1_multimer_v3')
# the annotation says a 2-tuple; it is not.  Index rather than unpack.
pred, grads = r[0], r[1]
m = pred['complex'].metrics
g = np.concatenate([np.asarray(v, np.float64).ravel()
                    for v in jax.tree_util.tree_leaves(grads)])
out = {k: float(np.asarray(m[k]).mean()) for k in
       ('plddt', 'ptm', 'pae') if k in m}
out['|grad|'] = float(np.linalg.norm(g))
print(f'{MODE:<10} post_native={mm.NATIVE_POST_TRUNK is not None} '
      f'traces={mm.NATIVE_POST_TRUNK_TRACES[0]}  ' +
      '  '.join(f'{k} {v:.6f}' for k, v in out.items()))
