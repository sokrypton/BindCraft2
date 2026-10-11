# Where does the STEP go, now that the trunk, the extra-MSA stack, the
# template stack and the post-trunk step are all native?  Compile is paid once
# per (model, length); this is paid every step of every trajectory, so it is
# the number that compounds.
#
# Same ablation arms as the compile budget, timed after warmup instead.
#
#   for a in full notpl nometrics; do python native/runtime_budget.py $a; done
#
# Measured 2026-10-11, N=179, native trunk + extra-MSA + templates + post-trunk:
#
#   full       step median 710.3 ms
#   notpl                  693.8 ms   -> the template path is 16.5 ms, 2.3%
#   nometrics              715.6 ms   -> the metrics are not measurable at all
#
# COMPILE AND RUNTIME ARE NEARLY DISJOINT HERE, which is the point of keeping
# both tables.  The template path is 48% of the compile and 2.3% of the step.
# The confidence metrics are 26% of the compile and do not register in the
# step -- removing them came out 5 ms SLOWER than leaving them, which is
# cross-process noise around zero.
#
# So the compile items that remain cost nothing to run, and anything done for
# RUNTIME has to go at the trunk, which is already native and already the
# subject of the kernel work.  Optimising one axis here does not move the
# other, and a change justified on one should not be sold on the other.
#
# VALIDATION IS 1.74% OF A TRAJECTORY and is not worth optimising.  A
# model_1_ptm predict at N=179 is 315 ms; a 500-step trajectory spends 355 s
# designing and 6.3 s validating its ten candidates across two models.
#
# That matters because there IS a gap in those 6.3 s: predict() fires the
# native post-trunk hook for model_1_multimer_v3 (traces=1) and NOT for
# model_1_ptm (traces=0), although both are installed, both run through
# modules_multimer -- model.py builds that graph unconditionally, monomer
# preset or not -- and their weighted head sets are identical, so the
# head-set check is not what declines it.  The fallback is safe: the hook
# returns None and JAX runs the span.
#
# WHY THE REMAINING COMPILE WINS ARE NOT SIMPLY TURNED ON.  Asked directly,
# and the answer turned out to be a number rather than a judgement.  Both of
# them, together:
#
#   baseline                      XLA 7.8 s   step 712.6 ms
#   + template query barrier      XLA   -     step 714.5 ms   (+1.9)
#   + barrier + native embedder   XLA 6.9 s   step 718.1 ms   (+5.5, +0.77%)
#
# 0.9 s of compile for 5.5 ms a step.  Compile is paid once per (model,
# length) -- with the persistent cache, once per machine.  The step is paid
# every iteration: 5.5 ms x 500 steps is 2.75 s a trajectory.  Break-even is
# under two trajectories and a campaign runs dozens, so enabling both makes
# campaigns SLOWER.  That is the disjointness above biting the other way.
#
# They also do not add: 0.8 s and 0.7 s separately, 0.9 s together.
#
# THE NEW DEFAULTS ARE CAMPAIGN-VERIFIED.  Flipping them on without a
# campaign would have broken the rule every other default flip in this work
# followed, so: hPDL1, N=179, embedder and barrier both on, both hooks
# confirmed FIRED in the trace --
#
#   screen  i_pTM 0.53  pLDDT 0.84
#   refine        0.79        0.91
#   anneal        0.77        0.93
#   harden        0.75        0.92
#   mutate        0.78        0.91
#
# All five design stages passed, where the first JAX baseline on this target
# was rejected at mutate.  That is not evidence the new defaults are BETTER --
# two identical campaigns here disagreed by more than this -- only that they
# are healthy end to end.
#
# The flags stay available because the trade REVERSES for a one-shot run or a
# cold machine, where the compile is most of the cost -- a Colab notebook, a
# single short job.  It is the campaign that cannot afford them.
#
# So it is a missed optimisation rather than a wrong answer, worth under a
# tenth of a percent of a trajectory, and it stays missed on purpose.  The
# cause was not chased past ruling out the obvious one, because the value did
# not justify the hunt -- which is the thing to know before repeating it.
import os, sys, time, shutil
ARM = sys.argv[1]
cache = f'/tmp/claude-1000/rt_{ARM}'
shutil.rmtree(cache, ignore_errors=True)
os.environ['JAX_COMPILATION_CACHE_DIR'] = cache
sys.path.insert(0, '/home/ubuntu/bc2')
import numpy as np, jax, jax.numpy as jnp
import bindcraft.af2 as af2mod
import bindcraft.af.alphafold.model.config as cfg

_real = cfg.model_config
def _patched(name):
    c = _real(name)
    e = c.model.embeddings_and_evoformer
    if ARM == 'notpl':
        e.template.enabled = False
    return c
cfg.model_config = _patched
if hasattr(af2mod, 'af_config'): af2mod.af_config.model_config = _patched
if ARM == 'nometrics':
    def only_plddt(outs, seq_mask, iface):
        m = {}
        if 'predicted_lddt' in outs:
            lg_ = outs['predicted_lddt']['logits']
            bw = 1.0 / lg_.shape[-1]
            bc = jnp.arange(0.5 * bw, 1.0, bw)
            m['plddt'] = (jax.nn.softmax(lg_, axis=-1) * bc[None, :]).sum(-1)
        return m
    af2mod.alphafold_prediction_metrics = only_plddt

from bindcraft.af2 import AlphaFoldDesignModel as AF2
from bindcraft.protein import Protein
from bindcraft.loss import build_losses
af = AF2(presets='model_1_multimer_v3', data_dir='/home/ubuntu/params',
         key=jax.random.PRNGKey(0), num_recycle=0, length_bucket_size=1,
         dropout=False, evoformer_backend='native')
p = Protein.empty(179, jax.random.PRNGKey(1))
st = {'complex': {'binder': p}, 'binder_alone': {'binder': p}}
ls = build_losses({'weights_plddt_loss': 1.0, 'weights_sequence_entropy': 0.1,
                   'weights_compactness': 0.1})
r = af.sequence_gradients(st, ls, model='model_1_multimer_v3')   # warm
jax.block_until_ready(jax.tree_util.tree_leaves(r[-1])[0])
ts = []
for _ in range(8):
    t = time.perf_counter()
    r = af.sequence_gradients(st, ls, model='model_1_multimer_v3')
    jax.block_until_ready(jax.tree_util.tree_leaves(r[-1])[0])
    ts.append((time.perf_counter() - t) * 1e3)
ts = np.array(ts)
print(f'  {ARM:<10} step  median {np.median(ts):7.1f} ms   min {ts.min():7.1f}   '
      f'spread {ts.max()-ts.min():5.1f}')
