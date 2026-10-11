# WHERE DOES THE REMAINING COMPILE GO?  The post-trunk boundary took the
# design graph from 17.0 s to 7.7 s; before porting anything else, measure
# which part of what is left is worth porting.  Stage 2 is the reason this
# comes first -- I estimated the embedder at ~3 s and it was 0.8 s, because
# the estimate counted the template EMBEDDING, which is a different thing.
#
# Arms, all with the native trunk and the native post-trunk step:
#   full      everything as shipped
#   notpl     template.enabled = False  -> the whole template path's share
#   norecyc   recycle_pos/features off  -> the recycle terms' share
#
#   for a in full notpl norecyc; do python native/compile_budget.py $a; done
#
# Measured 2026-10-11, N=179, cold cache, native trunk + native post-trunk:
#
#   full      design_loss XLA  7.7 s
#   notpl                      4.0 s   -> the template path is 3.7 s, 48%
#   norecyc                    7.5 s   -> the recycle terms are 0.2 s, 3%
#
# So the template EMBEDDING -- dgram, unit vectors, torsions, the projections
# and the aggregating attention, the pair stack already being native -- is
# the only large piece left, and it is 4.6x what the embedder was worth.  The
# recycle terms are worth nothing, which is why folding them into the native
# embedder instead of adding them after it would buy nothing.
#
# AND MOST OF THAT 3.7 s IS A CONSTANT.  SingleTemplateEmbedding.
# construct_input builds nine features and projects each with its own Linear;
# EIGHT of the nine depend only on the template -- its aatype, atom positions
# and atom mask -- and not on the designed sequence.  Designed residues are
# masked to aatype 21 whatever they currently are, so changing them changes
# nothing.  Measured, not argued: calling target_template_features twice with
# the designed half of the sequence altered returns all seven tensors BIT
# IDENTICAL.
#
# So make_backbone_affine, the rigid inverse and normalisation, pseudo_beta_fn,
# dgram_from_positions, the one-hots, the masks and eight of the nine
# projections are recompiled into every design graph to recompute a value that
# cannot change.  Only template_pair_embedding_8, on query_embedding, varies.
#
# HOISTING THEM IS WORTH 0.4 s, NOT 3 s, AND THAT WAS MEASURED BEFORE IT WAS
# BUILT.  Feeding the design graph a precomputed constant in place of those
# eight features -- a zeros array answers the question, since compile time is
# a property of the graph and not of the data -- takes design_loss from 7.8 s
# to 7.4 s.  The substitution itself is exact: running the whole of
# SingleTemplateEmbedding with the constant handed back reproduces the shipped
# output bit for bit (max|d| 0.000e+00).
#
# So the 3.7 s the template path costs is NOT mostly its feature
# construction.  What is left is the pair stack's presence in the graph even
# though its blocks are native, the hk.scan around it, the aggregation and
# output_linear.  Those are where a further attempt should look.
#
# The hoist was implemented, proven exact, measured at 0.4 s and then REMOVED,
# because the part that remained -- computing the constant outside the graph
# and threading it through af2.py's compiled path -- is real surgery on a hot
# path, and 0.4 s does not buy it.  Reverting cost nothing; building the
# plumbing first and measuring second would have.
import os, sys, time, logging, shutil
ARM = sys.argv[1]
cache = f'/tmp/claude-1000/budget_{ARM}'
shutil.rmtree(cache, ignore_errors=True)
os.environ['JAX_COMPILATION_CACHE_DIR'] = cache
sys.path.insert(0, '/home/ubuntu/bc2')
import jax
jax.config.update('jax_log_compiles', True)
ev = []
class G(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if 'Finished XLA compilation' in m and 'sequence_design_loss' in m:
            try: ev.append(float(m.rsplit(' in ', 1)[1].split(' ')[0]))
            except Exception: pass
lg = logging.getLogger('jax'); lg.addHandler(G()); lg.setLevel(logging.DEBUG)

import bindcraft.af.alphafold.model.config as cfg
_real = cfg.model_config
def _patched(name):
    c = _real(name)
    e = c.model.embeddings_and_evoformer
    if ARM == 'notpl':
        e.template.enabled = False
    elif ARM == 'norecyc':
        e.recycle_pos = False
        e.recycle_features = False
    return c
cfg.model_config = _patched
import bindcraft.af2 as m
if hasattr(m, 'af_config'): m.af_config.model_config = _patched
_e = _patched('model_1_multimer_v3').model.embeddings_and_evoformer
print(f'  [{ARM}] template={_e.template.enabled} '
      f'recycle_pos={_e.recycle_pos} recycle_features={_e.recycle_features}')

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
t = time.perf_counter()
r = af.sequence_gradients(st, ls, model='model_1_multimer_v3')
jax.block_until_ready(jax.tree_util.tree_leaves(r[-1])[0])
print(f'  [{ARM:<8}] first {time.perf_counter()-t:6.1f} s   '
      f'design_loss XLA {sum(ev):6.1f} s')
