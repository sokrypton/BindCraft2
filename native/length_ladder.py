# WHAT RUNS, HOW FAST, AND ON WHICH BACKEND, at a real target+binder complex.
#
#   python native/length_ladder.py                 # the whole ladder, native
#   python native/length_ladder.py jax             # the same on JAX
#   python native/length_ladder.py native hIL2R_beta_gamma.pdb 150
#
# WHY THIS EXISTS.  Every speed number this directory had been quoting came
# from a BINDER WITH NO TARGET -- Protein.empty(N) for both states -- which
# keeps the complex under af2.py's 384-residue subbatch threshold and under
# the length where the kernels run out of memory.  Both of the things worth
# measuring were therefore invisible.  A real complex is the point:
#
#   * the target chain must be named target_<STATE>, i.e. target_complex for
#     a state called `complex`.  Name it anything else and an interface loss
#     raises KeyError: 'target' -- and a binder-only loss set never notices.
#   * the loss set matters.  plddt_loss and compactness default to
#     prediction_state='binder_alone' and sequence_entropy reads the binder's
#     SEQUENCE, so with those three alone NO GRADIENT CROSSES THE INTERFACE:
#     |grad| comes back identical for different targets and different
#     lengths.  weights_interface_contacts is what makes it a binder problem,
#     and it doubles the step.
#
# Measured 2026-10-12, A10 23.7 GB, model_1_multimer_v3, 0 recycles:
#
#   complex   native      JAX       note
#      215     520 ms    1995 ms    1.75x (hPDL1 + a 100-mer)
#      487    1539       4234       the one shipped target that used to OOM
#      537    1999       6220       native since the shallow-stack margin fix
#      587    7692*      7692       *declines: past BC2_NATIVE_MAX_N (544)
#      837   21689*     28896       *declines
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bindcraft                      # FIRST: it sets PREALLOCATE=false, and
                                      # importing jax before it lets JAX take
                                      # 75% of the card
import jax, numpy as np
from bindcraft.af2 import AlphaFoldDesignModel as AF2
from bindcraft.protein import Protein, BINDER_CHAIN_PREFIX, target_chain_name
from bindcraft.protein_preparation import merge_receptor_chains
from bindcraft.loss import build_losses

HERE = os.path.dirname(os.path.abspath(__file__))
STRUCT = os.path.join(os.path.dirname(HERE), 'settings', 'target', 'structures')
MODEL = 'model_1_multimer_v3'
LADDER = [('hPDL1.pdb', 100), ('hIL2R_beta_gamma.pdb', 100),
          ('hIL2R_beta_gamma.pdb', 150), ('hIL2R_beta_gamma.pdb', 200)]

def one(backend, pdb, binder_len, reps=6):
    af = AF2(presets=MODEL, data_dir='/home/ubuntu/params', key=jax.random.PRNGKey(0),
             num_recycle=0, length_bucket_size=1, dropout=False, backend=backend)
    tgt = merge_receptor_chains(list(Protein.from_structure(os.path.join(STRUCT, pdb)).values()))
    binder = Protein.empty(binder_len, jax.random.PRNGKey(1))
    states = {'complex': {BINDER_CHAIN_PREFIX: binder,
                          target_chain_name('target', 'complex'): tgt},
              'binder_alone': {BINDER_CHAIN_PREFIX: binder}}
    losses = build_losses({'weights_plddt_loss': 1.0, 'weights_sequence_entropy': 0.1,
                           'weights_compactness': 0.1, 'weights_interface_contacts': 1.0})
    wait = lambda r: jax.block_until_ready(jax.tree_util.tree_leaves(r))
    t = time.perf_counter(); r = af.sequence_gradients(states, losses, model=MODEL); wait(r)
    compile_s = time.perf_counter() - t
    wait(af.sequence_gradients(states, losses, model=MODEL))      # the native path
                                                                  # wants two warm calls
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        r = af.sequence_gradients(states, losses, model=MODEL); wait(r)
        ts.append((time.perf_counter() - t) * 1e3)
    g = jax.tree_util.tree_leaves(r[-1])
    gn = float(sum((np.asarray(v, np.float64) ** 2).sum() for v in g) ** 0.5)
    print(f'  {backend:<7} {pdb:<24} binder {binder_len:<4} complex {len(tgt)+binder_len:<5} '
          f'compile {compile_s:6.2f} s  step {np.median(ts):8.1f} ms  |grad| {gn:.6g}', flush=True)

if __name__ == '__main__':
    backend = sys.argv[1] if len(sys.argv) > 1 else 'native'
    if len(sys.argv) > 3:
        one(backend, sys.argv[2], int(sys.argv[3]))
    else:
        # ONE LENGTH PER PROCESS, and not for tidiness.  g_stacks keys on N and
        # has no destructors, so a second length builds its stacks beside the
        # first one's; XLA's pool grows alongside and never shrinks.  Run the
        # ladder in a single process and everything after the first length
        # dies in XLA with RESOURCE_EXHAUSTED -- which says nothing about
        # whether that length can run.  (Same reason the determinism probes
        # take one arm per process: the native hooks are module-level globals.)
        import subprocess
        for pdb, bl in LADDER:
            subprocess.run([sys.executable, os.path.abspath(__file__),
                            backend, pdb, str(bl)])
