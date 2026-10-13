# DOES THE FUSED STEP ACTUALLY DROP?
#
#   python native/dropout_gate.py
#
# It did not, and nothing said so.  `BC2_NATIVE_STEP` passed [0, 0] as the
# seed -- seed 0, dropout DISABLED -- and DesignStep never pointed its stacks
# at a seed at all, so every campaign run through the one-call step trained
# without the dropout its settings asked for.  A silent difference in the
# model, not an error: the trajectory still finishes and the binders are
# merely worse, which is the failure mode this directory keeps meeting.
#
# Three questions, because any one alone can pass while the feature is broken:
#
#   1. does dropout=True CHANGE anything?  (if not, the seed never arrives)
#   2. is it REPRODUCIBLE from the same seed?  (the backward replays the
#      forward, and a mask drawn from a counter rather than a hash would make
#      the replay disagree with the pass it is replaying -- a wrong gradient
#      with a healthy-looking loss)
#   3. does a DIFFERENT seed give different masks?  (a seed that is read but
#      ignored passes 1 and 2)
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['BC2_NATIVE_STEP'] = '1'
import bindcraft                      # FIRST: it sets PREALLOCATE=false
import jax, numpy as np
from bindcraft.af2 import AlphaFoldDesignModel as AF2
from bindcraft.protein import Protein, BINDER_CHAIN_PREFIX, target_chain_name
from bindcraft.protein_preparation import merge_receptor_chains
from bindcraft.loss import build_losses

HERE = os.path.dirname(os.path.abspath(__file__))
STRUCT = os.path.join(os.path.dirname(HERE), 'settings', 'target', 'structures')
MODEL = 'model_1_multimer_v3'


def gradient_norm(dropout, key_seed):
    af = AF2(presets=MODEL, data_dir='/home/ubuntu/params',
             key=jax.random.PRNGKey(key_seed), num_recycle=0,
             length_bucket_size=1, dropout=dropout, backend='native')
    tgt = merge_receptor_chains(list(Protein.from_structure(
        os.path.join(STRUCT, 'hPDL1.pdb')).values()))
    binder = Protein.empty(100, jax.random.PRNGKey(1))
    states = {'complex': {BINDER_CHAIN_PREFIX: binder,
                          target_chain_name('target', 'complex'): tgt},
              'binder_alone': {BINDER_CHAIN_PREFIX: binder}}
    losses = build_losses({'weights_plddt_loss': 1.0,
                           'weights_sequence_entropy': 0.1,
                           'weights_compactness': 0.1,
                           'weights_interface_contacts': 1.0})
    out = []
    for _ in range(2):
        r = af.sequence_gradients(states, losses, model=MODEL)
        g = jax.tree_util.tree_leaves(r[-1])
        out.append(float(sum((np.asarray(v, np.float64) ** 2).sum()
                             for v in g) ** 0.5))
    return out


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else None
    if arm:
        drop, key_seed = arm == 'on', int(sys.argv[2])
        a, b = gradient_norm(drop, key_seed)
        print(f'RESULT {arm} key={key_seed} {a!r} {b!r}', flush=True)
        return 0
    # ONE ARM PER PROCESS: the step is cached on (nmax, recycles, msa_rows,
    # dropout, weights) and the native hooks are module-level globals, so a
    # second arm here would answer about the first one's object.
    import subprocess, re
    got = {}
    for name, key_seed in (('off', 0), ('on', 0), ('on2', 0), ('on', 7)):
        p = subprocess.run([sys.executable, os.path.abspath(__file__),
                            name.rstrip('2'), str(key_seed)],
                           capture_output=True, text=True)
        m = [l for l in p.stdout.splitlines() if l.startswith('RESULT ')]
        if not m:
            print(f'{name}/{key_seed}: FAILED rc {p.returncode}')
            print(p.stdout[-1500:], p.stderr[-2500:])
            return 1
        got[(name, key_seed)] = [float(x) for x in m[-1].split()[3:]]
        print(f'  {name:<4} key {key_seed}  |grad| '
              + '  '.join(f'{v:.6g}' for v in got[(name, key_seed)]),
              flush=True)

    off = got[('off', 0)]
    on = got[('on', 0)]
    on2 = got[('on2', 0)]
    on7 = got[('on', 7)]
    ok = True
    def check(label, cond):
        nonlocal ok
        print(f'  {"PASS" if cond else "FAIL"}  {label}')
        ok = ok and cond
    check('dropout changes the gradient', abs(on[0] - off[0]) > 1e-6)
    check('same seed, same answer, twice in one process',
          on[0] == on[1] and off[0] == off[1])
    check('same seed, same answer, in a FRESH process', on[0] == on2[0])
    check('a different key draws different masks', abs(on7[0] - on[0]) > 1e-6)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
