# DOES THE FUSED ONE-CALL STEP DESIGN AS WELL AS THE PATH IT REPLACES?
#
#   python native/outcome_ladder.py                       # all arms, all seeds
#   python native/outcome_ladder.py onecall                # one arm, its seeds
#
# WHY A DISTRIBUTION AND NOT A DIFF.  Parity is not available here and never
# will be: at identical parameters the CUDA trunk and XLA's give different
# losses and a nearly uncorrelated sequence gradient, because a bf16 Evoformer
# has no reproducible trajectory to match (4e-03 after one block, 7e-01 after
# forty-eight).  Worse, the native path is nondeterministic against ITSELF --
# two identical runs of the same config landed at i_con 3.69 and 3.81, rg 1.69
# and 2.68.  So a single run per arm measures the draw, not the arm.  The only
# answerable question is where the trajectories END UP, over enough draws that
# the spread is visible.
#
# THE ARMS, in the order the step was built:
#   jax      backend='jax'                       -- XLA does everything
#   native   backend='native'                    -- staged: our kernels, XLA glue
#   onecall  backend='native'                    -- the whole step as one call,
#                                                   which is now the DEFAULT;
#                                                   `native` is the one that
#                                                   has to ask to opt out
#
# ONE ARM PER PROCESS, all of its seeds inside it.  The native hooks are
# module-level globals and g_stacks keys on N with no destructor, so a second
# BACKEND in the same process runs beside the first one's allocations rather
# than instead of them -- which is also why the length ladder forks per length.
# Seeds are not a second configuration: same lengths, same stacks, so they cost
# one model build between them instead of one each.  That build is nine tenths
# of a one-seed process, and paying it per seed turned a forty-minute matrix
# into a three-hour one.
import json, os, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

ARMS = ('jax', 'native', 'onecall')
PDB = os.environ.get('PDB', 'hPDL1.pdb')
BINDER_LEN = int(os.environ.get('BINDER_LEN', '100'))
ITERS = int(os.environ.get('ITERS', '25'))
SEEDS = int(os.environ.get('SEEDS', '6'))
WEIGHTS = {'weights_plddt_loss': 1.0, 'weights_sequence_entropy': 0.1,
           'weights_compactness': 0.1, 'weights_interface_contacts': 1.0}


def one(arm, seeds):
    # THE OPT-OUT MOVED, not the arm.  The fused step is the default since
    # 2026-10-13, so it is the staged arm that now sets a variable -- and
    # this file is the reason that arm still has to be reachable at all.
    if arm == 'native':
        os.environ['BC2_NATIVE_DISABLE'] = 'step'
    import bindcraft                     # FIRST: it sets PREALLOCATE=false
    import jax, jax.numpy as jnp
    from bindcraft.af2 import AlphaFoldDesignModel as AF2
    from bindcraft.protein import Protein, BINDER_CHAIN_PREFIX, target_chain_name
    from bindcraft.protein_preparation import merge_receptor_chains
    from bindcraft.loss import build_losses, weighted_design_loss
    from bindcraft.sequence_optimization import LogitSequenceOptimizer
    from bindcraft.target_schedule import FixedTargetSchedule
    from bindcraft.trajectory import run_gradient_design_stage

    model = 'model_1_multimer_v3'
    af = AF2(presets=model, data_dir='/home/ubuntu/params',
             key=jax.random.PRNGKey(0), num_recycle=0,
             length_bucket_size=1, dropout=False,
             backend='jax' if arm == 'jax' else 'native')
    tgt = merge_receptor_chains(list(Protein.from_structure(
        os.path.join(ROOT, 'settings', 'target', 'structures', PDB)).values()))
    losses = build_losses(WEIGHTS)
    for seed in seeds:
        # THE SEED IS THE STARTING SEQUENCE.  Everything else -- the target,
        # the losses, the schedule -- is held fixed, so the spread across
        # seeds is the spread the arm produces, not the spread of the problem.
        binder = Protein.empty(BINDER_LEN, jax.random.PRNGKey(1000 + seed))
        states = {'complex': {BINDER_CHAIN_PREFIX: binder,
                              target_chain_name('target', 'complex'): tgt},
                  'binder_alone': {BINDER_CHAIN_PREFIX: binder}}
        t = time.perf_counter()
        final_states, predictions = run_gradient_design_stage(
            states, af, losses,
            LogitSequenceOptimizer(ITERS, learning_rate=0.1),
            FixedTargetSchedule(ITERS))
        dt = time.perf_counter() - t

        row = {'arm': arm, 'seed': seed, 'minutes': dt / 60.0,
               'complex': len(tgt) + BINDER_LEN}
        # EVERY loss separately, not just the total: a port can hold the sum
        # while trading plddt for contacts, and the sum would never say so.
        for name, entry in losses.items():
            if entry.required_states <= set(predictions):
                row[name] = float(entry.function(final_states, predictions))
        row['loss'] = float(
            weighted_design_loss(losses, final_states, predictions))
        for state, pred in predictions.items():
            for k, v in pred.metrics.items():
                try:
                    row[f'{state}.{k}'] = float(v)
                except (TypeError, ValueError):
                    pass
        print('ROW ' + json.dumps(row), flush=True)
    return 0


def summarise(rows):
    keys = [k for k in rows[0] if k not in ('arm', 'seed')]
    width = max(len(k) for k in keys) + 2
    print(f'\n{PDB} binder {BINDER_LEN}, {ITERS} iters, {SEEDS} seeds per arm')
    print(f'{"":<{width}}' + ''.join(f'{a:>24}' for a in ARMS))
    print(f'{"":<{width}}' + ''.join(f'{"mean":>11} {"sd":>6} {"best":>6}'
                                     for _ in ARMS))
    for k in keys:
        line = f'{k:<{width}}'
        for arm in ARMS:
            v = [r[k] for r in rows if r['arm'] == arm and k in r]
            if not v:
                line += f'{"-":>24}'
                continue
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            line += f'{statistics.mean(v):11.4f} {sd:6.3f} {min(v):6.3f}'
        print(line)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--arm':
        return one(sys.argv[2], range(SEEDS))
    arms = sys.argv[1].split(',') if len(sys.argv) > 1 else ARMS
    rows = []
    for arm in arms:
        p = subprocess.run([sys.executable, os.path.abspath(__file__),
                            '--arm', arm], capture_output=True, text=True)
        got = [l for l in p.stdout.splitlines() if l.startswith('ROW ')]
        for line in got:
            rows.append(json.loads(line[4:]))
            r = rows[-1]
            print(f'{arm:<8} seed {r["seed"]}  loss {r["loss"]:8.4f}  '
                  f'{r["minutes"]:.1f} min', flush=True)
        if len(got) < SEEDS:
            # A FAILURE IS A RESULT.  Print it; never let a dead arm look like
            # a missing column, and never let a SHORT arm look like a full one.
            print(f'{arm}: {len(got)} of {SEEDS} seeds (rc {p.returncode})',
                  flush=True)
            print(p.stdout[-2000:], p.stderr[-3000:], flush=True)
    if rows:
        summarise(rows)
    return 0


if __name__ == '__main__':
    sys.exit(main())
