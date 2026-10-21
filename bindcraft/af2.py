import copy
import platform
import threading
import time
import hashlib
import os
import warnings
from contextlib import contextmanager
from typing import Callable
import jax
import jax.numpy as jnp
from jax import Array
from bindcraft.af.alphafold.common import confidence, residue_constants
from bindcraft.af.alphafold.model import config as af_config, data as af_data, model as af_model, modules as af_modules
from bindcraft.af import accel
from bindcraft.prediction import DifferentiableProteinPredictor, CompiledModelCache, residue_chain_ids, concatenate_chain_arrays, collect_shared_chains, split_residue_arrays_by_chain
from bindcraft.loss import DesignLoss, chain_residue_slices, frozen_interface_arguments, renamed_loss_name, renamed_state_losses
from bindcraft.sequence_optimization import sequence_features_from_logits
from bindcraft.protein import AMINO_ACIDS, ATOM_INDEX, ATOM_NAMES, BINDER_ALONE, BINDER_CHAIN_PREFIX, StructurePrediction, StructurePredictions, Protein, ResidueFlags, ProteinStates, has_residue_flag, is_binder_chain, is_target_chain, kabsch, real_residue_weights

COMPILE_LOCK_HEARTBEAT_SECONDS = 30.0
COMPILE_LOCK_STALE_SECONDS = 120.0
COMPILE_LOCK_PATIENCE_SECONDS = 600.0

def compile_lock_owner_is_gone(lock_path: str) -> bool:
    try:
        node, pid = open(lock_path).read().split()[:2]
    except (OSError, ValueError):
        return False
    if node != platform.node():
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except (OSError, ValueError):
        return False
    return False

@contextmanager
def one_worker_compiles(compile_shape: tuple, blocking: bool=True):
    cache_directory = os.environ.get('JAX_COMPILATION_CACHE_DIR')
    if not cache_directory:
        yield True
        return
    os.makedirs(cache_directory, exist_ok=True)
    shape_digest = hashlib.sha256(repr(compile_shape).encode()).hexdigest()[:16]
    lock_path = os.path.join(cache_directory, f'compile_{shape_digest}.lock')
    waiting_since = time.time()
    while True:
        try:
            lock_handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if not blocking:
                yield False
                return
            if time.time() - waiting_since > COMPILE_LOCK_PATIENCE_SECONDS:
                print(f'compiling without the lock, {lock_path} has been held for {COMPILE_LOCK_PATIENCE_SECONDS:.0f}s', flush=True)
                yield True
                return
            time.sleep(1.0)
            try:
                if compile_lock_owner_is_gone(lock_path) or time.time() - os.path.getmtime(lock_path) > COMPILE_LOCK_STALE_SECONDS:
                    os.replace(lock_path, f'{lock_path}.stale')
                    os.unlink(f'{lock_path}.stale')
            except OSError:
                pass
    os.write(lock_handle, f'{platform.node()} {os.getpid()}\n'.encode())
    finished = threading.Event()
    def keep_the_lock_alive() -> None:
        while not finished.wait(COMPILE_LOCK_HEARTBEAT_SECONDS):
            try:
                os.utime(lock_path, None)
            except FileNotFoundError:
                return
    heartbeat = threading.Thread(target=keep_the_lock_alive, daemon=True)
    heartbeat.start()
    try:
        yield True
    finally:
        finished.set()
        os.close(lock_handle)
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass

def align_prediction_to_target_template(predicted_atom_positions: Array, predicted_atom_mask: Array, template_atom_positions: Array, template_atom_mask: Array, flags: Array) -> Array:
    ca_atom_index = ATOM_INDEX['CA']
    template_residue_mask = has_residue_flag(flags, ResidueFlags.TEMPLATE) & ~has_residue_flag(flags, ResidueFlags.DESIGN)
    alignment_weights = (template_residue_mask & template_atom_mask[:, ca_atom_index]).astype(jnp.float32)
    predicted_ca_coordinates = jax.lax.stop_gradient(predicted_atom_positions[:, ca_atom_index]).astype(jnp.float32)
    template_ca_coordinates = template_atom_positions[:, ca_atom_index].astype(jnp.float32)
    alignment_rotation, prediction_center, template_center = kabsch(predicted_ca_coordinates, template_ca_coordinates, alignment_weights)
    aligned_atom_positions = (predicted_atom_positions.astype(jnp.float32) - prediction_center) @ alignment_rotation.T + template_center
    return jnp.where(predicted_atom_mask[:, :, None], aligned_atom_positions, 0.0).astype(predicted_atom_positions.dtype)

def prepare_design_sequence_features(sequence: Array, flags: Array, softmax_weight: Array, one_hot_weight: Array, temperature: Array, logit_scale: Array, amino_acid_bias: Array | None=None) -> tuple[Array, Array]:
    designed_residue_mask = has_residue_flag(flags, ResidueFlags.DESIGN)[:, None]
    sequence_features = jnp.where(designed_residue_mask, sequence_features_from_logits(sequence, softmax_weight, one_hot_weight, temperature, logit_scale, amino_acid_bias), sequence)
    one_hot_weight_value = jnp.ones((), dtype=jnp.result_type(softmax_weight))
    sequence_profile = jnp.where(designed_residue_mask, sequence_features_from_logits(sequence, softmax_weight, one_hot_weight_value, temperature, logit_scale, amino_acid_bias), sequence)
    return sequence_features, sequence_profile

DEFAULT_LENGTH_BUCKET = 32

def campaign_length_bucket(settings: dict) -> int:
    return int(settings.get('length_bucket_size') or DEFAULT_LENGTH_BUCKET)

def padded_prediction_length(residue_count: int, bucket_size: int=DEFAULT_LENGTH_BUCKET) -> int:
    return -(-residue_count // bucket_size) * bucket_size

def padded_prediction_complex(protein_complex: dict[str, Protein], bucket_size: int=DEFAULT_LENGTH_BUCKET, target_pad_length: int=0) -> dict[str, Protein]:
    return {chain_name: protein.padded_to(padded_prediction_length(len(protein), bucket_size) if bool(has_residue_flag(protein.flags, ResidueFlags.DESIGN).any()) else target_pad_length) for chain_name, protein in protein_complex.items()}

def pad_design_chains(protein_states: ProteinStates, bucket_size: int=DEFAULT_LENGTH_BUCKET, target_pad_length: int=0) -> ProteinStates:
    return {state_name: padded_prediction_complex(protein_complex, bucket_size, target_pad_length) for state_name, protein_complex in protein_states.items()}

def shared_target_pad_length(target_states: ProteinStates, target_chain: str='target', bucket_size: int=DEFAULT_LENGTH_BUCKET) -> int:
    target_lengths = [len(protein) for protein_complex in target_states.values() for name, protein in protein_complex.items() if is_target_chain(name, target_chain)]
    return padded_prediction_length(max(target_lengths), bucket_size) if len(set(target_lengths)) > 1 else 0

def canonical_state_names(protein_states: ProteinStates) -> dict[str, str]:
    return {name: name if name == BINDER_ALONE else f'state_{index}' for index, name in enumerate(sorted(protein_states))}

def canonical_chain_names(protein_states: ProteinStates, state_names: dict[str, str]) -> dict[str, str]:
    chain_names = {}
    for state_name, protein_complex in protein_states.items():
        suffix = f'_{state_name}'
        for chain_name in protein_complex:
            chain_names[chain_name] = f'{chain_name[:-len(suffix)]}_{state_names[state_name]}' if chain_name.endswith(suffix) else chain_name
    return chain_names

def renamed_protein_states(protein_states: ProteinStates, state_names: dict[str, str], chain_names: dict[str, str]) -> ProteinStates:
    return {state_names[state_name]: {chain_names[chain_name]: protein for chain_name, protein in protein_complex.items()} for state_name, protein_complex in protein_states.items()}

def real_residue_positions(chain_lengths: tuple[int, ...], true_lengths: tuple[int, ...]) -> Array:
    positions, chain_start = [], 0
    for chain_length, true_length in zip(chain_lengths, true_lengths):
        positions.append(jnp.arange(chain_start, chain_start + true_length, dtype=jnp.int32))
        chain_start += chain_length
    return jnp.concatenate(positions)

def trim_padded_metrics(metrics: dict[str, Array], real_positions: Array) -> dict[str, Array]:
    return {name: value if value.ndim == 0 else value[real_positions] if value.ndim == 1 else value[real_positions[:, None], real_positions[None, :]] for name, value in metrics.items()}

SIDECHAIN_ATOMS = tuple(atom_name not in ('N', 'CA', 'C', 'CB', 'O') for atom_name in ATOM_NAMES)

def flexible_target_residues(residue_index: Array, target_template_mask: Array, target_flexibility: float) -> Array:
    return target_template_mask & (jnp.floor((residue_index + 1) * target_flexibility) > jnp.floor(residue_index * target_flexibility))

def target_template_features(aatype: Array, atoms: Array, atom_mask: Array, flags: Array, residue_index: Array, target_flexibility: float=0.0) -> dict[str, Array]:
    template_residue_mask = has_residue_flag(flags, ResidueFlags.TEMPLATE)
    flexible_mask = flexible_target_residues(residue_index, template_residue_mask & ~has_residue_flag(flags, ResidueFlags.DESIGN), target_flexibility) if target_flexibility else jnp.zeros_like(template_residue_mask)
    template_aatype = jnp.where(template_residue_mask & ~flexible_mask, aatype, 21)
    template_all_atom_mask = atom_mask & template_residue_mask[:, None] & ~(flexible_mask[:, None] & jnp.asarray(SIDECHAIN_ATOMS))
    template_all_atom_positions = jnp.where(template_residue_mask[:, None, None], atoms, 0.0)
    template_beta_coordinates, template_beta_mask = af_modules.pseudo_beta_fn(template_aatype, template_all_atom_positions, template_all_atom_mask)
    return {'template_aatype': template_aatype[None].astype(jnp.int32), 'template_all_atom_positions': template_all_atom_positions[None].astype(jnp.float32), 'template_all_atom_mask': template_all_atom_mask[None].astype(jnp.float32), 'template_pseudo_beta': template_beta_coordinates[None].astype(jnp.float32), 'template_pseudo_beta_mask': template_beta_mask[None].astype(jnp.float32), 'template_mask': jnp.ones((1,), dtype=jnp.float32), 'mask_template_interchain': jnp.asarray(True)}

def residue_entity_ids(chain_names: tuple[str, ...], chain_lengths: tuple[int, ...], multi_chain_binders: tuple[tuple[str, ...], ...]=()) -> Array:
    shared_entity = {name: min(group) for group in multi_chain_binders for name in group}
    entity_names = sorted({shared_entity.get(name, name) for name in chain_names})
    return jnp.concatenate([jnp.full((length,), entity_names.index(shared_entity.get(name, name)), dtype=jnp.int32) for name, length in zip(chain_names, chain_lengths)])

def interface_asym_ids(chain_names: tuple[str, ...], chain_lengths: tuple[int, ...]) -> Array:
    assembly_name = lambda chain_name: BINDER_CHAIN_PREFIX if is_binder_chain(chain_name) else chain_name
    assembly_names = sorted({assembly_name(name) for name in chain_names})
    if len(assembly_names) < 2:
        return residue_chain_ids(chain_lengths)
    return jnp.concatenate([jnp.full((length,), assembly_names.index(assembly_name(name)), dtype=jnp.int32) for name, length in zip(chain_names, chain_lengths)])

def cyclic_sequence_offsets(residue_index: Array, asym_id: Array, flags: Array, seq_mask: Array, offset_mode: str='direction') -> Array:
    offsets = residue_index[:, None] - residue_index[None, :]
    same_chain = asym_id[:, None] == asym_id[None, :]
    chain_lengths = (same_chain * seq_mask[None, :]).sum(-1)
    cyclic_mask = has_residue_flag(flags, ResidueFlags.CYCLIC) * seq_mask
    cyclic_pairs = same_chain * cyclic_mask[:, None] * cyclic_mask[None, :]
    cyclic_distance = jnp.minimum(jnp.abs(offsets), chain_lengths[:, None] - jnp.abs(offsets))
    if offset_mode in ('direction', 'neighbours'):
        cyclic_distance = jnp.where(cyclic_distance < jnp.abs(offsets), -cyclic_distance, cyclic_distance)
    if offset_mode == 'neighbours':
        cyclic_distance = jnp.where(jnp.abs(cyclic_distance) > 2, 32 * jnp.sign(cyclic_distance), cyclic_distance)
    return jnp.where(cyclic_pairs, cyclic_distance * jnp.sign(offsets), offsets).astype(jnp.int32)

def alphafold_input_features(sequence: Array, sequence_profile: Array, atoms: Array, atom_mask: Array, residue_index: Array, asym_id: Array, seq_mask: Array, flags: Array, dropout: Array, cyclic_offset_mode: str='direction', entity_id: Array | None=None, target_flexibility: float=0.0, large_complex_initialization: bool=False, template_act_const: Array | None=None) -> dict[str, Array]:
    length = sequence.shape[0]
    aatype = sequence.argmax(-1).astype(jnp.int32)
    mask = seq_mask[:, None]
    entity_id = asym_id if entity_id is None else entity_id
    msa_feat = jnp.zeros((1, length, 49)).at[:, :, 0:20].set(sequence[None]).at[:, :, 25:45].set(sequence_profile[None])
    return {'aatype': aatype, 'residue_index': residue_index, 'offset': cyclic_sequence_offsets(residue_index, asym_id, flags, seq_mask, cyclic_offset_mode), 'asym_id': asym_id, 'entity_id': entity_id, 'sym_id': asym_id, 'seq_mask': seq_mask, 'msa_mask': jnp.ones((1, length)) * seq_mask[None], 'target_feat': sequence, 'msa_feat': msa_feat, 'extra_msa': jnp.zeros((1, length), dtype=jnp.int32), 'extra_msa_mask': jnp.zeros((1, length), dtype=jnp.float32), 'extra_has_deletion': jnp.zeros((1, length), dtype=jnp.float32), 'extra_deletion_value': jnp.zeros((1, length)), 'all_atom_positions': atoms.astype(jnp.float32), 'all_atom_mask': atom_mask.astype(jnp.float32), 'atom14_atom_exists': jnp.where(mask, jnp.asarray(residue_constants.restype_atom14_mask)[aatype], 0), 'atom37_atom_exists': jnp.where(mask, jnp.asarray(residue_constants.restype_atom37_mask)[aatype], 0), 'residx_atom14_to_atom37': jnp.where(mask, jnp.asarray(residue_constants.restype_atom14_to_atom37)[aatype], 0), 'residx_atom37_to_atom14': jnp.where(mask, jnp.asarray(residue_constants.restype_atom37_to_atom14)[aatype], 0), 'use_dropout': dropout, 'prev': {'prev_msa_first_row': jnp.zeros((length, 256)), 'prev_pair': jnp.zeros((length, length, 128)), 'prev_pos': jnp.zeros((length, 37, 3))}, **({'initial_atom_pos': atoms.astype(jnp.float32)} if large_complex_initialization else {}), **target_template_features(aatype, atoms, atom_mask, flags, residue_index, target_flexibility), **({'template_act_const': template_act_const} if template_act_const is not None else {})}

def recycled_alphafold_outputs(alphafold_runner: af_model.RunModel, model_parameters: Array, key: Array, model_inputs: dict, num_recycle: int) -> dict:
    recycle_keys = jax.random.split(key, num_recycle + 1)
    previous_state = model_inputs['prev']
    for recycle_key in recycle_keys[:-1]:
        previous_state = jax.lax.stop_gradient(alphafold_runner.apply(model_parameters, recycle_key, {**model_inputs, 'prev': previous_state})['prev'])
        if 'initial_atom_pos' in model_inputs:
            model_inputs = {**model_inputs, 'initial_atom_pos': previous_state['prev_pos']}
    return alphafold_runner.apply(model_parameters, recycle_keys[-1], {**model_inputs, 'prev': previous_state})

def alphafold_prediction_metrics(alphafold_outputs: dict, seq_mask: Array, interface_asym_id: Array) -> dict[str, Array]:
    metrics: dict[str, Array] = {}
    if 'predicted_lddt' in alphafold_outputs:
        confidence_logits = alphafold_outputs['predicted_lddt']['logits']
        bin_width = 1.0 / confidence_logits.shape[-1]
        bin_centers = jnp.arange(0.5 * bin_width, 1.0, bin_width)
        metrics['plddt'] = (jax.nn.softmax(confidence_logits, axis=-1) * bin_centers[None, :]).sum(-1)
    if 'predicted_aligned_error' in alphafold_outputs:
        pae_head = alphafold_outputs['predicted_aligned_error']
        probabilities = jax.nn.softmax(pae_head['logits'], axis=-1)
        pae_bin_edges = pae_head['breaks']
        pae_bin_width = pae_bin_edges[1] - pae_bin_edges[0]
        bin_centers = jnp.append(pae_bin_edges + pae_bin_width / 2, pae_bin_edges[-1] + 1.5 * pae_bin_width)
        pae = (probabilities * bin_centers).sum(-1)
        metrics['pae'] = (pae + pae.T) / 2
        metrics['ptm'] = confidence.predicted_tm_score(pae_head['logits'], pae_head['breaks'], residue_weights=seq_mask, use_jnp=True)
        metrics['iptm'], metrics['iptm_per_residue'] = confidence.predicted_tm_score(pae_head['logits'], pae_head['breaks'], residue_weights=seq_mask, asym_id=interface_asym_id, use_jnp=True, return_per_alignment=True)
    if 'experimentally_resolved' in alphafold_outputs:
        metrics['experimentally_resolved_ca'] = jax.nn.sigmoid(alphafold_outputs['experimentally_resolved']['logits'][:, ATOM_INDEX['CA']])
    if 'distogram' in alphafold_outputs:
        metrics['distogram'] = alphafold_outputs['distogram']['logits']
    return metrics

def trim_prediction_padding(value: Array, residue_count: int) -> Array:
    if value.ndim == 0:
        return value
    if value.ndim == 1:
        return value[:residue_count]
    return value[:residue_count, :residue_count]

def reference_state_confidence(prediction: StructurePrediction, chain_names: dict[str, str], shared_chains: dict[str, Protein], state_chain_names: tuple[str, ...]) -> Array:
    confidence_values = jnp.asarray(prediction.metrics['plddt'], dtype=jnp.float32)
    per_chain = {chain_names.get(name, name): confidence_values[chain_slice] for name, chain_slice in chain_residue_slices(prediction.protein_complex).items()}
    return jnp.concatenate([jnp.pad(per_chain[name], (0, len(shared_chains[name]) - len(per_chain[name]))) for name in state_chain_names])

def protein_state_shapes(protein_states: ProteinStates) -> tuple[tuple[str, tuple[str, ...], tuple[int, ...]], ...]:
    complex_shapes = []
    for state_name in sorted(protein_states):
        chain_names = tuple(sorted(protein_states[state_name]))
        chain_lengths = tuple(len(protein_states[state_name][chain_name]) for chain_name in chain_names)
        complex_shapes.append((state_name, chain_names, chain_lengths))
    return tuple(complex_shapes)

MONOMER_CHAIN_GAP = 49

def monomer_chain_break_indices(chain_lengths: tuple[int, ...], residue_index: Array) -> Array:
    """Renumber a complex for the monomer models, holding the chains apart without closing the breaks inside them.

    merge_receptor_chains fuses a multi-chain target into one chain and keeps its receptor chains apart by a
    gap in the residue numbering; numbering the chain straight through would hand the models a peptide bond
    that is not there, and they fold the receptor chains into one another."""
    steps = jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), jnp.diff(residue_index)])
    chain_starts = jnp.cumsum(jnp.asarray(chain_lengths[:-1], dtype=jnp.int32))
    return jnp.cumsum(steps.at[chain_starts].set(MONOMER_CHAIN_GAP + 1))

def alphafold_model_family(model_name: str) -> tuple:
    if 'multimer' in model_name:
        return ('multimer',)
    return 'monomer', tuple(sorted(af_config.CONFIG_DIFFS.get(model_name, {}).items()))

ATTENTION_FALLBACK_BACKEND = 'chunked'
GRADIENT_MEMORY_SHARE = 0.80
BYTES_PER_GIB = 1 << 30

def worker_memory_budget_bytes() -> float:
    from bindcraft.design_workers import design_gpu_memory_gb, visible_design_gpus
    fraction = float(os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION') or 0.75)
    memory = design_gpu_memory_gb()
    return fraction * min((memory[gpu][1] for gpu in visible_design_gpus() if gpu in memory), default=0.0) * BYTES_PER_GIB

def executable_claimed_bytes(executable) -> float:
    analysis = executable.memory_analysis()
    return float(analysis.temp_size_in_bytes + analysis.argument_size_in_bytes + analysis.output_size_in_bytes)

def resident_design_bytes() -> float:
    try:
        return float((jax.devices()[0].memory_stats() or {}).get('bytes_in_use', 0))
    except Exception:
        return 0.0

def executable_fits(executable) -> bool:
    budget = worker_memory_budget_bytes()
    return not budget or executable_claimed_bytes(executable) + resident_design_bytes() <= GRADIENT_MEMORY_SHARE * budget

def is_out_of_memory(failure: BaseException) -> bool:
    text = str(failure).lower()
    return 'resource_exhausted' in text or 'out of memory' in text

SUBBATCH_RESIDUE_THRESHOLD = 384
LARGE_COMPLEX_SUBBATCH_SIZE = 256
SUBBATCH_LADDER = (None, 256, 128, 64)

def resolve_subbatch_size(residue_count: int, subbatch_size: int | None | str='auto') -> int | None:
    if subbatch_size != 'auto':
        return subbatch_size
    return LARGE_COMPLEX_SUBBATCH_SIZE if residue_count > SUBBATCH_RESIDUE_THRESHOLD else None

def _native_one_object() -> bool:
    """One native DesignStep for every state, run at each state's own
    length.  ON by default; NB_ONE_OBJECT=0 opts out and restores the
    per-bucket objects and the BC2_NATIVE_STEP_SHARE_N threshold.  See
    design_step.cuh's setN."""
    return os.environ.get('NB_ONE_OBJECT', '1') not in ('0', '')


def _native_step_off() -> bool:
    """BC2_NATIVE_DISABLE=step: run the staged native path, not the fused one."""
    from .af import native as _bcn
    return 'step' in _bcn.disabled_pieces()


def _native_predict_on() -> bool:
    """Run predict() on the fused native step, with no backward behind it.

    ON by default wherever the fused design step is; BC2_NATIVE_PREDICT=0
    opts out.  Three measurements put it there, on an A10, hIL2R_beta_gamma
    with a binder, model_1_multimer_v3, 0 recycles:

      * EXACT against the design step's own forward -- both handlers, the
        same inputs, all eight outputs at max|d| 0 at N=64 and N=608
        (ds_predict_gate.py).  So the kernels are not in question.
      * 2.4-2.7x faster than the JAX predict it replaces (complex 487:
        1454 ms against 3522; complex 607: 2120 against 5685), and 4x on
        the first call, 7.7 s against 25.9.  predict() feeds the *_final
        filters and the MPNN stage's reprediction, which are many forwards
        and no backward, so this is the price of half a campaign.
      * the same STRUCTURE wherever the model has an opinion.  Against the
        JAX path on a real 427-residue complex the 386 residues at pLDDT
        >= 70 agree to 0.013-0.017 A; the whole of the 0.148 A mean is the
        40 residues below pLDDT 50, which are the random binder the model
        has no opinion about (predict_seam_gate.py).  That is the check
        that mattered, because the two paths differ in WHERE they align --
        the native step aligns inside (kabsch.cuh) and bc2's predict path
        outside -- and a wrong convention would show in the coordinates and
        in no metric.

    WHAT IT COSTS, and it is one case only.  The DesignStep this calls is
    still sized for a BACKWARD, so a process that only predicts holds ~9.6
    GB more than the JAX path would (complex 607: 10.16 GB free against
    19.76).  In a campaign that is free, because the design step has
    already built that object and a predict at the same length SHARES it --
    same DsKey, one allocation.  A predict-only process is the case to set
    BC2_NATIVE_PREDICT=0 for, until the allocation itself learns to skip
    what only a backward reads.
    """
    return os.environ.get('BC2_NATIVE_PREDICT', '1') not in ('', '0')


class AlphaFoldDesignModel(DifferentiableProteinPredictor):
    def __init__(self, presets: str | tuple[str, ...]='model_1_ptm', data_dir: str | None=None, key: Array | None=None, max_cache_size: int=8, models: tuple[str, ...] | None=None, num_recycle: int=1, cyclic_offset_mode: str='direction', subbatch_size: int | None | str='auto', length_bucket_size: int=DEFAULT_LENGTH_BUCKET, attention_backend: str='auto', use_cueq: bool=False, use_remat: bool=True, backend: str='auto', evoformer_backend: str | None=None, dropout: bool=True, multi_chain_binders: tuple[tuple[str, ...], ...]=(), target_pad_length: int=0, target_flexibility: float=0.0, large_complex_initialization: bool=False, amino_acid_bias: dict[str, float] | None=None):
        self.cyclic_offset_mode = cyclic_offset_mode
        self.target_pad_length = target_pad_length
        self.multi_chain_binders = multi_chain_binders
        self.presets = (presets,) if isinstance(presets, str) else tuple(presets)
        self.models = self.presets if models is None else tuple(models)
        self.key = jax.random.PRNGKey(0) if key is None else key
        self.dropout = dropout

        # BC2_NATIVE forces either way for a dev run: =1 native, =0 jax.
        # `evoformer_backend` WAS THE OLD NAME and is honoured, because it
        # means exactly what `backend` means -- unlike the retired
        # BC2_NATIVE_* flags, where the defaults had flipped and obeying an
        # old setting would have given the opposite of what it asked for.
        # The name was wrong anyway: it selects the implementation for the
        # embedder, both stacks, the template embedding and the post-trunk
        # step as well, not just the evoformer.
        if evoformer_backend is not None:
            if backend != 'auto':
                raise ValueError(
                    'pass backend= or evoformer_backend=, not both; '
                    'evoformer_backend is the old name for backend')
            warnings.warn('evoformer_backend is now called backend; it '
                          'selects more than the evoformer',
                          DeprecationWarning, stacklevel=2)
            backend = evoformer_backend
        # BC2_NATIVE forces either way for a dev run: =1 native, =0 jax.
        _env = os.environ.get('BC2_NATIVE')
        if _env is not None:
            backend = 'native' if _env == '1' else 'jax'
        if backend not in ('auto', 'native', 'jax'):
            raise ValueError(
                f'backend must be auto, native or jax, not {backend!r}')
        self.backend = backend
        self.evoformer_backend = backend      # the old attribute name
        # 'native' INSISTS: it is a request, so a card or build that cannot
        # serve it should say so rather than quietly run something else.
        self.native_evoformer = backend in ('auto', 'native')
        self.native_installed = False
        #for the desperation autotuner only
        self.num_recycle = num_recycle
        self.target_flexibility = target_flexibility
        self.large_complex_initialization = large_complex_initialization
        #one row of log-odds read off the settings, not an array carried beside every chain
        self.amino_acid_bias = None if not amino_acid_bias else jnp.asarray([amino_acid_bias.get(amino_acid, 0.0) for amino_acid in AMINO_ACIDS], dtype=jnp.float32)
        self.subbatch_size = subbatch_size
        self.length_bucket_size = length_bucket_size
        self.attention_backend = accel.supported_attention_backend(attention_backend)
        self.use_cueq = use_cueq
        self.use_remat = use_remat
        self.model_families: dict[str, tuple] = {}
        self.model_family_models: dict[tuple, str] = {}
        self.alphafold_runners: dict[tuple, af_model.RunModel] = {}
        self.model_parameters: dict[str, object] = {}
        for model_name in self.presets:
            model_family = alphafold_model_family(model_name)
            self.model_families[model_name] = model_family
            self.model_family_models.setdefault(model_family, model_name)
            self._alphafold_runner(model_family, resolve_subbatch_size(4, self.subbatch_size))
            model_parameters = af_data.get_model_haiku_params(model_name=model_name, data_dir=data_dir, fuse=True) if data_dir else None
            if model_parameters is None:
                raise ValueError(f'no AlphaFold parameters for {model_name!r} in data_dir {data_dir!r}; run "bindcraft fetch-weights" to download them')
            self.model_parameters[model_name] = model_parameters
        # NATIVE CUDA TRUNK, selected by `evoformer_backend` (auto by
        # default, or native/jax).  The weights come from
        # the param tree just loaded, so no conversion step and no second copy
        # of the weights -- and a checkpoint this repo has never seen works
        # unchanged.  Only the Evoformer is replaced: it is ~99% of a design
        # step, while the losses cost 0.18 ms and 2.25 s of compile, so they
        # stay in JAX where a new term is a Python edit.
        if self.native_evoformer and self.model_parameters:
            from .af import native
            # EVERY model in the pool gets its own weights, because
            # _sample_design_model picks one per step and a single installed
            # directory would serve model_1's weights for all of them.
            # dropout decides only whether the native stack ALLOCATES its
            # mask scratch.  Whether the masks apply is traced through the
            # batch like stock's, because bindcraft flips it between stages
            # of one trajectory and a static flag would recompile there.
            installed = native.install_models(self.model_parameters, dropout=bool(dropout))
            # THE ONE-CALL STEP NEEDS THIS, and `installed` was a local.  A
            # campaign that asked for backend=auto on a card without the
            # kernels falls back to JAX, and the fused step must fall back
            # with it rather than reach for handlers that were never
            # registered.
            self.native_installed = bool(installed)
            if not installed and self.evoformer_backend == 'native':
                raise RuntimeError(
                    'evoformer_backend=native was asked for but the kernels '
                    f'cannot run here: {native.why_unavailable()}. Use '
                    "evoformer_backend=auto to fall back to JAX silently.")
        self.prediction_compile_cache = CompiledModelCache(max_cache_size)
        self.gradient_compile_cache = CompiledModelCache(max_cache_size)
        self.shape_attention_backend: dict = {}
        self.shape_subbatch_size: dict = {}
        self.executed_lengths: set[int] = set()
        self.last_attention_backend = self.attention_backend
        self.compiled_shapes: set = set()

    def subbatch_ladder(self) -> tuple:
        return SUBBATCH_LADDER if self.subbatch_size == 'auto' else (self.subbatch_size,)
    
    def _alphafold_runner(self, model_family: tuple, subbatch_size: int | None, attention_backend: str | None=None) -> af_model.RunModel:
        attention_backend = attention_backend or self.attention_backend
        runner_key = model_family, subbatch_size, attention_backend, self.use_cueq, self.use_remat
        alphafold_runner = self.alphafold_runners.get(runner_key)
        if alphafold_runner is None:
            model_name = self.model_family_models[model_family]
            use_multimer = 'multimer' in model_name
            model_config = copy.deepcopy(af_config.model_config(model_name))
            model_config.model.global_config.use_dgram = False
            model_config.model.global_config.use_remat = self.use_remat
            model_config.model.global_config.bfloat16 = True
            model_config.model.global_config.subbatch_size = subbatch_size
            model_config.model.global_config.attention_backend = attention_backend
            model_config.model.global_config.use_cueq = self.use_cueq
            model_config.model.num_recycle = 0
            alphafold_runner = af_model.RunModel(model_config, params=None, use_multimer=use_multimer)
            self.alphafold_runners[runner_key] = alphafold_runner
        return alphafold_runner

    def _sample_design_model(self) -> str:
        self.key, model_random_key = jax.random.split(self.key)
        return self.models[int(jax.random.randint(model_random_key, (), 0, len(self.models)))]

    def _resolve_model_name(self, model: str | None) -> str:
        if model is None:
            return self._sample_design_model()
        if model not in self.model_families:
            raise KeyError(f'model {model!r} not in the AlphaFold model pool {self.presets}')
        return model

    def _compiled_complex_prediction(self, model: str, padded_length: int) -> Callable:
        model_family = self.model_families[model]
        subbatch_size = resolve_subbatch_size(padded_length, self.subbatch_size)
        cache_key = model_family, padded_length, subbatch_size, self.multi_chain_binders, self.num_recycle, self.target_flexibility, self.large_complex_initialization
        compiled_prediction = self.prediction_compile_cache.get(cache_key)
        if compiled_prediction is None:
            alphafold_runner = self._alphafold_runner(model_family, subbatch_size)
            def predict_complex_arrays(model_parameters: Array, key: Array, sequence: Array, atoms: Array, atom_mask: Array, residue_index: Array, asym_id: Array, entity_id: Array, interface_asym_id: Array, seq_mask: Array, flags: Array, dropout: Array, softmax_weight: Array, one_hot_weight: Array, temperature: Array, logit_scale: Array):
                sequence_features, sequence_profile = prepare_design_sequence_features(sequence, flags, softmax_weight, one_hot_weight, temperature, logit_scale, self.amino_acid_bias)
                model_inputs = alphafold_input_features(sequence_features, sequence_profile, atoms, atom_mask, residue_index, asym_id, seq_mask, flags, dropout, self.cyclic_offset_mode, entity_id, self.target_flexibility, self.large_complex_initialization)
                alphafold_outputs = recycled_alphafold_outputs(alphafold_runner, model_parameters, key, model_inputs, self.num_recycle)
                predicted_atom_positions = alphafold_outputs['structure_module']['final_atom_positions'].astype(jnp.float16)
                predicted_atom_mask = alphafold_outputs['structure_module']['final_atom_mask'].astype(bool)
                return predicted_atom_positions, predicted_atom_mask, alphafold_prediction_metrics(alphafold_outputs, seq_mask, interface_asym_id)
            compiled_prediction = jax.jit(predict_complex_arrays)
            self.prediction_compile_cache.set(cache_key, compiled_prediction)
        return compiled_prediction

    def predict(self, protein_states: ProteinStates, model: str | None=None, softmax_weight: float=1.0, one_hot_weight: float=1.0, temperature: float=0.01, logit_scale: float=2.0, compile_only: bool=False, compile_blocking: bool=True) -> StructurePredictions:
        model = self._resolve_model_name(model)
        # DECLARE THE BIGGEST LENGTH BEFORE ANYTHING BUILDS, exactly as the
        # gradient path does.  With one DesignStep per process (NB_ONE_OBJECT)
        # the object is sized by whatever allocates FIRST and setN only ever
        # runs it shorter -- so a predict over several states, or a predict
        # before a design, would otherwise size it on whichever state came
        # first in dict order and kill the process on the next longer one:
        # `design_step: setN(544) past the allocation's 512`, exit(1).
        # States iterate in dict order, which says nothing about length.
        if self.native_installed and not _native_step_off() and _native_predict_on():
            from .af.native_step import pad_to_tile as _pad_to, set_alloc_floor
            # AND THE FLOOR IS THE DESIGN PATH'S LENGTH, not this one's.
            # The two pad differently: sequence_gradients pads each DESIGN
            # CHAIN to its own bucket (a 100-mer binder becomes 128) while
            # predict pads the TOTAL, so the same complex is 544 there and
            # 512 here.  With one object per process the FIRST allocation
            # wins and setN only runs shorter -- so a predict before a
            # design sized it at 512 and the design died, not with an
            # out-of-memory but with `setN(544) past the allocation's 512`,
            # exit(1).  Allocating at the design's length and running at
            # this one costs the gap and nothing else.
            set_alloc_floor(_pad_to(max(
                sum(len(c) for c in padded.values()) for padded in
                pad_design_chains(protein_states, self.length_bucket_size,
                                  self.target_pad_length).values())))
        return {name: self._predict_complex(protein_complex, model, softmax_weight, one_hot_weight, temperature, logit_scale, compile_only, compile_blocking) for name, protein_complex in protein_states.items()}

    def _predict_complex(self, protein_complex: dict[str, Protein], model: str, softmax_weight: float, one_hot_weight: float, temperature: float, logit_scale: float, compile_only: bool=False, compile_blocking: bool=True) -> StructurePrediction:
        chain_names = tuple(sorted(protein_complex))
        true_lengths = tuple(len(protein_complex[name]) for name in chain_names)
        padded_complex = padded_prediction_complex(protein_complex, self.length_bucket_size, self.target_pad_length) if self.target_pad_length else protein_complex
        chain_lengths = tuple(len(padded_complex[name]) for name in chain_names)
        residue_count = sum(chain_lengths)
        padded_residue_count = padded_prediction_length(residue_count, self.length_bucket_size)
        padding_length = padded_residue_count - residue_count
        chain_arrays = concatenate_chain_arrays(chain_names, padded_complex, 'sequence', 'atoms', 'atom_mask', 'flags', 'residue_index')
        sequence, atoms, atom_mask, flags, residue_index = chain_arrays['sequence'], chain_arrays['atoms'], chain_arrays['atom_mask'], chain_arrays['flags'], chain_arrays['residue_index']
        asym_id = residue_chain_ids(chain_lengths)
        entity_id = residue_entity_ids(chain_names, chain_lengths, self.multi_chain_binders)
        interface_asym_id = interface_asym_ids(chain_names, chain_lengths)
        if self.model_families[model][0] == 'monomer' and len(chain_names) > 1:
            residue_index = monomer_chain_break_indices(chain_lengths, residue_index)
        seq_mask = real_residue_weights(flags)
        if padding_length:
            sequence = jnp.pad(sequence, [[0, padding_length], [0, 0]])
            atoms = jnp.pad(atoms, [[0, padding_length], [0, 0], [0, 0]])
            atom_mask = jnp.pad(atom_mask, [[0, padding_length], [0, 0]])
            flags = jnp.pad(flags, [0, padding_length])
            residue_index = jnp.pad(residue_index, [0, padding_length])
            asym_id = jnp.pad(asym_id, [0, padding_length], constant_values=len(chain_names))
            entity_id = jnp.pad(entity_id, [0, padding_length], constant_values=len(chain_names))
            interface_asym_id = jnp.pad(interface_asym_id, [0, padding_length], constant_values=len(chain_names))
            seq_mask = jnp.pad(seq_mask, [0, padding_length])
        native_predict = self.native_installed and not _native_step_off() and _native_predict_on()
        if native_predict:
            # upstream's `compile_only` warms a shape for the XLA path; the
            # native step has no such shape to warm, so a warm-up call here
            # returns before running anything -- and does not compile an XLA
            # prediction nothing will ever call
            if compile_only:
                return StructurePrediction(protein_complex=protein_complex, metrics={})
            # THE WHOLE PREDICTION AS ONE CALL, with no backward behind it.
            # Same handler family as the gradient path's, same DsKey, so a
            # predict and a design at the same length share one DesignStep
            # and one allocation -- and this one hands XLA none of the four
            # carry buffers, which are its largest.
            from .af.native_step import pad_to_tile as _pad_to, predict_step as _predict
            from .af import native as _bcn
            _n = padded_residue_count
            _nmax = _pad_to(_n)
            _dn = _nmax - _n
            _pad_id = len(chain_names)
            _p1 = lambda a, v=0: jnp.pad(a, [0, _dn], constant_values=v) if _dn else a
            _p2 = lambda a, v=0: (jnp.pad(a, [[0, _dn]] + [[0, 0]] * (a.ndim - 1), constant_values=v) if _dn else a)
            # act_const is harvested OUTSIDE the trace, exactly as
            # sequence_gradients does it -- and from the same aatype, which
            # is prepare_design_sequence_features' argmax and not the raw
            # sequence's: the softmax, bias and temperature move it at some
            # template positions.
            _seqf, _ = prepare_design_sequence_features(_p2(sequence), _p1(flags), jnp.asarray(softmax_weight), jnp.asarray(one_hot_weight), jnp.asarray(temperature), jnp.asarray(logit_scale), self.amino_acid_bias)
            # the MASK stays bool here: target_template_features ANDs it, and
            # a float32 mask is a TypeError from inside jnp, not from here
            _tf = target_template_features(_seqf.argmax(-1).astype(jnp.int32), _p2(jnp.asarray(atoms, jnp.float32)), _p2(atom_mask), _p1(flags), _p1(residue_index), self.target_flexibility)
            _tb = {k: _tf[k][0] for k in ('template_aatype', 'template_all_atom_positions', 'template_all_atom_mask')}
            _asym = _p1(asym_id, _pad_id)
            _mc = _asym[:, None] == _asym[None, :]
            _ac = jnp.asarray(_bcn.template_act_const(self.model_parameters[model], _tb, _mc, (_nmax, _nmax, 128), jnp.bfloat16), jnp.float32)
            _out = _predict(
                _p2(sequence), _p2(jnp.asarray(atoms, jnp.float32)),
                _p2(jnp.asarray(atom_mask, jnp.float32)), _p1(flags),
                _p1(residue_index), _asym, _p1(entity_id, _pad_id),
                _p1(interface_asym_id, _pad_id), _p1(seq_mask), _ac,
                jnp.asarray([_n], jnp.int32), jnp.stack([jax.random.bits(self.key, (), jnp.uint32), jnp.zeros((), jnp.uint32)]),
                softmax_weight=jnp.asarray(softmax_weight), one_hot_weight=jnp.asarray(one_hot_weight),
                temperature=jnp.asarray(temperature), logit_scale=jnp.asarray(logit_scale),
                recycles=self.num_recycle, msa_rows=1,
                weight_dirs={'trunk': _bcn.weights_from_params(self.model_parameters[model], depth=48),
                             'extra': _bcn.weights_from_params(self.model_parameters[model], depth=4, prefix=_bcn._EXTRA),
                             'tpl': _bcn.tplembed_weights_from_params(self.model_parameters[model]),
                             'post': _bcn.post_weights_from_params(self.model_parameters[model]),
                             'emb': _bcn.embed_weights_from_params(self.model_parameters[model])},
                # THE ATTRIBUTE IS "CAN IT DROP", not "does it drop now",
                # and it is part of the DsKey.  Passing False here because a
                # prediction is a measurement would build a SECOND
                # DesignStep beside the design one -- a second set of N^2
                # buffers, which is the whole allocation -- and the claim
                # that a campaign's predict is free would be false.  It
                # takes self.dropout, exactly as the gradient path does, and
                # seed[1] = 0 above is what actually keeps the masks away.
                dropout=bool(self.dropout))
            positions, mask = _out.atoms, _out.atom_mask
            metrics = {'plddt': _out.plddt, 'pae': _out.pae, 'ptm': _out.ptm, 'iptm': _out.iptm, 'iptm_per_residue': _out.iptm_per_residue, 'experimentally_resolved_ca': _out.experimentally_resolved_ca, 'distogram': _out.distogram}
        else:
            compiled_prediction = self._compiled_complex_prediction(model, padded_residue_count)
            prediction_arguments = (self.model_parameters[model], self.key, sequence, atoms, atom_mask, residue_index, asym_id, entity_id, interface_asym_id, seq_mask, flags, jnp.asarray(self.dropout), jnp.asarray(softmax_weight), jnp.asarray(one_hot_weight), jnp.asarray(temperature), jnp.asarray(logit_scale))
            prediction_shape = (self.model_families[model], padded_residue_count, resolve_subbatch_size(padded_residue_count, self.subbatch_size), self.multi_chain_binders, self.num_recycle, self.target_flexibility, self.large_complex_initialization)
            if prediction_shape not in self.compiled_shapes:
                with one_worker_compiles(prediction_shape, compile_blocking) as compiling:
                    if compiling:
                        compiled_prediction.lower(*prediction_arguments).compile()
                        self.compiled_shapes.add(prediction_shape)
            if compile_only:
                return StructurePrediction(protein_complex=protein_complex, metrics={})
            positions, mask, metrics = compiled_prediction(*prediction_arguments)
        positions, mask = positions[:residue_count], mask[:residue_count]
        metrics = {name: trim_prediction_padding(value, residue_count) for name, value in metrics.items()}
        # THE NATIVE STEP HAS ALREADY ALIGNED (kabsch.cuh), which is why the
        # gradient path does not call this either.  Running it twice would
        # not be wrong -- aligning aligned coordinates to the same target is
        # the identity -- but it would be a second Kabsch on every filter.
        if not native_predict:
            positions = align_prediction_to_target_template(positions, mask, atoms[:residue_count], atom_mask[:residue_count], flags[:residue_count])
        predicted_chain_arrays = split_residue_arrays_by_chain(chain_names, chain_lengths, atoms=positions, atom_mask=mask)
        predicted_complex = {name: protein_complex[name].replace(**{field: value[:true_length] for field, value in predicted_chain_arrays[name].items()}) for name, true_length in zip(chain_names, true_lengths)}
        if chain_lengths != true_lengths:
            metrics = trim_padded_metrics(metrics, real_residue_positions(chain_lengths, true_lengths))
        return StructurePrediction(protein_complex=predicted_complex, metrics=metrics)

    def _fall_back_to_chunked(self, model: str, complex_shapes: tuple, reference_shapes: tuple, losses: dict[str, DesignLoss], gradient_arguments: tuple, shape_signature: tuple, residue_total: int, subbatch_size: int | None | str, reason: str) -> Callable:
        print(f'{self.attention_backend} attention {reason} at {residue_total} padded residues; using the {ATTENTION_FALLBACK_BACKEND} route for this shape instead', flush=True)
        self.shape_attention_backend[shape_signature] = ATTENTION_FALLBACK_BACKEND
        compiled_gradient = self._compiled_sequence_gradients(model, complex_shapes, reference_shapes, losses, ATTENTION_FALLBACK_BACKEND, subbatch_size)
        compiled_gradient.lower(*gradient_arguments).compile()
        return compiled_gradient

    def _widest_shard_that_fits(self, model: str, complex_shapes: tuple, reference_shapes: tuple, losses: dict[str, DesignLoss], gradient_arguments: tuple, shape_signature: tuple, attention_backend: str, residue_total: int) -> tuple[Callable, str, int | None | str]:
        ladder = self.subbatch_ladder()
        for subbatch_size in ladder:
            compiled_gradient = self._compiled_sequence_gradients(model, complex_shapes, reference_shapes, losses, attention_backend, subbatch_size)
            if executable_fits(compiled_gradient.lower(*gradient_arguments).compile()):
                return compiled_gradient, attention_backend, subbatch_size
            print(f'{residue_total} padded residues need more than this worker may allocate at subbatch size {subbatch_size}; trying a narrower shard', flush=True)
        if attention_backend == ATTENTION_FALLBACK_BACKEND:
            return compiled_gradient, attention_backend, ladder[-1]
        return self._fall_back_to_chunked(model, complex_shapes, reference_shapes, losses, gradient_arguments, shape_signature, residue_total, ladder[-1], 'needs more than this worker may allocate at every subbatch size'), ATTENTION_FALLBACK_BACKEND, ladder[-1]


    def _compiled_sequence_gradients(self, model: str, complex_shapes: tuple[tuple[str, tuple[str, ...], tuple[int, ...]], ...], reference_shapes: tuple[tuple[str, tuple[str, ...], tuple[int, ...]], ...], losses: dict[str, DesignLoss], attention_backend: str | None=None, subbatch_size: int | None | str='auto') -> Callable:
        model_family = self.model_families[model]
        loss_signature = tuple((name, losses[name].function, losses[name].required_states) for name in sorted(losses))
        state_subbatch_sizes = tuple((state_name, resolve_subbatch_size(sum(chain_lengths), subbatch_size)) for state_name, _, chain_lengths in complex_shapes)
        attention_backend = attention_backend or self.attention_backend
        cache_key = model_family, complex_shapes, reference_shapes, loss_signature, state_subbatch_sizes, self.multi_chain_binders, self.num_recycle, self.target_flexibility, self.large_complex_initialization, attention_backend
        compiled_gradient = self.gradient_compile_cache.get(cache_key)
        if compiled_gradient is None:
            state_alphafold_runners = {state_name: self._alphafold_runner(model_family, subbatch_size, attention_backend) for state_name, subbatch_size in state_subbatch_sizes}
            # THE WHOLE STEP AS ONE CALL, on by default since 2026-10-13,
            # which is the convention every other native piece uses:
            # BC2_NATIVE_DISABLE=step leaves the staged path in place.
            #
            # What it took to get here, since "it is faster" was never the
            # question: the acceptance ladder (native/outcome_ladder.py) at
            # 12 seeds x 50 iters put the fused arm's outcome distribution
            # on top of the staged one and both ahead of jax, which is the
            # only answerable form of the question -- a bf16 Evoformer has
            # no reproducible trajectory, so there is no parity to check.
            # Dropout, which this path silently never applied, is wired and
            # gated (native/dropout_gate.py).  And the +15% it used to cost
            # at complex 215 is gone: its stacks were never handed their
            # keep tiers, so every block replayed.  6 seeds x 50 iters now
            # read 1.7520 min (jax) / 1.0129 (staged) / 1.0386 (fused).
            #
            # It follows the kernels, not the flag: on a card where they
            # cannot run, `backend=auto` falls back to JAX and so does this.
            #
            # The dirs are derived HERE, outside the traced function:
            # weights_from_params writes a manifest to disk and must not
            # run under a trace.
            native_step_dirs = None
            if self.native_installed and not _native_step_off():
                from .af import native as _bcn
                _p = self.model_parameters[model]
                native_step_dirs = {
                    'trunk': _bcn.weights_from_params(_p, depth=48),
                    'extra': _bcn.weights_from_params(_p, depth=4,
                                                      prefix=_bcn._EXTRA),
                    'tpl': _bcn.tplembed_weights_from_params(_p),
                    'post': _bcn.post_weights_from_params(_p),
                    'emb': _bcn.embed_weights_from_params(_p)}
                print('native: THE DESIGN STEP IS ONE CALL '
                      f'(model {model})', flush=True)
            def predict_complex_arrays(alphafold_runner: af_model.RunModel, model_parameters: Array, key: Array, chain_names: tuple[str, ...], chain_lengths: tuple[int, ...], sequence: Array, atoms: Array, atom_mask: Array, flags: Array, residue_index: Array, dropout: Array, softmax_weight: Array, one_hot_weight: Array, temperature: Array, logit_scale: Array, template_act_const: Array | None=None):
                asym_id = residue_chain_ids(chain_lengths)
                entity_id = residue_entity_ids(chain_names, chain_lengths, self.multi_chain_binders)
                seq_mask = real_residue_weights(flags)
                if native_step_dirs is not None:
                    # ONE CALL replaces features -> runner -> metrics ->
                    # align.  It aligns internally (kabsch.cuh), which is
                    # why this is the right seam and the predict path is
                    # not: that one aligns OUTSIDE its compiled function.
                    from .af.native_step import (design_step as _one_call,
                                                 pad_to_tile as _pad_to)
                    _n = sum(chain_lengths)
                    # PAD TO A TILE THE KERNELS CAN TAKE.  `length` being a
                    # runtime value collapses the per-bucket graphs; it does
                    # not excuse the BUFFERS from 32-alignment, and an
                    # unaligned NMAX dies in cuBLAS, not in a shape check.
                    # The padded tail is a chain of its own -- id
                    # len(chain_names), the same convention _predict_complex
                    # uses -- with seq_mask 0 so nothing real attends to it.
                    # ONE DesignStep FOR EVERY STATE.  `length` is a
                    # runtime input, so an object sized NMAX serves any
                    # length <= NMAX -- which is the whole reason the call
                    # takes a length instead of a shape.  Padding each
                    # state to its OWN tile builds one per state and holds
                    # both: at complex 647 that is 4.26 GB for a
                    # binder_alone that is idle while the complex runs, and
                    # it is what puts the step 2 GB over the card.  One
                    # NMAX across the states shares the allocation AND the
                    # graph.
                    # SHARE ONE DesignStep ONLY WHERE IT BUYS SOMETHING.
                    # `length` is a runtime input, so one object sized NMAX
                    # serves every length <= NMAX -- but `n` is a DEVICE
                    # value, so host-side launch dimensions cannot depend
                    # on it: the kernels mask past `n` and still launch
                    # NMAX threads.  A 260-residue binder_alone in a 672
                    # buffer therefore pays 672-sized compute, and sharing
                    # costs ~2x the step.
                    #
                    # Per-state objects are faster and hold two
                    # allocations, so share only above the measured point
                    # where per-state stops fitting.
                    #
                    # THAT POINT MOVED when the fused step started being
                    # handed its keep tiers (it never was -- see
                    # design_step.cuh's regrant).  A tier is memory, taken
                    # from exactly the headroom the second allocation needs,
                    # and 640 was measured when there were none.  It left a
                    # HOLE: complex 627 took per-state at 640+256 and died
                    # in trimul, while 647 was above the threshold, shared
                    # one 672 object, and ran.  A longer complex running
                    # where a shorter one crashes is the signature.
                    #
                    # Re-measured 2026-10-13, hIL2R_beta_gamma + a binder,
                    # the realistic four-term loss, A10:
                    #
                    #   complex  per-state      shared
                    #      597   14.1 s         --
                    #      607   14.2 s         --
                    #      617   OOM (xla)      --
                    #      627   OOM (xla)      29.3 s
                    #      647   --             33.8 s
                    #      667   --             34.0 s
                    #      687   --             37.7 s
                    #
                    # The two OOM sites that used to name a kernel file
                    # (trimul_body.cuh:1407 at 617/627, stack.cuh:540 at
                    # 687) were lazy cudaMallocs inside the step; they are
                    # allocated at alloc() time now.  That bought 687, which
                    # is why the shared column has a number there.  It did
                    # NOT buy 617-640 per-state: those bytes came back out
                    # of XLA's pool and the same length still does not fit.
                    # Per-state is a SIZE wall above padded 608, so the
                    # threshold below is unchanged.
                    # THE COMPARISON IS AGAINST THE PADDED LENGTH, which
                    # is what makes the right answer 608 and not 607.
                    # `_biggest` is _pad_to(...), so complex 607 arrives
                    # here as 608 and a threshold of 607 sends it to the
                    # shared object -- 25.8 s where per-state is 14.2.
                    # Picking the largest length that FITS rather than the
                    # bucket it lands in silently cost 1.8x across the whole
                    # 577-608 band.  Per-state fits at padded 608 and dies
                    # at padded 640, so the threshold is 608.
                    # BC2_NATIVE_STEP_SHARE_N moves it.
                    _share_n = int(os.environ.get(
                        'BC2_NATIVE_STEP_SHARE_N', '608'))
                    _biggest = _pad_to(max(sum(cl) for _, _, cl
                                           in complex_shapes))
                    # NB_ONE_OBJECT: pad every state to its OWN bucket and
                    # let the native side run one allocation at whatever
                    # length each state is.  That is per-state padding (the
                    # fast arm) with the shared arm's single allocation, so
                    # the threshold above does not apply -- but the object is
                    # sized by the first alloc, so the biggest has to be
                    # declared before anything traces.
                    if os.environ.get('NB_ONE_OBJECT') == 'big':
                        # DEBUG: one object, every state padded up to it, so
                        # setN is always a no-op.  Isolates the ds_get change
                        # from the re-derive.
                        from .af.native_step import set_alloc_floor
                        set_alloc_floor(_biggest)
                        _nmax = _biggest
                    elif _native_one_object():
                        from .af.native_step import set_alloc_floor
                        set_alloc_floor(_biggest)
                        _nmax = _pad_to(_n)
                    else:
                        _nmax = _biggest if _biggest > _share_n else _pad_to(_n)
                    _dn = _nmax - _n
                    _pad_id = len(chain_names)

                    def _p1(a, v=0):
                        return jnp.pad(a, [0, _dn], constant_values=v) if _dn else a

                    def _p2(a, v=0):
                        return (jnp.pad(a, [[0, _dn]] + [[0, 0]] * (a.ndim - 1),
                                        constant_values=v) if _dn else a)

                    _seq = _p2(sequence)
                    _atoms = _p2(jnp.asarray(atoms, jnp.float32))
                    _amask = _p2(jnp.asarray(atom_mask, jnp.float32))
                    _ac = jnp.asarray(template_act_const[0], jnp.float32)
                    if _dn:
                        _ac = jnp.pad(_ac, [[0, _dn], [0, _dn], [0, 0]])
                    if os.environ.get('BC2_NATIVE_STEP_DEBUG'):
                        for _nm, _v in (('sequence', sequence), ('atoms', atoms),
                                        ('atom_mask', atom_mask), ('flags', flags),
                                        ('residue_index', residue_index),
                                        ('asym_id', asym_id), ('entity_id', entity_id),
                                        ('seq_mask', seq_mask),
                                        ('act_const', template_act_const)):
                            print(f'  one-call operand {_nm:<16} '
                                  f'{getattr(_v, "dtype", None)} '
                                  f'{getattr(_v, "shape", None)}', flush=True)
                    # THE OPERAND CONTRACT IS EXACT (ffi.cu's Bind list):
                    # F32 sequence/atoms/atom_mask, U8 flags, S32 index and
                    # the three id arrays, F32 seq_mask/act_const, S32
                    # length, U32 seed.  bc2 carries atoms as float16 and
                    # atom_mask as bool, and a mismatch is not a shape error
                    # -- it is "Failed to decode all FFI handler operands
                    # (bad operands at: N)", with N COUNTED FROM 1.
                    #
                    # act_const loses its leading TEMPLATE axis here: bc2
                    # adds it with [None] because template_batch is scanned
                    # over it, and the one-call step takes the constant
                    # itself.
                    _out = _one_call(
                        _seq, _atoms, _amask,
                        _p1(flags), _p1(residue_index),
                        _p1(asym_id, _pad_id), _p1(entity_id, _pad_id),
                        _p1(interface_asym_ids(chain_names, chain_lengths),
                            _pad_id),
                        _p1(seq_mask), _ac,
                        jnp.asarray([_n], jnp.int32),
                        # [seed, enabled], exactly as the staged trunk hook
                        # builds it.  The seed comes from the model's own
                        # key so a run stays reproducible from it; `enabled`
                        # is the TRACED dropout flag, because bc2 flips it
                        # between the stages of one trajectory and a static
                        # one would build a second step -- a second set of
                        # N^2 buffers -- beside this one.  It was [0, 0]
                        # here, which ran every campaign without dropout and
                        # said nothing about it.
                        jnp.stack([
                            jax.random.bits(key, (), jnp.uint32),
                            jnp.asarray(dropout, jnp.uint32).reshape(())]),
                        softmax_weight=softmax_weight,
                        one_hot_weight=one_hot_weight,
                        temperature=temperature, logit_scale=logit_scale,
                        recycles=self.num_recycle, msa_rows=1,
                        weight_dirs=native_step_dirs,
                        # CAN it drop: whether the stacks reserve their mask
                        # scratches, decided once when the step is built.
                        dropout=bool(self.dropout))
                    predicted_atom_positions = _out.atoms[:_n]
                    predicted_atom_mask = _out.atom_mask[:_n]
                    metrics = {'plddt': _out.plddt[:_n],
                               'pae': _out.pae[:_n, :_n],
                               'ptm': _out.ptm, 'iptm': _out.iptm,
                               'iptm_per_residue': _out.iptm_per_residue[:_n],
                               'experimentally_resolved_ca':
                                   _out.experimentally_resolved_ca[:_n],
                               'distogram': _out.distogram[:_n, :_n]}
                else:
                    sequence_features, sequence_profile = prepare_design_sequence_features(sequence, flags, softmax_weight, one_hot_weight, temperature, logit_scale, self.amino_acid_bias)
                    model_inputs = alphafold_input_features(sequence_features, sequence_profile, atoms, atom_mask, residue_index, asym_id, seq_mask, flags, dropout, self.cyclic_offset_mode, entity_id, self.target_flexibility, self.large_complex_initialization, template_act_const)
                    alphafold_outputs = recycled_alphafold_outputs(alphafold_runner, model_parameters, key, model_inputs, self.num_recycle)
                    predicted_atom_positions = alphafold_outputs['structure_module']['final_atom_positions'].astype(jnp.float16)
                    predicted_atom_mask = alphafold_outputs['structure_module']['final_atom_mask'].astype(bool)
                    metrics = alphafold_prediction_metrics(alphafold_outputs, seq_mask, interface_asym_ids(chain_names, chain_lengths))
                    predicted_atom_positions = align_prediction_to_target_template(predicted_atom_positions, predicted_atom_mask, atoms, atom_mask, flags)
                input_chain_arrays = split_residue_arrays_by_chain(chain_names, chain_lengths, sequence=sequence, atoms=atoms, atom_mask=atom_mask, flags=flags, residue_index=residue_index)
                predicted_chain_arrays = split_residue_arrays_by_chain(chain_names, chain_lengths, atoms=predicted_atom_positions, atom_mask=predicted_atom_mask)
                input_complex = {name: Protein(**input_chain_arrays[name]) for name in chain_names}
                protein_complex = {name: input_complex[name].replace(**predicted_chain_arrays[name]) for name in chain_names}
                return input_complex, protein_complex, predicted_atom_positions, predicted_atom_mask, metrics
            def sequence_design_loss(model_parameters: Array, key: Array, sequences: dict[str, Array], state_templates: dict[str, dict[str, Array]], reference_templates: dict[str, dict[str, Array]], weights: dict[str, Array], frozen_interfaces: dict[str, dict[str, Array]], dropout: Array, softmax_weight: Array, one_hot_weight: Array, temperature: Array, logit_scale: Array):
                protein_states: ProteinStates = {}
                predictions: StructurePredictions = {}
                prediction_arrays: dict[str, tuple[Array, Array, dict]] = {}
                for state_name, state_chain_names, chain_lengths in complex_shapes:
                    template = state_templates[state_name]
                    original_complex, predicted_complex, predicted_atom_positions, predicted_atom_mask, metrics = predict_complex_arrays(state_alphafold_runners[state_name], model_parameters, key, state_chain_names, chain_lengths, jnp.concatenate([sequences[chain_name] for chain_name in state_chain_names], axis=0), template['atoms'], template['atom_mask'], template['flags'], template['residue_index'], dropout, softmax_weight, one_hot_weight, temperature, logit_scale, template.get('act_const'))
                    protein_states[state_name] = original_complex
                    predictions[state_name] = StructurePrediction(protein_complex=predicted_complex, metrics=dict(metrics))
                    prediction_arrays[state_name] = predicted_atom_positions, predicted_atom_mask, metrics
                loss_predictions = dict(predictions)
                for state_name, state_chain_names, chain_lengths in reference_shapes:
                    template = reference_templates[state_name]
                    chain_arrays = split_residue_arrays_by_chain(state_chain_names, chain_lengths, sequence=jnp.concatenate([sequences[name] for name in state_chain_names], axis=0), atoms=template['atoms'], atom_mask=template['atom_mask'], flags=template['flags'], residue_index=template['residue_index'])
                    loss_predictions[state_name] = StructurePrediction(protein_complex={name: Protein(**chain_arrays[name]) for name in state_chain_names}, metrics={'plddt': template['plddt']})
                weighted_losses = {name: weights[name] * entry.function(protein_states, loss_predictions, **frozen_interfaces.get(name, {})) for name, entry in losses.items()}
                total_loss = sum(weighted_losses.values()) if weighted_losses else jnp.asarray(0.0)
                for name, entry in losses.items():
                    for state_name in (entry.required_states or prediction_arrays.keys()) & prediction_arrays.keys():
                        metrics = prediction_arrays[state_name][2]
                        if name in metrics:
                            raise ValueError(f'loss {name!r} would overwrite an existing metric on {state_name!r}')
                        metrics[name] = weighted_losses[name]
                return total_loss, prediction_arrays
            compiled_gradient = jax.jit(jax.value_and_grad(sequence_design_loss, argnums=2, has_aux=True))
            self.gradient_compile_cache.set(cache_key, compiled_gradient)
        return compiled_gradient

    def sequence_gradients(self, protein_states: ProteinStates, losses: dict[str, DesignLoss], model: str | None=None, softmax_weight: float=1.0, one_hot_weight: float=0.0, temperature: float=1.0, logit_scale: float=2.0, reference_predictions: StructurePredictions | None=None, compile_only: bool=False, compile_blocking: bool=True) -> tuple[StructurePredictions, dict[str, Array]]:
        model = self._resolve_model_name(model)
        _, original_shared_chains = collect_shared_chains(protein_states)
        state_names = canonical_state_names(protein_states)
        chain_names = canonical_chain_names(protein_states, state_names)
        padded_states = pad_design_chains(renamed_protein_states(protein_states, state_names, chain_names), self.length_bucket_size, self.target_pad_length)
        metric_source_names = {renamed_loss_name(name, state_names): name for name in losses}
        losses = renamed_state_losses(losses, state_names)
        state_source_names = {canonical: name for name, canonical in state_names.items()}
        chain_source_names = {canonical: name for name, canonical in chain_names.items()}
        complex_shapes = protein_state_shapes(padded_states)
        shared_chain_names, shared_chains = collect_shared_chains(padded_states)
        sequences = {name: shared_chains[name].sequence for name in shared_chain_names}
        state_templates = {state_name: concatenate_chain_arrays(state_chain_names, padded_states[state_name], 'atoms', 'atom_mask', 'flags', 'residue_index') for state_name, state_chain_names, _ in complex_shapes}
        reference_states = {state_names.get(state, state): {chain_names.get(name, name): protein.padded_to(len(shared_chains[chain_names.get(name, name)])) for name, protein in prediction.protein_complex.items() if chain_names.get(name, name) in shared_chains} for state, prediction in (reference_predictions or {}).items()}
        reference_states = {state: protein_complex for state, protein_complex in reference_states.items() if protein_complex}
        reference_shapes = protein_state_shapes(reference_states)
        reference_confidence = {state_names.get(state, state): prediction for state, prediction in (reference_predictions or {}).items()}
        reference_templates = {state_name: {**concatenate_chain_arrays(state_chain_names, reference_states[state_name], 'atoms', 'atom_mask', 'flags', 'residue_index'), 'plddt': reference_state_confidence(reference_confidence[state_name], chain_names, shared_chains, state_chain_names)} for state_name, state_chain_names, _ in reference_shapes}
        weights = {name: jnp.asarray(entry.weight, dtype=jnp.float32) for name, entry in losses.items()}
        frozen_interfaces = frozen_interface_arguments(losses, shared_chains)
        if self.native_evoformer:
            from .af import native
            native.use_model(model)   # bind THIS model's weights before tracing
            # THE TEMPLATE CONSTANT, harvested once per target rather than
            # rebuilt every step.  Eight of SingleTemplateEmbedding's nine
            # feature projections depend only on the template, so this lifts
            # the backbone affine, the rigid inverse, the pseudo-beta, the
            # distogram, the one-hots and eight Linears out of the graph --
            # off the compile AND off every step.
            #
            # aatype must come from the SAME place the graph gets it:
            # prepare_design_sequence_features, then argmax.  Taking the raw
            # sequence's argmax instead looks equivalent -- template residues
            # are not designed, so their sequence is fixed -- and is not: the
            # softmax, bias and temperature move the argmax at some template
            # positions, and the metrics moved with it (pLDDT 0.451212 ->
            # 0.450127).  The end-to-end check is what caught it.
            # ON by default; BC2_TEMPLATE_CONST=0 turns it off.
            #
            # The substitution is proven exact: harvest the constant through
            # the module's own __const_only__ path, hand it back, and the
            # whole of SingleTemplateEmbedding reproduces bit for bit,
            # max|d| 0.000e+00.  In model it still shifts pLDDT 1.1e-03
            # (0.451212 -> 0.450127), which I could not account for -- aatype
            # was the obvious suspect and is not it, and with a single chain
            # the multichain mask cannot be either.
            #
            # It was held off for that, which was an inconsistent bar: the
            # template query barrier is a PROVABLE no-op and moves this model
            # by the same 1.1e-03 through fusion alone, and it ships on.  Both
            # shifts are orders below what a trajectory resolves -- two
            # identical campaigns on this target disagreed by more and flipped
            # the final accept -- so either both are acceptable or neither is.
            if os.environ.get('BC2_TEMPLATE_CONST') != '0':
                for state_name, state_chain_names, _ in complex_shapes:
                    tmpl = state_templates[state_name]
                    seq = jnp.concatenate(
                        [sequences[c] for c in state_chain_names], axis=0)
                    seq_feat, _ = prepare_design_sequence_features(
                        seq, tmpl['flags'], jnp.asarray(softmax_weight),
                        jnp.asarray(one_hot_weight), jnp.asarray(temperature),
                        jnp.asarray(logit_scale), self.amino_acid_bias)
                    tf = target_template_features(
                        seq_feat.argmax(-1).astype(jnp.int32), tmpl['atoms'],
                        tmpl['atom_mask'], tmpl['flags'],
                        tmpl['residue_index'], self.target_flexibility)
                    # target_template_features emits a leading TEMPLATE
                    # axis (hk.scan maps over it); SingleTemplateEmbedding
                    # sees one slice, so drop it here
                    tb = {k: tf[k][0] for k in ('template_aatype',
                                                'template_all_atom_positions',
                                                'template_all_atom_mask')}
                    asym = residue_chain_ids(
                        tuple(len(sequences[c]) for c in state_chain_names))
                    mc = asym[:, None] == asym[None, :]
                    n = seq.shape[0]
                    pair_ch = 128
                    # back on the TEMPLATE axis: template_batch is scanned
                    # over it, so every entry needs one
                    tmpl['act_const'] = native.template_act_const(
                        self.model_parameters[model], tb, mc,
                        (n, n, pair_ch), jnp.bfloat16)[None]
        shape_signature = self.model_families[model], complex_shapes, reference_shapes
        residue_total = sum(sum(chain_lengths) for _, _, chain_lengths in complex_shapes)
        attention_backend = self.shape_attention_backend.get(shape_signature, self.attention_backend)
        subbatch_size = self.shape_subbatch_size.get(shape_signature, self.subbatch_ladder()[0])
        compiled_sequence_gradients = self._compiled_sequence_gradients(model, complex_shapes, reference_shapes, losses, attention_backend, subbatch_size)
        gradient_arguments = (self.model_parameters[model], self.key, sequences, state_templates, reference_templates, weights, frozen_interfaces, jnp.asarray(self.dropout), jnp.asarray(softmax_weight), jnp.asarray(one_hot_weight), jnp.asarray(temperature), jnp.asarray(logit_scale))
        if shape_signature not in self.shape_attention_backend:
            with one_worker_compiles((model, complex_shapes, reference_shapes, tuple(sorted(losses))), compile_blocking) as compiling:
                if compiling:
                    compiled_sequence_gradients, attention_backend, subbatch_size = self._widest_shard_that_fits(model, complex_shapes, reference_shapes, losses, gradient_arguments, shape_signature, attention_backend, residue_total)
                    self.shape_attention_backend[shape_signature] = attention_backend
                    self.shape_subbatch_size[shape_signature] = subbatch_size
        self.last_attention_backend = attention_backend
        if compile_only:
            return {}, {}, jnp.asarray(0.0)
        try:
            (design_loss, prediction_arrays), shared_chain_gradients = compiled_sequence_gradients(*gradient_arguments)
            if residue_total not in self.executed_lengths:
                jax.block_until_ready(((design_loss, prediction_arrays), shared_chain_gradients))
                self.executed_lengths.add(residue_total)
        except Exception as gradient_failure:
            if attention_backend == ATTENTION_FALLBACK_BACKEND or not is_out_of_memory(gradient_failure):
                raise
            compiled_sequence_gradients = self._fall_back_to_chunked(model, complex_shapes, reference_shapes, losses, gradient_arguments, shape_signature, residue_total, subbatch_size, 'ran out of memory despite fitting the check')
            self.last_attention_backend = ATTENTION_FALLBACK_BACKEND
            (design_loss, prediction_arrays), shared_chain_gradients = compiled_sequence_gradients(*gradient_arguments)
            jax.block_until_ready(((design_loss, prediction_arrays), shared_chain_gradients))
            self.executed_lengths.add(residue_total)
        predictions: StructurePredictions = {}
        for canonical_state, canonical_chains, chain_lengths in complex_shapes:
            state_name = state_source_names[canonical_state]
            source_chains = tuple(chain_source_names[name] for name in canonical_chains)
            positions, mask, metrics = prediction_arrays[canonical_state]
            predicted_chain_arrays = split_residue_arrays_by_chain(canonical_chains, chain_lengths, atoms=positions, atom_mask=mask)
            true_lengths = tuple(len(protein_states[state_name][name]) for name in source_chains)
            predicted_complex = {name: protein_states[state_name][name].replace(**{field: value[:true_length] for field, value in predicted_chain_arrays[canonical_chain].items()}) for name, canonical_chain, true_length in zip(source_chains, canonical_chains, true_lengths)}
            metrics = {metric_source_names.get(name, name): value for name, value in metrics.items()}
            predictions[state_name] = StructurePrediction(protein_complex=predicted_complex, metrics=trim_padded_metrics(metrics, real_residue_positions(chain_lengths, true_lengths)) if chain_lengths != true_lengths else metrics)
        sequence_gradients = {chain_source_names[name]: shared_chain_gradients[name][:len(original_shared_chains[chain_source_names[name]])] for name in shared_chain_names}
        return predictions, sequence_gradients, design_loss
MULTIMER_POOL: tuple[str, ...] = tuple(f'model_{i}_multimer_v3' for i in range(1, 6))
MONOMER_POOL: tuple[str, ...] = ('model_1_ptm', 'model_2_ptm')
