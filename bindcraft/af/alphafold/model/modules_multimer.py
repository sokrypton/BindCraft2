# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core modules, which have been refactored in AlphaFold-Multimer.

The main difference is that MSA sampling pipeline is moved inside the JAX model
for easier implementation of recycling and ensembling.

Lower-level modules up to EvoformerIteration are reused from modules.py.
"""

import functools
from typing import Sequence

from bindcraft.af.alphafold.common import residue_constants
from bindcraft.af.alphafold.model import all_atom_multimer
from bindcraft.af.alphafold.model import common_modules
from bindcraft.af.alphafold.model import folding_multimer
from bindcraft.af.alphafold.model import geometry
from bindcraft.af.alphafold.model import layer_stack
from bindcraft.af.alphafold.model import modules
from bindcraft.af.alphafold.model import prng
from bindcraft.af.alphafold.model import utils

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

def create_extra_msa_feature(batch, num_extra_msa):
  """Expand extra_msa into 1hot and concat with other extra msa features.
  We do this as late as possible as the one_hot extra msa can be very large.
  Args:
    batch: a dictionary with the following keys:
     * 'extra_msa': [num_seq, num_res] MSA that wasn't selected as a cluster
       centre. Note - This isn't one-hotted.
     * 'extra_deletion_matrix': [num_seq, num_res] Number of deletions at given
        position.
    num_extra_msa: Number of extra msa to use.
  Returns:
    Concatenated tensor of extra MSA features.
  """
  # 23 = 20 amino acids + 'X' for unknown + gap + bert mask
  extra_msa = batch['extra_msa'][:num_extra_msa]
  deletion_matrix = batch['extra_deletion_value'][:num_extra_msa]
  msa_1hot = jax.nn.one_hot(extra_msa, 23)
  has_deletion = jnp.clip(deletion_matrix, 0., 1.)[..., None]
  deletion_value = (jnp.arctan(deletion_matrix / 3.) * (2. / jnp.pi))[..., None]
  extra_msa_mask = batch['extra_msa_mask'][:num_extra_msa]
  return jnp.concatenate([msa_1hot, has_deletion, deletion_value],
                         axis=-1), extra_msa_mask

# NATIVE KERNELS, opt-in.  bindcraft/af/native.py sets these to callables and
# the call site below uses them instead of the JAX stack.  Left as None the
# file behaves exactly as upstream: this is an INSTALL POINT, not a fork.
#
# The boundary is the Evoformer's inputs and outputs, because the trunk is
# ~99% of a design step's runtime (measured: a JAX trunk-only step at L=128
# costs 1372 ms against 1382 ms for the whole chain).  Heads and losses stay
# in JAX, where a new loss term is a Python edit rather than a CUDA rebuild.
NATIVE_EVOFORMER = None
NATIVE_EXTRA_MSA_STACK = None
NATIVE_TEMPLATE_STACK = None


# Set by bindcraft.af.native.install_models(); None leaves the JAX path alone.
NATIVE_POST_TRUNK = None
# INSTALLED IS NOT FIRED.  The extra-MSA stack once gated 'clean' three ways
# while its hook was never reached -- predict() does not call it, use_model()
# rebinds it, and a warm compile cache can skip the trace entirely.  This
# counts traces so a comparison run can state, not assume, that the native
# path is the one that ran.
NATIVE_POST_TRUNK_TRACES = [0]

# Set by bindcraft.af.native.install_models(); None leaves the JAX path alone.
NATIVE_EMBEDDER = None
NATIVE_EMBEDDER_TRACES = [0]

# See the note at the template call site: 0.7 s of compile, 1.9 ms a step.
TEMPLATE_QUERY_BARRIER = True

# The WHOLE template embedding in one custom call -- query_embedding_norm,
# template_pair_embedding_8, both stack blocks, output_layer_norm, the relu
# and output_linear.  Supersedes NATIVE_TEMPLATE_STACK, which takes only the
# two blocks in the middle; set one or the other, not both.
NATIVE_TEMPLATE_EMBED = None
NATIVE_TEMPLATE_EMBED_TRACES = [0]


def _native_post_trunk(representations, batch, config, safe_key):
  """Run the structure module and the four heads natively.

  Returns None -- and the caller falls back to JAX -- whenever the native
  path cannot produce exactly what the JAX one would: a head this model
  weights that the call does not cover, or a shape it was not built for.
  Falling back is always allowed; producing something subtly different is
  not.
  """
  import jax.numpy as jnp
  from . import all_atom_multimer

  single, pair = representations['single'], representations['pair']
  aatype, seq_mask = batch['aatype'], batch['seq_mask']
  heads_cfg = config.heads
  # every head this model actually weights has to be one the call produces
  wanted = {n for n, c in heads_cfg.items()
            if getattr(c, 'weight', 0.0) and n != 'structure_module'}
  if not wanted <= {'distogram', 'predicted_lddt', 'predicted_aligned_error',
                    'experimentally_resolved', 'masked_msa'}:
    return None

  pair_mask = seq_mask[:, None] * seq_mask[None, :]
  # [seed, enabled], both RUNTIME.  use_dropout is a traced value -- a stage
  # turns dropout off without recompiling -- so it rides in the buffer rather
  # than becoming an attribute, exactly as the three Evoformer hooks do.
  seed = jnp.stack([
      jax.random.bits(safe_key.get(), (), jnp.uint32),
      jnp.asarray(batch['use_dropout'], jnp.uint32).reshape(()),
  ])
  NATIVE_POST_TRUNK_TRACES[0] += 1
  if NATIVE_POST_TRUNK_TRACES[0] == 1:
    print(f'native: post-trunk hook FIRED (N={single.shape[0]})', flush=True)
  out = NATIVE_POST_TRUNK(single, pair, pair_mask, aatype, seed)

  # atom14 -> atom37 STAYS IN JAX.  It is one gather through a constant
  # per-restype table: the same cost in XLA as in CUDA, and porting the
  # tables would put a second copy of AF2's geometry behind a kernel.
  atom14 = out['atom14']
  atom37_mask = all_atom_multimer.get_atom37_mask(aatype) * seq_mask[:, None]
  atom37 = all_atom_multimer.atom14_to_atom37(atom14, aatype)
  atom37 = atom37 * atom37_mask[:, :, None]
  atom14_mask = all_atom_multimer.get_atom14_mask(aatype) * seq_mask[:, None]

  ret = {'structure_module': {
      'final_atom14_positions': atom14,
      'final_atom14_mask': atom14_mask,
      'final_atom_positions': atom37,
      'final_atom_mask': atom37_mask,
  }}
  if 'distogram' in wanted:
    c = heads_cfg.distogram
    ret['distogram'] = {
        'logits': out['distogram'],
        'bin_edges': jnp.linspace(c.first_break, c.last_break, c.num_bins - 1),
    }
  if 'predicted_aligned_error' in wanted:
    c = heads_cfg.predicted_aligned_error
    ret['predicted_aligned_error'] = {
        'logits': out['pae'],
        'breaks': jnp.linspace(0., c.max_error_bin, c.num_bins - 1),
    }
  if 'predicted_lddt' in wanted:
    ret['predicted_lddt'] = {'logits': out['plddt']}
  if 'experimentally_resolved' in wanted:
    ret['experimentally_resolved'] = {'logits': out['expres']}
  return {'ret': ret}


class AlphaFoldIteration(hk.Module):
  """A single recycling iteration of AlphaFold architecture.

  Computes ensembled (averaged) representations from the provided features.
  These representations are then passed to the various heads
  that have been requested by the configuration file.
  """

  def __init__(self, config, global_config, name='alphafold_iteration'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
               batch,
               return_representations=False,
               safe_key=None):


    # Compute representations for each MSA sample and average.
    embedding_module = EmbeddingsAndEvoformer(
        self.config.embeddings_and_evoformer, self.global_config)

    safe_key, safe_subkey = safe_key.split()
    representations = embedding_module(batch, safe_key=safe_subkey)

    self.representations = representations
    self.batch = batch
    self.heads = {}
    for head_name, head_config in sorted(self.config.heads.items()):
      if not head_config.weight:
        continue  # Do not instantiate zero-weight heads.

      head_factory = {
          'masked_msa':
              modules.MaskedMsaHead,
          'distogram':
              modules.DistogramHead,
          'structure_module':
              folding_multimer.StructureModule,
          'predicted_aligned_error':
              modules.PredictedAlignedErrorHead,
          'predicted_lddt':
              modules.PredictedLDDTHead,
          'experimentally_resolved':
              modules.ExperimentallyResolvedHead,
      }[head_name]
      self.heads[head_name] = (head_config,
                               head_factory(head_config, self.global_config))

    structure_module_output = None
    native_post = None
    if 'entity_id' in batch and 'all_atom_positions' in batch:
      # THE POST-TRUNK STEP IN ONE CALL.  Everything from here to the tensors
      # a loss reads -- the eight fold iterations, the atom builder and the
      # four heads -- is fixed model code nobody edits, and compiling it is
      # most of what a design step pays XLA before its first gradient.  The
      # native path replaces the lot; the loss above it stays ordinary
      # differentiable Python.
      if NATIVE_POST_TRUNK is not None:
        safe_key, post_subkey = safe_key.split()
        native_post = _native_post_trunk(representations, batch, self.config,
                                         post_subkey)
      if native_post is None:
        _, fold_module = self.heads['structure_module']
        structure_module_output = fold_module(representations, batch)


    ret = {}
    ret['representations'] = representations

    if native_post is not None:
      # the heads below are skipped: their outputs are already here.
      # representations['structure_module'] is deliberately NOT set.  The JAX
      # path puts the fold module's act there for one reader, PredictedLDDTHead,
      # and this path has already produced that head's logits.  Leaving the key
      # absent means a future reader raises instead of quietly getting a None
      # that would look like a zero activation.
      ret.update(native_post['ret'])
      for name, (head_config, module) in self.heads.items():
        if name in ret or name == 'structure_module':
          continue
        ret[name] = module(representations, batch)
      if self.config.heads.get('predicted_aligned_error.weight', 0.0):
        ret['predicted_aligned_error']['asym_id'] = batch['asym_id']
      return ret

    for name, (head_config, module) in self.heads.items():
      if name == 'structure_module' and structure_module_output is not None:
        ret[name] = structure_module_output
        representations['structure_module'] = structure_module_output.pop('act')
      # Skip confidence heads until StructureModule is executed.
      elif name in {'predicted_lddt', 'predicted_aligned_error',
                    'experimentally_resolved'}:
        continue
      else:
        ret[name] = module(representations, batch)


    # Add confidence heads after StructureModule is executed.
    if self.config.heads.get('predicted_lddt.weight', 0.0):
      name = 'predicted_lddt'
      head_config, module = self.heads[name]
      ret[name] = module(representations, batch)

    if self.config.heads.experimentally_resolved.weight:
      name = 'experimentally_resolved'
      head_config, module = self.heads[name]
      ret[name] = module(representations, batch)

    if self.config.heads.get('predicted_aligned_error.weight', 0.0):
      name = 'predicted_aligned_error'
      head_config, module = self.heads[name]
      ret[name] = module(representations, batch)
      # Will be used for ipTM computation.
      ret[name]['asym_id'] = batch['asym_id']

    return ret

class AlphaFold(hk.Module):
  """AlphaFold-Multimer model with recycling.
  """

  def __init__(self, config, name='alphafold'):
    super().__init__(name=name)
    self.config = config
    self.global_config = config.global_config

  def __call__(
      self,
      batch,
      return_representations=False,
      safe_key=None):

    c = self.config
    impl = AlphaFoldIteration(c, self.global_config)

    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())
    elif isinstance(safe_key, jnp.ndarray):
      safe_key = prng.SafeKey(safe_key)

    assert isinstance(batch, dict)
    num_res = batch['aatype'].shape[0]
    
    def get_prev(ret):
      new_prev = {
          'prev_pos': ret['structure_module']['final_atom_positions'],
          'prev_msa_first_row': ret['representations']['msa_first_row'],
          'prev_pair': ret['representations']['pair'],
      }
      return new_prev

    def apply_network(prev, safe_key):
      recycled_batch = {**batch, **prev}
      return impl(
          batch=recycled_batch,
          safe_key=safe_key)
        
    ret = apply_network(prev=batch.pop("prev"), safe_key=safe_key)
    ret["prev"] = get_prev(ret)
    
    if not return_representations:
      del ret['representations']
    return ret

class EmbeddingsAndEvoformer(hk.Module):
  """Embeds the input data and runs Evoformer.

  Produces the MSA, single and pair representations.
  """

  def __init__(self, config, global_config, name='evoformer'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def _relative_encoding(self, batch):
    """Add relative position encodings.

    For position (i, j), the value is (i-j) clipped to [-k, k] and one-hotted.

    When not using 'use_chain_relative' the residue indices are used as is, e.g.
    for heteromers relative positions will be computed using the positions in
    the corresponding chains.

    When using 'use_chain_relative' we add an extra bin that denotes
    'different chain'. Furthermore we also provide the relative chain index
    (i.e. sym_id) clipped and one-hotted to the network. And an extra feature
    which denotes whether they belong to the same chain type, i.e. it's 0 if
    they are in different heteromer chains and 1 otherwise.

    Args:
      batch: batch.
    Returns:
      Feature embedding using the features as described before.
    """
    c = self.config
    gc = self.global_config
    rel_feats = []
    asym_id = batch['asym_id']
    asym_id_same = jnp.equal(asym_id[:, None], asym_id[None, :])

    if "offset" in batch:
      offset = batch['offset']
    else:
      pos = batch['residue_index']
      offset = pos[:, None] - pos[None, :]

    clipped_offset = jnp.clip(
        offset + c.max_relative_idx, min=0, max=2 * c.max_relative_idx)

    dtype = jnp.bfloat16 if gc.bfloat16 else jnp.float32

    if c.use_chain_relative:

      final_offset = jnp.where(asym_id_same, clipped_offset,
                               (2 * c.max_relative_idx + 1) *
                               jnp.ones_like(clipped_offset))

      rel_pos = jax.nn.one_hot(final_offset, 2 * c.max_relative_idx + 2)

      rel_feats.append(rel_pos)

      entity_id = batch['entity_id']
      entity_id_same = jnp.equal(entity_id[:, None], entity_id[None, :])
      rel_feats.append(entity_id_same.astype(rel_pos.dtype)[..., None])

      sym_id = batch['sym_id']
      rel_sym_id = sym_id[:, None] - sym_id[None, :]

      max_rel_chain = c.max_relative_chain

      clipped_rel_chain = jnp.clip(
          rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain)

      final_rel_chain = jnp.where(entity_id_same, clipped_rel_chain,
                                  (2 * max_rel_chain + 1) *
                                  jnp.ones_like(clipped_rel_chain))
      rel_chain = jax.nn.one_hot(final_rel_chain, 2 * c.max_relative_chain + 2)

      rel_feats.append(rel_chain)

    else:
      rel_pos = jax.nn.one_hot(clipped_offset, 2 * c.max_relative_idx + 1)
      rel_feats.append(rel_pos)

    rel_feat = jnp.concatenate(rel_feats, axis=-1)

    rel_feat = rel_feat.astype(dtype)
    return common_modules.Linear(
        c.pair_channel,
        name='position_activations')(rel_feat)

  def __call__(self, batch, safe_key=None):

    c = self.config
    gc = self.global_config

    batch = dict(batch)
    dtype = jnp.bfloat16 if gc.bfloat16 else jnp.float32

    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())

    output = {}
    with utils.bfloat16_context():
      
      msa_feat = batch['msa_feat'].astype(dtype)
      target_feat = jnp.pad(batch["target_feat"].astype(dtype),[[0,0],[0,1]])
      # THE FOUR INPUT LINEARS AND THE RELATIVE-POSITION GATHER, natively.
      # This is where a design step's gradient reaches the sequence.
      #
      # It returns the relative encoding ALREADY ADDED into the pair, so the
      # `max_relative_idx` block below is skipped.  The recycle terms between
      # here and there are not folded in: they are added afterwards instead,
      # which is the same sum because every one of them is a `+=`.
      native_emb = NATIVE_EMBEDDER is not None
      if native_emb:
        NATIVE_EMBEDDER_TRACES[0] += 1
        if NATIVE_EMBEDDER_TRACES[0] == 1:
          print(f'native: embedder hook FIRED (N={target_feat.shape[0]})',
                flush=True)
        msa_activations, pair_activations = NATIVE_EMBEDDER(
            target_feat, msa_feat, batch['residue_index'], batch['asym_id'],
            batch['entity_id'], batch['sym_id'])
        # the call works in f32 where this block works in bfloat16; rounding
        # back here keeps everything downstream byte-identical in shape and
        # dtype, and leaves the Linears computed on the more accurate side
        msa_activations = msa_activations.astype(dtype)
        pair_activations = pair_activations.astype(dtype)
        num_msa_sequences = msa_activations.shape[0]
      else:
        preprocess_1d = common_modules.Linear(c.msa_channel, name='preprocess_1d')(target_feat)
        preprocess_msa = common_modules.Linear(c.msa_channel, name='preprocess_msa')(msa_feat)
        msa_activations = preprocess_1d[None] + preprocess_msa
        num_msa_sequences = msa_activations.shape[0]

        left_single = common_modules.Linear(c.pair_channel, name='left_single')(target_feat)
        right_single = common_modules.Linear(c.pair_channel, name='right_single')(target_feat)
        pair_activations = left_single[:, None] + right_single[None]
      mask_2d = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]
      mask_2d = mask_2d.astype(dtype)

      if c.recycle_pos:
        prev_pseudo_beta = modules.pseudo_beta_fn(batch['aatype'], batch['prev_pos'], None)
        dgram = modules.dgram_from_positions(prev_pseudo_beta, **self.config.prev_pos)
        dgram = dgram.astype(dtype)
        pair_activations += common_modules.Linear(c.pair_channel, name='prev_pos_linear')(dgram)

      if c.recycle_features:
        prev_msa_first_row = common_modules.LayerNorm(
            axis=[-1],
            create_scale=True,
            create_offset=True,
            name='prev_msa_first_row_norm')(batch['prev_msa_first_row']).astype(dtype)
        
        msa_activations = msa_activations.at[0].add(prev_msa_first_row)

        pair_activations += common_modules.LayerNorm(
            axis=[-1],
            create_scale=True,
            create_offset=True,
            name='prev_pair_norm')(batch['prev_pair']).astype(dtype)

      # already in the pair when the native embedder produced it
      if c.max_relative_idx and not native_emb:
        pair_activations += self._relative_encoding(batch)

      if c.template.enabled:
        template_module = TemplateEmbedding(c.template, gc)
        template_batch = {
            'template_aatype': batch['template_aatype'],
            'template_all_atom_positions': batch['template_all_atom_positions'],
            'template_all_atom_mask': batch['template_all_atom_mask']
        }
        if "template_dgram" in batch:
          template_batch["template_dgram"] = batch["template_dgram"]
        if "template_act_const" in batch:
          template_batch["template_act_const"] = batch["template_act_const"]

        # Construct a mask such that only intra-chain template features are
        # computed, since all templates are for each chain individually.
        multichain_mask = batch['asym_id'][:, None] == batch['asym_id'][None, :]
        multichain_mask = jnp.where(batch["mask_template_interchain"], multichain_mask, True)

        safe_key, safe_subkey = safe_key.split()
        # Splits the one long JAX region -- input Linears, relative encoding,
        # recycle terms, construct_input -- that XLA otherwise optimises as a
        # unit.  0.7 s of compile for 1.9 ms a step.  A semantic no-op that is
        # not output-preserving: it reorders fusion, hence f32 accumulation,
        # which this model amplifies to 1.1e-03 on pLDDT.
        query_for_templates = (
            jax.lax.optimization_barrier(pair_activations)
            if TEMPLATE_QUERY_BARRIER else pair_activations)
        template_act = template_module(
            query_embedding=query_for_templates,
            template_batch=template_batch,
            padding_mask_2d=mask_2d,
            multichain_mask_2d=multichain_mask,
            use_dropout=batch["use_dropout"],
            safe_key=safe_subkey)
        pair_activations += template_act

      # Extra MSA stack.
      (extra_msa_feat, extra_msa_mask) = create_extra_msa_feature(batch, c.num_extra_msa)
      extra_msa_activations = common_modules.Linear(c.extra_msa_channel,
                                                    name='extra_msa_activations')(extra_msa_feat).astype(dtype)
      extra_msa_mask = extra_msa_mask.astype(dtype)
      extra_evoformer_input = {'msa': extra_msa_activations, 'pair': pair_activations}
      extra_masks = {'msa': extra_msa_mask, 'pair': mask_2d}
      extra_evoformer_iteration = modules.EvoformerIteration(c.evoformer, gc, is_extra_msa=True, name='extra_msa_stack')

      def extra_evoformer_fn(x):
        act, safe_key = x
        safe_key, safe_subkey = safe_key.split()
        extra_evoformer_output = extra_evoformer_iteration(
            activations=act,
            masks=extra_masks,
            use_dropout=batch["use_dropout"],
            safe_key=safe_subkey)
        return (extra_evoformer_output, safe_key)

      if gc.use_remat:
        extra_evoformer_fn = hk.remat(extra_evoformer_fn)

      safe_key, safe_subkey = safe_key.split()
      extra_evoformer_stack = layer_stack.layer_stack(
          c.extra_msa_stack_num_block)(
              extra_evoformer_fn)

      def run_extra(x):
        out, _ = extra_evoformer_stack((x, safe_subkey))
        return out

      if NATIVE_EXTRA_MSA_STACK is not None:
        # Same deal as the trunk: eval_shape first so haiku still creates the
        # parameters, then the kernels.  Worth taking even though it is four
        # blocks and not forty-eight -- layer_stack scans them, so XLA compiles
        # ONE block body either way, forward and reverse, and the reverse is
        # 70% of this graph's compile.
        hk.eval_shape(run_extra, extra_evoformer_input)
        from ...native import supported_length
        if supported_length(extra_evoformer_input['pair'].shape[0]):
          extra_evoformer_output = NATIVE_EXTRA_MSA_STACK(
              extra_evoformer_input, extra_masks,
              int(c.extra_msa_stack_num_block),
              jnp.stack([
                  jax.random.bits(safe_subkey.get(), (), jnp.uint32),
                  jnp.asarray(batch["use_dropout"], jnp.uint32).reshape(()),
              ]),
              # the mask this path's validity rests on, so the runner can
              # refuse rather than silently substitute the wrong constant
              extra_msa_mask=batch.get('extra_msa_mask'))
        else:
          extra_evoformer_output = run_extra(extra_evoformer_input)
      else:
        extra_evoformer_output, safe_key = extra_evoformer_stack(
            (extra_evoformer_input, safe_subkey))

      pair_activations = extra_evoformer_output['pair']
      # Get the size of the MSA before potentially adding templates, so we
      # can crop out the templates later.
      num_msa_sequences = msa_activations.shape[0]
      evoformer_input = {
          'msa': msa_activations,
          'pair': pair_activations,
      }
      evoformer_masks = {'msa': batch['msa_mask'].astype(dtype), 'pair': mask_2d}      
      if c.template.enabled:
        template_features, template_masks = (
            template_embedding_1d(batch=batch, num_channel=c.msa_channel, global_config=gc))

        evoformer_input['msa'] = jnp.concatenate([evoformer_input['msa'], template_features], axis=0)
        evoformer_masks['msa'] = jnp.concatenate([evoformer_masks['msa'], template_masks], axis=0)
        
      evoformer_iteration = modules.EvoformerIteration(
          c.evoformer, gc, is_extra_msa=False, name='evoformer_iteration')
    
      def evoformer_fn(x):
        act, safe_key = x
        safe_key, safe_subkey = safe_key.split()
        evoformer_output = evoformer_iteration(
            activations=act,
            masks=evoformer_masks,
            use_dropout=batch["use_dropout"],
            safe_key=safe_subkey)
        return (evoformer_output, safe_key)

      if gc.use_remat:
        evoformer_fn = hk.remat(evoformer_fn)

      safe_key, safe_subkey = safe_key.split()
      evoformer_stack = layer_stack.layer_stack(c.evoformer_num_block)(
          evoformer_fn)

      def run_evoformer(evoformer_input):
        evoformer_output, _ = evoformer_stack((evoformer_input, safe_subkey))
        return evoformer_output

      if NATIVE_EVOFORMER is not None:
        # eval_shape FIRST so haiku still creates every parameter: the native
        # path never calls the modules, and without this their weights are
        # absent from the param tree and the checkpoint will not load.
        hk.eval_shape(run_evoformer, evoformer_input)
        from ...native import supported_length
        # DECIDE SUPPORT BEFORE DRAWING THE SEED.  The kernels take 32-aligned
        # lengths only, and a campaign hits several lengths (the complex, the
        # binder alone, the validation fold), so one unsupported length used
        # to kill the whole run.  The check is at TRACE time, once per
        # compiled function, and it has to come first: drawing the seed
        # consumes safe_subkey, and the JAX fallback cannot be handed a key
        # that has already been used.
        if supported_length(evoformer_input['pair'].shape[0]):
          # THE SEED COMES FROM AF2'S OWN KEY CHAIN, not from a counter here,
          # so a design run stays reproducible from the model's seed -- it
          # just draws a different mask from it than threefry would.
          evoformer_output = NATIVE_EVOFORMER(
              evoformer_input, evoformer_masks, int(c.evoformer_num_block),
              # [seed, enabled].  `use_dropout` is TRACED -- bindcraft flips
              # it between stages of one trajectory -- so it crosses as a
              # value, not as a static flag that would recompile the stack at
              # every stage boundary.
              jnp.stack([
                  jax.random.bits(safe_subkey.get(), (), jnp.uint32),
                  jnp.asarray(batch["use_dropout"], jnp.uint32).reshape(()),
              ]))
        else:
          print(f'native: complex length '
                f'{evoformer_input["pair"].shape[0]} is not a multiple of 32; '
                'this one length runs on JAX', flush=True)
          evoformer_output = run_evoformer(evoformer_input)
      else:
        evoformer_output = run_evoformer(evoformer_input)

      msa_activations = evoformer_output['msa']
      pair_activations = evoformer_output['pair']
      
      single_activations = common_modules.Linear(
          c.seq_channel, name='single_activations')(msa_activations[0])
    output.update({
        'single':
            single_activations,
        'pair':
            pair_activations,
        # Crop away template rows such that they are not used in MaskedMsaHead.
        'msa':
            msa_activations[:num_msa_sequences, :, :],
        'msa_first_row':
            msa_activations[0],
    })

    # Convert back to float32 if we're not saving memory.
    if not gc.bfloat16_output:
      for k, v in output.items():
        if v.dtype == jnp.bfloat16:
          output[k] = v.astype(jnp.float32)

    return output


class TemplateEmbedding(hk.Module):
  """Embed a set of templates."""

  def __init__(self, config, global_config, name='template_embedding'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, query_embedding, template_batch, padding_mask_2d,
               multichain_mask_2d, use_dropout, safe_key=None,
               _force_jax=False):
    """Generate an embedding for a set of templates.

    Args:
      query_embedding: [num_res, num_res, num_channel] a query tensor that will
        be used to attend over the templates to remove the num_templates
        dimension.
      template_batch: A dictionary containing:
        `template_aatype`: [num_templates, num_res] aatype for each template.
        `template_all_atom_positions`: [num_templates, num_res, 37, 3] atom
          positions for all templates.
        `template_all_atom_mask`: [num_templates, num_res, 37] mask for each
          template.
      padding_mask_2d: [num_res, num_res] Pair mask for attention operations.
      multichain_mask_2d: [num_res, num_res] Mask indicating which residue pairs
        are intra-chain, used to mask out residue distance based features
        between chains.
      safe_key: random key generator.

    Returns:
      An embedding of size [num_res, num_res, num_channels]
    """
    c = self.config
    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())

    num_templates = template_batch['template_aatype'].shape[0]
    num_res, _, query_num_channels = query_embedding.shape

    # THE WHOLE MODULE AS ONE CUSTOM CALL.  It lives here rather than in
    # SingleTemplateEmbedding because that one returns before output_linear,
    # whose weights are in THIS scope -- and it is spelled as a branch of
    # __call__ rather than a sibling method because haiku scopes a module
    # created in any other method under `~_<method>/`, which renames every
    # weight the checkpoint carries.
    #
    # TWO PRECONDITIONS, checked rather than assumed; getting either wrong is
    # silent.  Exactly one template, because the scan and the /num_templates
    # mean are identities only then.  And the constant eight already
    # harvested, because the native call has no distogram, backbone frame or
    # atom position to rebuild them from.
    act_const = template_batch.get('template_act_const')
    if (NATIVE_TEMPLATE_EMBED is not None and not _force_jax
        and num_templates == 1 and act_const is not None):
      from ...native import supported_length
      if supported_length(num_res):
        # SPLIT FIRST.  A SafeKey is single-use and the reference pass
        # consumes this one; reaching for it again afterwards raises
        # 'Random key has been used previously'.
        safe_key, native_key = safe_key.split()
        # the reference pass creates the parameters; its output is discarded
        hk.eval_shape(
            lambda q: self(q, template_batch, padding_mask_2d,
                           multichain_mask_2d, use_dropout, safe_key,
                           _force_jax=True), query_embedding)
        NATIVE_TEMPLATE_EMBED_TRACES[0] += 1
        ac = act_const[0] if act_const.ndim == 4 else act_const
        return NATIVE_TEMPLATE_EMBED(
            query_embedding, ac, padding_mask_2d,
            int(c.template_pair_stack.num_block),
            jnp.stack([
                jax.random.bits(native_key.get(), (), jnp.uint32),
                jnp.asarray(use_dropout, jnp.uint32).reshape(()),
            ]))

    # Embed each template separately.
    template_embedder = SingleTemplateEmbedding(self.config, self.global_config)
    def partial_template_embedder(template_batch, unsafe_key):
      safe_key = prng.SafeKey(unsafe_key)
      return template_embedder(query_embedding,
                               template_batch,
                               padding_mask_2d,
                               multichain_mask_2d,
                               use_dropout,
                               safe_key)

    safe_key, unsafe_key = safe_key.split()
    unsafe_keys = jax.random.split(unsafe_key._key, num_templates)

    def scan_fn(carry, x):
      return carry + partial_template_embedder(*x), None

    scan_init = jnp.zeros((num_res, num_res, c.num_channels), dtype=query_embedding.dtype)
    summed_template_embeddings, _ = hk.scan(scan_fn, scan_init, (template_batch, unsafe_keys))

    embedding = summed_template_embeddings / num_templates
    embedding = jax.nn.relu(embedding)
    embedding = common_modules.Linear(
        query_num_channels,
        initializer='relu',
        name='output_linear')(embedding)

    return embedding


class SingleTemplateEmbedding(hk.Module):
  """Embed a single template."""

  def __init__(self, config, global_config, name='single_template_embedding'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, query_embedding, template_batch,               
               padding_mask_2d, multichain_mask_2d, use_dropout, safe_key):
    """Build the single template embedding graph.

    Args:
      query_embedding: (num_res, num_res, num_channels) - embedding of the
        query sequence/msa.
      template_aatype: [num_res] aatype for each template.
      template_all_atom_positions: [num_res, 37, 3] atom positions for all
        templates.
      template_all_atom_mask: [num_res, 37] mask for each template.
      padding_mask_2d: Padding mask (Note: this doesn't care if a template
        exists, unlike the template_pseudo_beta_mask).
      multichain_mask_2d: A mask indicating intra-chain residue pairs, used
        to mask out between chain distances/features when templates are for
        single chains.
      safe_key: Random key generator.

    Returns:
      A template embedding (num_res, num_res, num_channels).
    """
    gc = self.global_config
    c = self.config
    assert padding_mask_2d.dtype == query_embedding.dtype
    dtype = query_embedding.dtype
    num_channels = self.config.num_channels
    
    def construct_input(query_embedding, template_batch, multichain_mask_2d,
                        const_only=False):

      if "template_dgram" in template_batch:
        template_dgram = template_batch["template_dgram"].astype(dtype)
        template_dgram *= multichain_mask_2d[...,None]
        pseudo_beta_mask_2d = template_dgram.sum(-1)
      
      else:
        # Compute distogram feature for the template.
        template_positions, pseudo_beta_mask = modules.pseudo_beta_fn(
            template_batch["template_aatype"],
            template_batch["template_all_atom_positions"],
            template_batch["template_all_atom_mask"])
        pseudo_beta_mask_2d = (pseudo_beta_mask[:, None] *
                               pseudo_beta_mask[None, :])
        pseudo_beta_mask_2d *= multichain_mask_2d
        template_dgram = modules.dgram_from_positions(
            template_positions, **self.config.dgram_features)
        template_dgram *= pseudo_beta_mask_2d[..., None]
        template_dgram = template_dgram.astype(dtype)    
        pseudo_beta_mask_2d = pseudo_beta_mask_2d.astype(dtype)

      to_concat = [(template_dgram, 1), (pseudo_beta_mask_2d, 0)]
      aatype = jax.nn.one_hot(template_batch["template_aatype"], 22, axis=-1, dtype=dtype)
      to_concat.append((aatype[None, :, :], 1))
      to_concat.append((aatype[:, None, :], 1))

      # Compute a feature representing the normalized vector between each
      # backbone affine - i.e. in each residues local frame, what direction are
      # each of the other residues.
      raw_atom_pos = template_batch["template_all_atom_positions"]
      if gc.bfloat16:
        raw_atom_pos = raw_atom_pos.astype(jnp.float32)
        
      atom_pos = geometry.Vec3Array.from_array(raw_atom_pos)
      rigid, backbone_mask = folding_multimer.make_backbone_affine(
          atom_pos,
          template_batch["template_all_atom_mask"],
          template_batch["template_aatype"])
      points = rigid.translation
      rigid_vec = rigid[:, None].inverse().apply_to_point(points)
      unit_vector = rigid_vec.normalized()
      unit_vector = [unit_vector.x, unit_vector.y, unit_vector.z]

      if gc.bfloat16:
        unit_vector = [x.astype(jnp.bfloat16) for x in unit_vector]
        backbone_mask = backbone_mask.astype(jnp.bfloat16)

      backbone_mask_2d = jnp.sqrt(backbone_mask[:,None] * backbone_mask[None,:])
      backbone_mask_2d *= multichain_mask_2d
      unit_vector = [x*backbone_mask_2d for x in unit_vector]

      # Note that the backbone_mask takes into account C, CA and N (unlike
      # pseudo beta mask which just needs CB) so we add both masks as features.
      to_concat.extend([(x, 0) for x in unit_vector])
      to_concat.append((backbone_mask_2d, 0))          
      # Allow the template embedder to see the query embedding.  Note this
      # contains the position relative feature, so this is how the network knows
      # which residues are next to each other.
      # EIGHT OF THE NINE FEATURES ARE CONSTANT WITHIN A DESIGN STEP.  Each
      # is built from the TEMPLATE -- its aatype, atom positions and atom
      # mask -- and designed residues are masked to aatype 21 whatever they
      # currently are, so changing the sequence changes none of them.  Only
      # query_embedding, the ninth, varies.
      #
      # const_only leaves the ninth off, producing exactly the sum a caller
      # can compute once per target and hand back as template_act_const.
      if not const_only:
        to_concat.append((common_modules.LayerNorm(
            axis=[-1], create_scale=True, create_offset=True,
            name='query_embedding_norm')(query_embedding), 1))

      act = 0

      for i, (x, n_input_dims) in enumerate(to_concat):
        act += common_modules.Linear(
            num_channels,
            num_input_dims=n_input_dims,
            initializer='relu',
            name=f'template_pair_embedding_{i}')(x)
      return act

    def query_projection(query_embedding):
      '''the ninth feature alone.  Both names are construct_input's own, in
      the same scope, so this binds the same weights -- restating the
      arithmetic anywhere else would bind template_pair_embedding_8 from
      nothing.'''
      q = common_modules.LayerNorm(
          axis=[-1], create_scale=True, create_offset=True,
          name='query_embedding_norm')(query_embedding)
      return common_modules.Linear(
          num_channels, num_input_dims=1, initializer='relu',
          name='template_pair_embedding_8')(q)

    if template_batch.get('__const_only__'):
      # the harvest, from inside this module so the eight Linears keep their
      # haiku scope and therefore their weights
      return construct_input(query_embedding, template_batch,
                             multichain_mask_2d, const_only=True)

    act_const = template_batch.get('template_act_const')
    if act_const is None:
      act = construct_input(query_embedding, template_batch, multichain_mask_2d)
    else:
      # the same sum, with the constant eight read instead of rebuilt --
      # removing them from EVERY step, not just from the compile
      act = act_const.astype(dtype) + query_projection(query_embedding)

    template_iteration = TemplateEmbeddingIteration(
        c.template_pair_stack, gc, name='template_embedding_iteration')

    def template_iteration_fn(x):
      act, safe_key = x

      safe_key, safe_subkey = safe_key.split()
      act = template_iteration(
          act=act,
          pair_mask=padding_mask_2d,
          use_dropout=use_dropout,
          safe_key=safe_subkey)
      return (act, safe_key)

    if gc.use_remat:
      template_iteration_fn = hk.remat(template_iteration_fn)

    safe_key, safe_subkey = safe_key.split()
    template_stack = layer_stack.layer_stack(
        c.template_pair_stack.num_block)(
            template_iteration_fn)

    def run_template(x):
      out, _ = template_stack((x, safe_subkey))
      return out

    if NATIVE_TEMPLATE_STACK is not None:
      # eval_shape first so haiku still creates the parameters, as the trunk
      # and the extra-MSA stack do.  Templates are the largest remaining
      # piece of the design step's compile once the extra stack is native.
      hk.eval_shape(run_template, act)
      from ...native import supported_length
      if supported_length(act.shape[0]):
        act = NATIVE_TEMPLATE_STACK(
            act, padding_mask_2d, int(c.template_pair_stack.num_block),
            jnp.stack([
                jax.random.bits(safe_subkey.get(), (), jnp.uint32),
                jnp.asarray(use_dropout, jnp.uint32).reshape(()),
            ]))
      else:
        act = run_template(act)
    else:
      act, safe_key = template_stack((act, safe_subkey))

    act = common_modules.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='output_layer_norm')(
            act)
    return act

class TemplateEmbeddingIteration(hk.Module):
  """Single Iteration of Template Embedding."""

  def __init__(self, config, global_config,
               name='template_embedding_iteration'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, act, pair_mask, use_dropout, safe_key=None):
    """Build a single iteration of the template embedder.

    Args:
      act: [num_res, num_res, num_channel] Input pairwise activations.
      pair_mask: [num_res, num_res] padding mask.
      safe_key: Safe pseudo-random generator key.

    Returns:
      [num_res, num_res, num_channel] tensor of activations.
    """
    c = self.config
    gc = self.global_config

    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())

    dropout_wrapper_fn = functools.partial(
        modules.dropout_wrapper,
        use_dropout=use_dropout,
        global_config=gc)

    safe_key, *sub_keys = safe_key.split(20)
    sub_keys = iter(sub_keys)

    act = dropout_wrapper_fn(
        modules.TriangleMultiplication(c.triangle_multiplication_outgoing, gc,
                                       name='triangle_multiplication_outgoing'),
        act,
        pair_mask,
        safe_key=next(sub_keys))

    act = dropout_wrapper_fn(
        modules.TriangleMultiplication(c.triangle_multiplication_incoming, gc,
                                       name='triangle_multiplication_incoming'),
        act,
        pair_mask,
        safe_key=next(sub_keys))

    act = dropout_wrapper_fn(
        modules.TriangleAttention(c.triangle_attention_starting_node, gc,
                                  name='triangle_attention_starting_node'),
        act,
        pair_mask,
        safe_key=next(sub_keys))

    act = dropout_wrapper_fn(
        modules.TriangleAttention(c.triangle_attention_ending_node, gc,
                                  name='triangle_attention_ending_node'),
        act,
        pair_mask,
        safe_key=next(sub_keys))

    act = dropout_wrapper_fn(
        modules.Transition(c.pair_transition, gc,
                           name='pair_transition'),
        act,
        pair_mask,
        safe_key=next(sub_keys))

    return act


def template_embedding_1d(batch, num_channel, global_config):
  """Embed templates into an (num_res, num_templates, num_channels) embedding.

  Args:
    batch: A batch containing:
      template_aatype, (num_templates, num_res) aatype for the templates.
      template_all_atom_positions, (num_templates, num_residues, 37, 3) atom
        positions for the templates.
      template_all_atom_mask, (num_templates, num_residues, 37) atom mask for
        each template.
    num_channel: The number of channels in the output.

  Returns:
    An embedding of shape (num_templates, num_res, num_channels) and a mask of
    shape (num_templates, num_res).
  """

  # Embed the templates aatypes.
  aatype_one_hot = jax.nn.one_hot(batch['template_aatype'], 22, axis=-1)

  num_templates = batch['template_aatype'].shape[0]
  all_chi_angles = []
  all_chi_masks = []
  for i in range(num_templates):
    atom_pos = geometry.Vec3Array.from_array(
        batch['template_all_atom_positions'][i, :, :, :])
    template_chi_angles, template_chi_mask = all_atom_multimer.compute_chi_angles(
        atom_pos,
        batch['template_all_atom_mask'][i, :, :],
        batch['template_aatype'][i, :])
    all_chi_angles.append(template_chi_angles)
    all_chi_masks.append(template_chi_mask)
  chi_angles = jnp.stack(all_chi_angles, axis=0)
  chi_mask = jnp.stack(all_chi_masks, axis=0)

  template_features = jnp.concatenate([
      aatype_one_hot,
      jnp.sin(chi_angles) * chi_mask,
      jnp.cos(chi_angles) * chi_mask,
      chi_mask], axis=-1)

  template_mask = chi_mask[:, :, 0]

  if global_config.bfloat16:
    template_features = template_features.astype(jnp.bfloat16)
    template_mask = template_mask.astype(jnp.bfloat16)

  template_activations = common_modules.Linear(
      num_channel,
      initializer='relu',
      name='template_single_embedding')(
          template_features)
  template_activations = jax.nn.relu(template_activations)
  template_activations = common_modules.Linear(
      num_channel,
      initializer='relu',
      name='template_projection')(
          template_activations)
  return template_activations, template_mask
