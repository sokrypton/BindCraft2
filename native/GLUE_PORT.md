# Porting the glue: what is left, and what is known about it

The goal: a design step where **only the loss is JAX**, so a user writes new
losses in Python and never touches CUDA. The contract for that is
`bindcraft/af/boundary.py`, checked by `native/boundary_gate.py`.

Already native: the trunk, the extra-MSA stack, the template pair stack, the
whole template embedding, the embedder, and the post-trunk structure module
and heads. What follows is everything else between a sequence and the nine
boundary arrays.

## Measured

| | |
| --- | --- |
| full design step compile | **2.9 s** (N=192, bucket 32, dropout off) |
| the loss and its gradient alone | **0.54 s** (`native/loss_floor.py`) |
| so a complete glue port is worth | **~2.4 s per compile, and no more** |
| feature construction alone | **0.23 s** forward, 0.03 s with gradient |
| the whole backward | **0.3 s** |
| compile recurs | **once per 32-bucket**, not per step or per length |
| step time | 755 ms; this work does not move it |

A campaign pays the 2.9 s a few times, so **the compile saving is not the
reason to do this**. The reason is below.

## The real prize: one graph for every length

A custom call takes STATIC shapes, and the pad-to-32 happens in JAX before
the call, so today the whole module is specialised to the padded length.
That is why L=179, 180 and 185 share a graph and L=200 recompiles.

If the native step absorbs the padding and takes the length as a RUNTIME
input, one graph serves every length and the recompile disappears. Same
work, structural win rather than 2.4 s.

## What has to move

In rough dependency order. Sizes are "what it does", not line counts --
these are dense one-liners.

1. **`prepare_design_sequence_features`** -- softmax / one-hot / temperature
   / logit scale / amino-acid bias. Small, pure, no state. Easiest first.
2. **`alphafold_input_features`** -- the big one. ~25 features: msa_feat,
   atom14 and atom37 masks, the residx_atom14_to_atom37 maps and their
   inverse, `target_template_features` (template aatype, positions, mask,
   pseudo-beta), `cyclic_sequence_offsets`, and the zeroed `prev`.
   Mostly gathers off constant tables, so mechanical but wide.
3. **The 32-padding** -- 12 kernels today, and the thing that must move for
   the length to become runtime.
4. **In-model glue** -- `prev_pos_linear` and `template_single_embedding`
   einsums, `_threefry_split` RNG plumbing, dtype converts and stacks.
5. **`alphafold_prediction_metrics`** -- plddt, pae, ptm, iptm,
   iptm_per_residue from the head logits. Produces five of the nine
   boundary arrays, so it is gated directly by `boundary_gate.py`.
6. **`align_prediction_to_target_template`** -- Kabsch, and the one piece
   needing a linear-algebra routine: it is the `cusolver_gesvdj` call in the
   current graph. Needs an SVD or a Kabsch that avoids one.
7. **The state loop** -- two states per step today (complex, binder_alone).
   Either stays in Python around the call, or becomes a batch dimension.

## Not measured, and worth knowing before committing

- The compile cost of each item above, separately. Only feature
  construction (0.23 s) has been isolated. **The 2.4 s is a ceiling for the
  whole set, not a budget per item**, and items may interact.
- Whether the backward of each is as cheap as the forward.
- Whether a runtime length costs step time. A kernel written for a compile
  time constant can be faster than one reading a length from memory.

## Two things that will bite

- **The glue is the churn.** Feature construction and the state loop are
  where new BindCraft features land. Porting them moves churn-prone code
  into CUDA.
- **The boundary is a ceiling on losses.** A loss wanting the pair
  representation, the MSA, or per-block activations is impossible without a
  CUDA change once this is done. Widen `boundary.py` BEFORE, not after.
