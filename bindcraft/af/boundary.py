"""What a design step hands to a loss, and nothing else.

THIS IS A CONTRACT, NOT A CONVENIENCE.  A loss is written in Python and
differentiated by JAX; it never touches CUDA.  That only holds while
everything a loss could want is on this list, because the step that produces
it is moving into native code -- and once it has, a loss that needs
something NOT here cannot be written without a CUDA change.

So: adding to this list is the supported way to enable a new kind of loss.
Removing from it, or changing a dtype, breaks every loss in the wild.

`differentiable` marks the arrays a gradient flows back through, into the
native backward.  The rest are masks and indices that are constant with
respect to the sequence being designed.
"""
from __future__ import annotations

from typing import NamedTuple


class Field(NamedTuple):
    name: str
    dtype: str
    shape: str            # N is the padded residue count, B the distogram bins
    differentiable: bool
    note: str


# the complex's predicted structure
STRUCTURE = (
    Field('atoms', 'float16', '(N, 37, 3)', True,
          'final_atom_positions, aligned to the target template'),
    Field('atom_mask', 'bool', '(N, 37)', False,
          'final_atom_mask; which of the 37 slots are real'),
)

# everything the confidence heads and the distogram produce
METRICS = (
    Field('plddt', 'float32', '(N,)', True, 'per residue, 0-1'),
    Field('pae', 'float32', '(N, N)', True, 'predicted aligned error'),
    Field('ptm', 'float32', '()', True, 'whole complex'),
    Field('iptm', 'float32', '()', True, 'interface only'),
    Field('iptm_per_residue', 'float32', '(N,)', True,
          'interface pTM attributed per residue'),
    Field('experimentally_resolved_ca', 'float32', '(N,)', True,
          'the experimentally-resolved head, CA slot'),
    Field('distogram', 'float32', '(N, N, B)', True,
          'pair distance logits; B is 64 for AF2'),
)

# what the loss ALSO receives, unchanged from its own inputs -- these do not
# come from the model and carry no gradient from it
INPUT_SIDE = (
    Field('sequence', 'float16', '(N, 20)', True,
          'the designed sequence; differentiable because the loss is what '
          'drives it'),
    Field('flags', 'uint8', '(N,)', False, 'per-residue role bits'),
    Field('residue_index', 'int32', '(N,)', False, 'author numbering'),
)

ALL = STRUCTURE + METRICS + INPUT_SIDE
DIFFERENTIABLE = tuple(f.name for f in ALL if f.differentiable)


def describe() -> str:
    rows = [f'{f.name:<28} {f.dtype:<8} {f.shape:<12} '
            f'{"grad" if f.differentiable else "    "}  {f.note}' for f in ALL]
    return '\n'.join(rows)


if __name__ == '__main__':
    print(describe())
