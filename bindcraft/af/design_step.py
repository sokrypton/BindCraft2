"""One call from a sequence to the nine arrays a loss reads.

SCAFFOLDING.  The body is still the existing JAX pipeline; the point of
this module is the SEAM.  Fixing the signature first means the glue can
move behind it one piece at a time without the callers, the losses or the
gate changing -- and `native/boundary_gate.py` already checks the output
side of it.

The signature is the native one (dev/native_block/design_step.cuh), not
today's:

  * buffers are sized NMAX and `length` says how much of them is real
  * `length` is a VALUE, not a shape

Until the native call lands, `length` is carried and not used, because JAX
specialises on shapes regardless -- so this does NOT yet give one graph for
every length.  That arrives with the kernel, and it is the only part of the
port worth the work; see native/GLUE_PORT.md.
"""
from __future__ import annotations

from typing import NamedTuple

from jax import Array


class DesignStepOutputs(NamedTuple):
    """The nine, in the order bindcraft/af/boundary.py lists them."""
    atoms: Array                    # [NMAX, 37, 3] float16
    atom_mask: Array                # [NMAX, 37]    bool
    plddt: Array                    # [NMAX]
    pae: Array                      # [NMAX, NMAX]
    ptm: Array                      # []
    iptm: Array                     # []
    iptm_per_residue: Array         # [NMAX]
    experimentally_resolved_ca: Array   # [NMAX]
    distogram: Array                # [NMAX, NMAX, 64]

    def as_metrics(self) -> dict[str, Array]:
        """The seven that travel as `metrics`, named as the loss sees them."""
        return {'plddt': self.plddt, 'pae': self.pae, 'ptm': self.ptm,
                'iptm': self.iptm, 'iptm_per_residue': self.iptm_per_residue,
                'experimentally_resolved_ca': self.experimentally_resolved_ca,
                'distogram': self.distogram}


def from_prediction(predicted_atom_positions: Array, predicted_atom_mask: Array,
                    metrics: dict[str, Array]) -> DesignStepOutputs:
    """Adapt today's pipeline output into the contract's shape.

    The one place that knows both spellings, so when the native call lands
    only this function changes.
    """
    return DesignStepOutputs(
        atoms=predicted_atom_positions,
        atom_mask=predicted_atom_mask,
        plddt=metrics['plddt'], pae=metrics['pae'], ptm=metrics['ptm'],
        iptm=metrics['iptm'], iptm_per_residue=metrics['iptm_per_residue'],
        experimentally_resolved_ca=metrics['experimentally_resolved_ca'],
        distogram=metrics['distogram'])
