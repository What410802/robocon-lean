"""Plum (梅花桩) expert: privileged teacher -> depth-CNN student.

Method (ported from the old repo, rebuilt for the rules-PDF 6-cube field):

1. A PPO *teacher* sees proprio history (54x6=324) plus a privileged
   per-wheel fine height scan; it is trained on the exact plum field.
2. A *student* replaces that scan with a small CNN fed by one depth frame.
   The teacher MLP is FROZEN: only the CNN + a linear scan head are trained,
   by regressing the ground-truth scan and matching the teacher's actions.
3. The student is exported as TorchScript ``policy(obs_324, [depth])`` plus a
   ``depth.json`` camera contract, and scored by the eval-side policy runner.

Everything geometric comes from ``spec.py`` -> ``terrain/plum/plum_spec.json``.
No constants from the old repo's 24-cube ramped field are used.
"""

from .spec import SPEC, PlumSpec, load_spec  # noqa: F401

__all__ = ["SPEC", "PlumSpec", "load_spec"]
