"""MPCC racing controller tuned for a high success rate at a moderate speed.

Same controller as :mod:`racing_mpcc`, flown more conservatively: lower progress reward, lower
speed limits before the gates that are still unknown, lower maximum tilt and progress speed, and
the safer last-gate approach of :mod:`racing_mpcc_finish`. The start detour keeps more distance
from gate 2, whose frame the slower drone could otherwise touch, and the progress no longer falls
back after the stuck detection made it jump.

Simulation (level 2, 100 identical episodes): 84 % success, 6.33 s.
Select it with ``python scripts/deploy.py --controller racing_mpcc_reliable.py`` (or ``sim.py``).
"""

from __future__ import annotations

from dataclasses import replace

from lsy_drone_racing.control.racing_mpcc import MPCCTuning, RacingMPCC


class RacingMPCCReliable(RacingMPCC):
    """MPCC racing controller, conservative tuning for a high success rate."""

    TUNING = replace(
        MPCCTuning(),
        q_progress=4.0,
        max_tilt=0.9,
        v_max=4.0,
        v_unknown=1.8,
        v_unknown0=1.8,
        v_unknown3=1.8,
        v_last=2.5,
        last_before=1.2,
        last_pre_d=0.5,
        stuck_hold=True,
        start_via=(-1.0, 0.45, 0.4),
    )
