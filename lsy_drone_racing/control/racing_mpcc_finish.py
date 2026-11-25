"""MPCC racing controller with a safer approach to the last gate.

Same tuning as :mod:`racing_mpcc` (same speed on the rest of the track), except before the last gate
of the sequence: there the drone otherwise arrives at its top speed (> 4 m/s, nothing follows the
last gate), where the model mismatch and the latency of the real system make it hit the frame. The
speed is limited before that gate and an extra approach point on its axis straightens the final
approach. The progress also no longer falls back after the stuck detection made it jump.

Simulation (level 2, 100 identical episodes): 60 % success, 5.27 s (racing_mpcc: 57 %, 5.15 s);
no crash at the last gate left (racing_mpcc: 3 %).

Select it with ``python scripts/deploy.py --controller racing_mpcc_finish.py`` (or ``sim.py``).
"""

from __future__ import annotations

from dataclasses import replace

from lsy_drone_racing.control.racing_mpcc import MPCCTuning, RacingMPCC


class RacingMPCCFinish(RacingMPCC):
    """MPCC racing controller, limited speed and straight approach at the last gate."""

    TUNING = replace(MPCCTuning(), v_last=3.0, last_before=1.2, last_pre_d=0.5, stuck_hold=True)
