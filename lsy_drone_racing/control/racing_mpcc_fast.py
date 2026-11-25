"""MPCC racing controller tuned for the shortest lap time.

Same controller as :mod:`racing_mpcc`, flown more aggressively: higher progress reward and higher
speed limits before the gates that are still unknown (much higher before gates 1 and 2, slightly
higher before gate 0, whose late reveal causes most crashes when it is approached too fast). The
last gate keeps the safer approach of :mod:`racing_mpcc_finish` (limited speed, straight final
approach), and the progress no longer falls back after the stuck detection made it jump.

Simulation (level 2, 100 identical episodes): 45 % success, 4.79 s (racing_mpcc: 57 %, 5.15 s).
Select it with ``python scripts/deploy.py --controller racing_mpcc_fast.py`` (or ``sim.py``).
"""

from __future__ import annotations

from dataclasses import replace

from lsy_drone_racing.control.racing_mpcc import MPCCTuning, RacingMPCC


class RacingMPCCFast(RacingMPCC):
    """MPCC racing controller, aggressive tuning for the shortest lap time."""

    TUNING = replace(
        MPCCTuning(),
        q_progress=7.0,
        v_unknown=3.5,
        v_unknown0=2.5,
        v_unknown3=3.0,
        v_last=3.5,
        last_before=1.2,
        last_pre_d=0.5,
        stuck_hold=True,
    )
