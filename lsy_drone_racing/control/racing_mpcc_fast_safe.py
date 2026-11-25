"""MPCC racing controller: the fast tuning with a safer approach to gate 3.

Same tuning as :mod:`racing_mpcc_fast`, except before gate 3, which is passed behind a pole lying on
its axis and revealed only about 0.6 m before the drone reaches it: there the speed is limited to
2.2 m/s (instead of 3 m/s) until the reveal, so that the drone can still correct its approach
afterwards. The gate tunnel also starts a little earlier.

Simulation (level 2, 400 episodes not used for the tuning): 52 % success, 4.99 s
(racing_mpcc_fast: 42 %, 4.80 s on the same episodes); the crashes at gate 3 drop by 44 %.
Select it with ``python scripts/deploy.py --controller racing_mpcc_fast_safe.py`` (or ``sim.py``).
"""

from __future__ import annotations

from dataclasses import replace

from lsy_drone_racing.control.racing_mpcc_fast import RacingMPCCFast


class RacingMPCCFastSafe(RacingMPCCFast):
    """MPCC racing controller, fast tuning with a slower approach to gate 3."""

    TUNING = replace(RacingMPCCFast.TUNING, v_unknown3=2.2, tunnel_half_len=0.39)
