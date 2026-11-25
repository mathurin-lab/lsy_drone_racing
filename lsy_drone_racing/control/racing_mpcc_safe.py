"""MPCC racing controller with the largest safety margins (slowest variant).

Like :mod:`racing_mpcc_reliable` with an even lower progress reward and maximum tilt: the drone
flies at moderate attitudes everywhere, which leaves the largest margins against the model mismatch
and the latency of the real system. Suited for the first flights on a new track or drone.

Simulation (level 2, 100 identical episodes): 86 % success, 7.27 s.
Select it with ``python scripts/deploy.py --controller racing_mpcc_safe.py`` (or ``sim.py``).
"""

from __future__ import annotations

from dataclasses import replace

from lsy_drone_racing.control.racing_mpcc_reliable import RacingMPCCReliable


class RacingMPCCSafe(RacingMPCCReliable):
    """MPCC racing controller, slowest and most conservative tuning."""

    TUNING = replace(RacingMPCCReliable.TUNING, q_progress=3.0, max_tilt=0.8)
