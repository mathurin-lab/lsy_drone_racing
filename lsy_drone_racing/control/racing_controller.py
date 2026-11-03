"""Racing controller that plans a spline through the gates and re-plans online.

The controller builds waypoints directly from the gate poses: for every gate in the race sequence
it adds a point in front of the gate, the gate center, and a point behind the gate, all aligned
with the gate's x-axis (the direction in which a gate has to be crossed). A cubic spline with a
distance-based time allocation is fitted through these waypoints. Obstacles (poles) and the frames
and stands of the gates are treated as vertical pillars; whenever the sampled trajectory comes too
close to a pillar, a detour waypoint is inserted and the spline is refitted.

The spline only defines the geometry of the path. It is re-timed afterwards: the path is
parameterized by arc length and a velocity profile is computed that respects a maximum speed, a
lateral acceleration limit (curvature) and a longitudinal acceleration limit (forward/backward
pass). The drone therefore flies fast on straight segments and only slows down in tight turns.

The environment only reveals the exact gate and obstacle poses once the drone is within sensor
range. Whenever the observed poses change, the remaining trajectory is re-planned starting from the
current reference state, so the commanded trajectory stays continuous.

The controller outputs full state commands (position, velocity and acceleration feed-forward),
i.e. it requires ``control_mode = "state"`` in the config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


# Tuning parameters
CRUISE_SPEED = 1.2  # Average speed used for the time allocation [m/s]
TAKEOFF_SPEED = 1.0  # Average speed of the first segment from the ground [m/s]
CLIMB_FRACTION = 0.25  # Height of the takeoff waypoint as a fraction of the first gate height
MAX_ACC = 2.5  # Maximum acceleration of the reference trajectory [m/s^2]
TIME_STRETCH_ITERS = 15  # Iterations of the segment time stretching
# Velocity profile along the path (re-timing)
MAX_SPEED = 3.0  # Maximum speed along the path [m/s]
MAX_LAT_ACC = 3.0  # Maximum lateral (centripetal) acceleration [m/s^2]
MAX_LON_ACC = 2.5  # Maximum longitudinal acceleration / deceleration [m/s^2]
RETIME_DS = 0.02  # Arc length resolution of the velocity profile [m]
# Gates whose exact pose has not been observed yet may still move by up to 0.15 m. Near them the
# speed is limited, so the drone can still follow the re-planned path once the pose is revealed.
UNKNOWN_GATE_SPEED = 10.0  # Speed limit near gates that have not been observed yet [m/s]
UNKNOWN_GATE_RADIUS = 0.8  # Distance from such a gate within which the limit applies [m]
UNKNOWN_OBSTACLE_SPEED = 10.0  # Same for obstacles that have not been observed yet [m/s]
UNKNOWN_OBSTACLE_RADIUS = 0.8  # [m]
# The velocity profile switches the acceleration instantly, which the drone cannot follow (it first
# has to tilt). The re-timed trajectory is therefore smoothed in time with a Gaussian kernel, which
# limits the jerk.
SMOOTH_SIGMA = 0.15  # Standard deviation of the smoothing kernel [s]
SMOOTH_DT = 0.01  # Time resolution of the smoothed trajectory [s]
GATE_APPROACH_DIST = 0.35  # Preferred distance of the pre/post gate waypoints from the center [m]
GATE_APPROACH_MIN = 0.12  # Minimum distance of the pre/post waypoints if a pole is in the way [m]
# If a pole blocks a pre/post waypoint, the waypoint is first shifted sideways (towards the previous
# or next waypoint) before its distance is reduced
GATE_APPROACH_SHIFT = 0.35  # Maximum sideways shift [m]
MIN_WAYPOINT_DIST = 0.08  # Waypoints closer than this are merged [m]
DRONE_RADIUS = 0.1  # Clearance between the drone center and the inner gate opening edges [m]
GATE_OUTER_CLEARANCE = 0.15  # Clearance when flying around (not through) a gate frame [m]
GATE_PLANE_CLEARANCE = 0.12  # Clearance from the gate plane within which the frame is checked [m]
POLE_CLEARANCE = 0.15  # Clearance between the drone center and obstacle pole surfaces [m]
# The onboard state controller of the simulator settles ~0.105 m above the commanded height (hover
# and level flight). We cancel this identified offset with a feed-forward term.
Z_OFFSET_COMPENSATION = 0.105  # [m]
POS_INTEGRAL_GAIN = 1.5  # Gain of the outer position integral correction [1/s]
POS_INTEGRAL_MAX = 0.15  # Saturation of the integral correction per axis [m]
# The onboard position gains are low. Shifting the commanded position/velocity by a multiple of the
# current error effectively increases them: kp_eff = kp * (1 + TRACK_P_BOOST), same for kd.
TRACK_P_BOOST = 1.0
TRACK_D_BOOST = 0.5
TRACK_BOOST_MAX = 0.3  # Saturation of the boost offsets [m] / [m/s]
# The drone lags behind the commanded state (attitude dynamics). The setpoint is therefore sampled
# slightly ahead in time along the trajectory.
LOOKAHEAD = 0.0  # [s]
# The acceleration feed-forward is applied through the attitude, which needs time to change. It is
# therefore taken from a point ahead in time.
ACC_LOOKAHEAD = 0.0  # [s]
POLE_RADIUS = 0.015  # Radius of the obstacle poles [m]
POLE_HEIGHT = 1.55  # Height of the obstacle poles [m]
GATE_HALF_SIZE = 0.36  # Half of the outer gate frame size [m]
GATE_OPENING_HALF = 0.2  # Half of the free gate opening [m]
GATE_DEPTH_HALF = 0.03  # Half of the gate frame thickness [m]
GATE_STAND_RADIUS = 0.05  # Radius of the gate stand [m]
DETOUR_EXTRA = 0.05  # Additional distance of detour waypoints beyond the safety radius [m]
MAX_DETOURS = 25  # Maximum number of detour waypoints inserted per plan
REPLAN_THRESHOLD = 0.005  # Minimum change in observed poses that triggers a re-plan [m]
SAMPLE_DT = 0.02  # Time resolution for the collision check [s]


class SampledTrajectory:
    """Trajectory given by time-stamped samples of position, velocity and acceleration.

    Mimics the call interface of scipy splines: ``traj(t)`` returns the position, ``traj(t, 1)``
    the velocity and ``traj(t, 2)`` the acceleration, linearly interpolated between samples.
    """

    def __init__(
        self,
        t: NDArray[np.floating],
        pos: NDArray[np.floating],
        vel: NDArray[np.floating],
        acc: NDArray[np.floating],
    ):
        """Store the samples.

        Args:
            t: Strictly increasing sample times with shape (N,).
            pos: Positions with shape (N, 3).
            vel: Velocities with shape (N, 3).
            acc: Accelerations with shape (N, 3).
        """
        self.x = t
        self._data = (pos, vel, acc)

    def __call__(self, t: float | NDArray[np.floating], nu: int = 0) -> NDArray[np.floating]:
        """Evaluate the nu-th derivative (0: position, 1: velocity, 2: acceleration) at t."""
        values = self._data[nu]
        t = np.asarray(t, dtype=float)
        out = np.stack([np.interp(t, self.x, values[:, k]) for k in range(3)], axis=-1)
        return out


class RacingController(Controller):
    """Spline-based racing controller with online re-planning."""

    # Limits of the velocity profile. Subclasses with a better trajectory tracking can raise them.
    max_speed = MAX_SPEED
    max_lat_acc = MAX_LAT_ACC
    max_lon_acc = MAX_LON_ACC
    unknown_gate_speed = UNKNOWN_GATE_SPEED
    unknown_obstacle_speed = UNKNOWN_OBSTACLE_SPEED

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller and plan the initial trajectory.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._tick = 0
        self._finished = False
        self._gates_pos = np.asarray(obs["gates_pos"], dtype=float).copy()
        self._gates_quat = np.asarray(obs["gates_quat"], dtype=float).copy()
        self._obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float).copy()
        self._gates_visited = np.asarray(obs["gates_visited"], dtype=bool).copy()
        self._obstacles_visited = np.asarray(obs["obstacles_visited"], dtype=bool).copy()
        self._gate_sequence = np.asarray(obs["gate_sequence"], dtype=int)
        direction = obs.get("gate_sequence_direction")
        self._gate_direction = (
            np.ones(len(self._gate_sequence))
            if direction is None
            else np.sign(np.asarray(direction, dtype=float))
        )
        self._pos_integral = np.zeros(3)
        self._spline: SampledTrajectory | None = None
        self._plan(
            start_pos=np.asarray(obs["pos"], dtype=float),
            start_vel=np.zeros(3),
            n_gates_passed=0,
            t_start=0.0,
            takeoff=True,
        )

    # region Planning

    def _gate_normal(self, gate_id: int, sequence_idx: int) -> NDArray[np.floating]:
        """Direction in which the gate has to be crossed (world frame, unit vector)."""
        normal = R.from_quat(self._gates_quat[gate_id]).apply([1.0, 0.0, 0.0])
        return normal * self._gate_direction[sequence_idx]

    def _gate_waypoints(
        self, start_pos: NDArray[np.floating], n_gates_passed: int
    ) -> tuple[list[NDArray[np.floating]], list[int]]:
        """Waypoints through all remaining gates.

        Returns:
            The waypoints and, for each waypoint, the sequence index of the gate it belongs to.
        """
        waypoints, owners = [], []
        sequence = range(n_gates_passed, len(self._gate_sequence))
        centers = [self._gates_pos[self._gate_sequence[i]] for i in sequence]
        for k, seq_idx in enumerate(sequence):
            gate_id = self._gate_sequence[seq_idx]
            center = self._gates_pos[gate_id]
            normal = self._gate_normal(gate_id, seq_idx)
            prev_point = centers[k - 1] if k > 0 else start_pos
            next_point = centers[k + 1] if k + 1 < len(centers) else center + normal
            pre = self._approach_point(center, -normal, prev_point)
            post = self._approach_point(center, normal, next_point)
            points = [pre, center, post]
            # For the current target gate, skip the points the reference is already past (the
            # reference may lead the drone, which has then not yet been counted as passing the gate)
            if seq_idx == n_gates_passed:
                points = [q for q in points if np.dot(start_pos - q, normal) < -0.05] or [post]
            waypoints.extend(points)
            owners.extend([seq_idx] * len(points))
        return waypoints, owners

    def _approach_point(
        self,
        center: NDArray[np.floating],
        direction: NDArray[np.floating],
        toward: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Pre/post waypoint of a gate that keeps clear of all poles.

        Args:
            center: Gate center.
            direction: Unit vector from the center to the waypoint (gate normal or its opposite).
            toward: The previous/next waypoint. Sideways shifts prefer its side of the gate.

        Returns:
            The waypoint at the preferred distance, shifted sideways or moved closer to the gate if
            a pole is in the way.
        """
        min_clearance = POLE_RADIUS + POLE_CLEARANCE + DETOUR_EXTRA
        side = np.cross(direction, [0.0, 0.0, 1.0])
        side = side / (np.linalg.norm(side) + 1e-9)
        if np.dot(toward - center, side) < 0:
            side = -side
        shifts = [0.0]
        if GATE_APPROACH_SHIFT > 0:
            steps = np.linspace(0.0, GATE_APPROACH_SHIFT, 4)[1:]
            shifts += [x for step in steps for x in (step, -step)]
        for dist in np.linspace(GATE_APPROACH_DIST, GATE_APPROACH_MIN, 6):
            for shift in shifts:
                point = center + dist * direction + shift * side
                d = np.linalg.norm(self._obstacles_pos[:, :2] - point[:2], axis=1)
                if np.all(d > min_clearance):
                    return point
        return center + GATE_APPROACH_MIN * direction

    def _pillars(self) -> list[tuple[NDArray[np.floating], float, float, float]]:
        """Vertical cylinders to avoid as (xy, radius, z_min, z_max): poles and gate stands."""
        pillars = []
        for pos in self._obstacles_pos:
            pillars.append((pos[:2], POLE_RADIUS + POLE_CLEARANCE, 0.0, POLE_HEIGHT + 0.1))
        for pos in self._gates_pos:
            z_lo = pos[2] - GATE_HALF_SIZE
            pillars.append((pos[:2], GATE_STAND_RADIUS + DRONE_RADIUS, 0.0, z_lo - 0.05))
        return pillars

    def _worst_collision(
        self,
        samples: NDArray[np.floating],
        spline: CubicSpline,
        t: NDArray[np.floating],
        active_gate: NDArray[np.integer],
    ) -> tuple[float, int, NDArray[np.floating]] | None:
        """Find the deepest collision of the sampled trajectory.

        Args:
            samples: Sampled positions of the trajectory.
            spline: The trajectory.
            t: Sample times.
            active_gate: For each sample, the ID of the gate the drone is currently flying to.
                Only this gate may resolve a frame collision by moving into its opening; all other
                gates are avoided by flying around them.

        Returns:
            None if collision free, otherwise the penetration depth, the index of the colliding
            sample and a detour waypoint that resolves the collision.
        """
        worst = None
        for xy, radius, z_lo, z_hi in self._pillars():
            d = np.linalg.norm(samples[:, :2] - xy, axis=1)
            in_z = (samples[:, 2] > z_lo) & (samples[:, 2] < z_hi)
            pen = np.where(in_z, radius - d, -np.inf)
            k = int(np.argmax(pen))
            if pen[k] > 0 and (worst is None or pen[k] > worst[0]):
                away = samples[k, :2] - xy
                if np.linalg.norm(away) < 1e-6:  # Exactly on the pillar: push sideways
                    vel = spline(t[k], 1)[:2]
                    away = np.array([-vel[1], vel[0]])
                away = away / (np.linalg.norm(away) + 1e-9)
                detour = np.array([*(xy + away * (radius + DETOUR_EXTRA)), samples[k, 2]])
                worst = (pen[k], k, detour)

        if worst is not None:  # Resolve pillars first, their margins are the tightest
            return worst

        # Gate frames: a square ring in the gate's y-z plane. A sample collides if it is close to
        # the gate plane, inside the outer square but outside the free opening.
        outer = GATE_HALF_SIZE + GATE_OUTER_CLEARANCE
        inner = GATE_OPENING_HALF - DRONE_RADIUS
        for gate_id, (pos, quat) in enumerate(zip(self._gates_pos, self._gates_quat)):
            rot = R.from_quat(quat)
            local = rot.apply(samples - pos, inverse=True)
            near_plane = np.abs(local[:, 0]) < GATE_DEPTH_HALF + GATE_PLANE_CLEARANCE
            ay, az = np.abs(local[:, 1]), np.abs(local[:, 2])
            in_outer = (ay < outer) & (az < outer)
            in_opening = (ay < inner) & (az < inner)
            hit = near_plane & in_outer & ~in_opening
            if not np.any(hit):
                continue
            # Escape distances: into the opening, past the sides, or over the top
            to_opening = np.maximum(ay - inner, 0.0) + np.maximum(az - inner, 0.0)
            to_opening = np.where(active_gate == gate_id, to_opening, np.inf)
            to_side = outer - ay
            to_top = outer - local[:, 2]
            escape = np.minimum(np.minimum(to_opening, to_side), to_top)
            pen = np.where(hit, escape, -np.inf)
            k = int(np.argmax(pen))
            if worst is not None and pen[k] <= worst[0]:
                continue
            target = local[k].copy()
            margin = DETOUR_EXTRA
            if to_opening[k] <= min(to_side[k], to_top[k]):
                target[1] = np.clip(target[1], -inner + margin, inner - margin)
                target[2] = np.clip(target[2], -inner + margin, inner - margin)
            elif to_side[k] <= to_top[k]:
                target[1] = np.sign(target[1] + 1e-9) * (outer + margin)
            else:
                target[2] = outer + margin
            worst = (pen[k], k, rot.apply(target) + pos)
        return worst

    @staticmethod
    def _dedupe(
        points: list[NDArray[np.floating]], owners: list[int]
    ) -> tuple[list[NDArray[np.floating]], list[int]]:
        """Remove consecutive waypoints that are too close to each other."""
        kept, kept_owners = [points[0]], [owners[0]]
        for point, owner in zip(points[1:], owners[1:]):
            if np.linalg.norm(point - kept[-1]) >= MIN_WAYPOINT_DIST:
                kept.append(point)
                kept_owners.append(owner)
        return kept, kept_owners

    @staticmethod
    def _fit(
        points: list[NDArray[np.floating]],
        start_vel: NDArray[np.floating],
        t_start: float,
        takeoff: bool,
    ) -> tuple[CubicSpline, NDArray[np.floating]]:
        """Fit a cubic spline and stretch segments whose acceleration exceeds the limit.

        The initial time allocation is distance based. Since the acceleration of a segment scales
        with the inverse square of its duration, segments violating ``MAX_ACC`` are slowed down
        iteratively until the whole spline is feasible.
        """
        pts = np.array(points)
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        speeds = np.full(len(dists), CRUISE_SPEED)
        if takeoff:
            speeds[0] = TAKEOFF_SPEED
        durations = np.maximum(dists / speeds, 0.05)
        for _ in range(TIME_STRETCH_ITERS):
            knots = t_start + np.concatenate(([0.0], np.cumsum(durations)))
            spline = CubicSpline(knots, pts, bc_type=((1, start_vel), (2, np.zeros(3))))
            # Peak acceleration of each segment (at the knots and inside the segment)
            frac = np.linspace(0.0, 1.0, 6)
            t = knots[:-1, None] + frac[None, :] * durations[:, None]
            acc = np.linalg.norm(spline(t, 2), axis=-1).max(axis=1)
            too_fast = acc > MAX_ACC
            if not np.any(too_fast):
                break
            durations[too_fast] *= np.minimum(np.sqrt(acc[too_fast] / MAX_ACC), 1.3)
        return spline, knots

    def _plan(
        self,
        start_pos: NDArray[np.floating],
        start_vel: NDArray[np.floating],
        n_gates_passed: int,
        t_start: float,
        takeoff: bool = False,
        start_acc: NDArray[np.floating] | None = None,
    ):
        """Plan a collision-free trajectory through the remaining gates."""
        gate_points, owners = self._gate_waypoints(start_pos, n_gates_passed)
        if not gate_points:
            return
        points = [start_pos] + gate_points
        owners = [-1] + owners
        if takeoff:  # Climb first to avoid sweeping over the ground
            first = gate_points[0]
            climb = start_pos + np.array([0.0, 0.0, CLIMB_FRACTION * (first[2] - start_pos[2])])
            points.insert(1, climb)
            owners.insert(1, -1)
        points, owners = self._dedupe(points, owners)
        # The spline only defines the geometry, its time allocation assumes the cruise speed. The
        # start tangent must use the same speed scale, otherwise a fast start velocity makes the
        # path bulge out.
        speed = float(np.linalg.norm(start_vel))
        geo_vel = start_vel * min(1.0, CRUISE_SPEED / speed) if speed > 1e-6 else start_vel

        start_acc = np.zeros(3) if start_acc is None else start_acc
        spline, knots = self._fit(points, geo_vel, t_start, takeoff)
        for _ in range(MAX_DETOURS):
            # The collision check runs on the final (re-timed and smoothed) trajectory, because
            # the smoothing slightly cuts corners
            traj, t_end = self._retime(spline, knots[0], knots[-1], start_vel, start_acc)
            t = np.arange(traj.x[0], t_end, SAMPLE_DT)
            samples = traj(t)
            # Parameter of the geometric spline closest to each sample
            t_geo_grid = np.arange(knots[0], knots[-1], 0.5 * SAMPLE_DT)
            geo_pts = spline(t_geo_grid)
            closest = np.argmin(
                np.linalg.norm(samples[:, None, :] - geo_pts[None, :, :], axis=-1), axis=1
            )
            t_geo = t_geo_grid[closest]
            # Gate the drone is heading to at each sample: owner of the next gate waypoint
            next_owner = np.full(len(owners) + 1, -1)
            for i in range(len(owners) - 1, -1, -1):
                next_owner[i] = owners[i] if owners[i] >= 0 else next_owner[i + 1]
            seg = np.clip(np.searchsorted(knots, t_geo, side="right"), 1, len(owners) - 1)
            seq = next_owner[seg]
            active_gate = np.where(seq >= 0, self._gate_sequence[np.maximum(seq, 0)], -1)
            worst = self._worst_collision(samples, traj, t, active_gate)
            if worst is None:
                break
            _, k, detour = worst
            insert_idx = int(np.searchsorted(knots, t_geo[k]))
            insert_idx = min(max(insert_idx, 1), len(points))
            points.insert(insert_idx, detour)
            owners.insert(insert_idx, -1)
            points, owners = self._dedupe(points, owners)
            spline, knots = self._fit(points, geo_vel, t_start, takeoff)
        else:
            traj, t_end = self._retime(spline, knots[0], knots[-1], start_vel, start_acc)

        self._waypoints = np.array(points)
        self._spline, self._t_end = traj, t_end

    def _retime(
        self,
        spline: CubicSpline,
        t_start: float,
        t_end: float,
        start_vel: NDArray[np.floating],
        start_acc: NDArray[np.floating],
    ) -> tuple[SampledTrajectory, float]:
        """Re-time the path of a spline with a curvature-limited velocity profile.

        Args:
            spline: Spline whose geometry defines the path.
            t_start: Start time of the spline.
            t_end: End time of the spline.
            start_vel: Current reference velocity (the profile starts at its magnitude).
            start_acc: Current reference acceleration (continued before the start for smoothing).

        Returns:
            The re-timed trajectory starting at ``t_start`` and its end time.
        """
        # Arc length parameterization of the geometric path
        tau = np.linspace(t_start, t_end, max(int((t_end - t_start) / 0.005), 20))
        dense = spline(tau)
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))))
        n = max(int(arc[-1] / RETIME_DS), 2)
        s = np.linspace(0.0, arc[-1], n + 1)
        pts = np.stack([np.interp(s, arc, dense[:, k]) for k in range(3)], axis=1)
        ds = s[1] - s[0]
        # Curvature of the path
        geo = CubicSpline(s, pts)
        d1, d2 = geo(s, 1), geo(s, 2)
        speed = np.maximum(np.linalg.norm(d1, axis=1), 1e-6)
        kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / speed**3
        # Speed limits: curvature, then forward (acceleration) and backward (braking) passes
        v = np.minimum(self.max_speed, np.sqrt(self.max_lat_acc / np.maximum(kappa, 1e-6)))
        for gate_pos in self._gates_pos[~self._gates_visited]:
            near = np.linalg.norm(pts - gate_pos, axis=1) < UNKNOWN_GATE_RADIUS
            v[near] = np.minimum(v[near], self.unknown_gate_speed)
        for obstacle_pos in self._obstacles_pos[~self._obstacles_visited]:
            near = np.linalg.norm(pts[:, :2] - obstacle_pos[:2], axis=1) < UNKNOWN_OBSTACLE_RADIUS
            v[near] = np.minimum(v[near], self.unknown_obstacle_speed)
        v[0] = min(v[0], float(np.linalg.norm(start_vel)))
        for i in range(n):
            v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * self.max_lon_acc * ds))
        for i in range(n - 1, -1, -1):
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * self.max_lon_acc * ds))
        dt = 2 * ds / np.maximum(v[:-1] + v[1:], 1e-3)
        t = t_start + np.concatenate(([0.0], np.cumsum(dt)))
        # Analytic derivatives: v * tangent and tangential + centripetal acceleration
        tangent = d1 / speed[:, None]
        normal_acc = d2 - np.sum(d2 * tangent, axis=1, keepdims=True) * tangent  # = kappa * n
        a_tan = np.zeros(n + 1)
        a_tan[:-1] = (v[1:] ** 2 - v[:-1] ** 2) / (2 * ds)
        vel = v[:, None] * tangent
        acc = a_tan[:, None] * tangent + (v**2)[:, None] * normal_acc / speed[:, None] ** 2
        if SMOOTH_SIGMA <= 0:
            return SampledTrajectory(t, pts, vel, acc), float(t[-1])
        # Uniform resampling in time, padded with the start state before and a hover at the end
        tu = np.arange(t[0], t[-1] + SMOOTH_DT, SMOOTH_DT)
        pu = np.stack([np.interp(tu, t, pts[:, k]) for k in range(3)], axis=1)
        n_pad = int(4 * SMOOTH_SIGMA / SMOOTH_DT) + 1
        tau = SMOOTH_DT * np.arange(n_pad, 0, -1)[:, None]  # Time before the start
        before = pu[0] - start_vel * tau + 0.5 * start_acc * tau**2
        after = np.repeat(pu[-1:], n_pad, axis=0)
        padded = np.concatenate((before, pu, after))
        smooth = gaussian_filter1d(padded, SMOOTH_SIGMA / SMOOTH_DT, axis=0, mode="nearest")
        vel_s = np.gradient(smooth, SMOOTH_DT, axis=0)
        acc_s = np.gradient(vel_s, SMOOTH_DT, axis=0)
        sl = slice(n_pad, n_pad + len(tu))
        return SampledTrajectory(tu, smooth[sl], vel_s[sl], acc_s[sl]), float(tu[-1])

    # endregion

    def _maybe_replan(self, obs: dict[str, NDArray[np.floating]], t: float):
        """Re-plan if the observed gate or obstacle poses changed."""
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_quat = np.asarray(obs["gates_quat"], dtype=float)
        obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float)
        changed = (
            np.max(np.abs(gates_pos - self._gates_pos)) > REPLAN_THRESHOLD
            or np.max(np.abs(gates_quat - self._gates_quat)) > REPLAN_THRESHOLD
            or np.max(np.abs(obstacles_pos - self._obstacles_pos)) > REPLAN_THRESHOLD
        )
        if not changed:
            return
        self._gates_pos, self._gates_quat = gates_pos.copy(), gates_quat.copy()
        self._obstacles_pos = obstacles_pos.copy()
        self._gates_visited = np.asarray(obs["gates_visited"], dtype=bool).copy()
        self._obstacles_visited = np.asarray(obs["obstacles_visited"], dtype=bool).copy()
        t_ref = min(t, self._t_end)
        self._plan(
            start_pos=self._spline(t_ref),
            start_vel=self._spline(t_ref, 1),
            start_acc=self._spline(t_ref, 2),
            n_gates_passed=int(obs["n_gates_passed"]),
            t_start=t_ref,
        )

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, qx, qy, qz, qw, wx, wy, wz].
        """
        t = self._tick / self._freq
        if int(obs["n_gates_passed"]) >= len(self._gate_sequence):
            self._finished = True
        else:
            self._maybe_replan(obs, t)
        t = min(t + LOOKAHEAD, self._t_end)
        des_pos = self._spline(t)
        des_vel = self._spline(t, 1) if t < self._t_end else np.zeros(3)
        t_acc = min(t + ACC_LOOKAHEAD, self._t_end)
        des_acc = self._spline(t_acc, 2) if t < self._t_end else np.zeros(3)
        # Outer integral loop: compensates steady offsets (e.g. randomized mass) that the onboard
        # controller does not remove quickly enough
        if t > 0.5:  # Skip the takeoff, where the ground contact biases the error
            err = des_pos - np.asarray(obs["pos"], dtype=float)
            self._pos_integral += POS_INTEGRAL_GAIN * err / self._freq
            self._pos_integral = np.clip(self._pos_integral, -POS_INTEGRAL_MAX, POS_INTEGRAL_MAX)
        pos_err = des_pos - np.asarray(obs["pos"], dtype=float)
        vel_err = des_vel - np.asarray(obs["vel"], dtype=float)
        p_boost = np.clip(TRACK_P_BOOST * pos_err, -TRACK_BOOST_MAX, TRACK_BOOST_MAX)
        d_boost = np.clip(TRACK_D_BOOST * vel_err, -TRACK_BOOST_MAX, TRACK_BOOST_MAX)
        cmd_pos = des_pos + self._pos_integral + p_boost
        cmd_pos[2] -= Z_OFFSET_COMPENSATION
        des_vel = des_vel + d_boost
        des_quat = R.from_euler("z", 0.0).as_quat()
        return np.concatenate((cmd_pos, des_vel, des_acc, des_quat, np.zeros(3)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the internal clock.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal clock and the integral correction."""
        self._tick = 0
        self._pos_integral = np.zeros(3)

    def render_callback(self, sim: Sim):
        """Visualize the planned trajectory, the waypoints and the current setpoint."""
        t = min(self._tick / self._freq, self._t_end)
        draw_points(sim, self._spline(t).reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        draw_points(sim, self._waypoints, rgba=(0.0, 0.0, 1.0, 1.0), size=0.015)
        trajectory = self._spline(np.linspace(t, self._t_end, 100))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))
