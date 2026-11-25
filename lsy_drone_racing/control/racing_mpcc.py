"""Model predictive contouring control (MPCC) racing controller.

Instead of tracking a time-parameterized trajectory, the controller follows a purely geometric
racing path and decides the speed itself: the MPC maximizes the progress along the path while the
deviation from it is penalized. This follows A. Romero et al., "Model Predictive Contouring Control
for Time-Optimal Quadrotor Flight", IEEE T-RO 2022.

* Path: cubic spline through approach, center and exit points of every gate plus a few via points
  that keep the path away from the poles. It is parameterized by its arc length theta. When the
  environment reveals the exact pose of a gate or pole, the path is rebuilt and the progress is
  re-projected; there is no schedule that could become invalid.
* Prediction model: the identified attitude dynamics of the simulator with first-order thrust
  dynamics and drag (``so_rpy_rotor_drag``), augmented by the progress theta, its speed v_theta and
  the progress acceleration as an additional input.
* Cost (per stage, nonlinear least squares): contour and lag error with respect to the path
  linearized around the predicted progress, a reward for v_theta (pull towards a high reference
  speed) and small input/attitude-rate penalties. The contour weight is increased around the gate
  centers so that the drone passes the openings precisely and may cut corners elsewhere.
* Poles: soft constraints on the horizontal distance to every pole.

The controller requires ``control_mode = "attitude"`` in the config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, NamedTuple

import casadi as cs
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.so_rpy_rotor_drag import symbolic_dynamics_euler
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class MPCCTuning:
    """Tuning parameters of the controller (controller-internal, the track is not changed).

    The variants of the controller (other files in this directory) subclass :class:`RacingMPCC` and
    replace some of these values. Lengths in [m], speeds in [m/s], angles in [rad].
    """

    # Horizon
    n_horizon: int = 30  # Number of prediction steps
    t_horizon: float = 0.6  # Prediction horizon [s]
    max_tilt: float = 1.0  # Maximum roll and pitch
    # Cost weights
    q_contour: float = 4000.0  # Contour error weight away from the gates
    q_contour_gate: float = 3000.0  # Additional contour error weight at the gate centers
    gate_sigma: float = 0.1  # Arc length scale of the decay of the gate emphasis
    gate_before: float = 0.5  # Full emphasis from this arc length before the gate center ...
    gate_after: float = 0.25  # ... to this arc length after it
    q_lag: float = 300.0  # Lag error weight (keeps theta at the drone position)
    q_progress: float = 6.0  # Weight pulling v_theta towards v_ref (progress reward)
    v_ref: float = 10.0  # Reference progress speed
    v_max: float = 6.0  # Maximum progress speed
    a_theta_max: float = 30.0  # Maximum progress acceleration [m/s^2]
    q_drpy: float = 5.0  # Attitude rate weight
    r_rpy: float = 20.0  # Attitude command weight
    r_thrust: float = 5.0  # Weight of the thrust command deviation from hover
    r_a_theta: float = 0.01  # Progress acceleration weight
    q_rp: float = 0.0  # Roll and pitch weight
    q_yaw: float = 50.0  # Yaw weight (zero yaw)
    r_du: float = 5.0  # Weight of the difference between the attitude command and the attitude
    r_d_rpy: float = 0.01  # Weight of the attitude command rate
    r_d_thrust: float = 0.1  # Weight of the thrust command rate
    d_rpy_max: float = 30.0  # Maximum attitude command rate [rad/s]
    d_thrust_max: float = 15.0  # Maximum thrust command rate [N/s]
    # Speed limits (soft constraints on the squared speed). Gates are revealed only 0.7 m before
    # them: approaching a gate whose exact pose is unknown, the speed is limited so that the drone
    # can still correct the offset after the observation.
    v_unknown: float = 2.2  # Before unknown gates
    v_unknown0: float = 2.2  # Before gate 0 while unknown
    v_unknown3: float = 2.2  # Before gate 3 while unknown (tight approach around its pole)
    slow_before: float = 1.2  # Arc length before the gate center where the limits apply
    v_last: float = 100.0  # Before the last gate of the sequence, known or not (100: off)
    last_before: float = 1.0  # Arc length before the last gate center where v_last applies
    speed_slack_w: float = 1e3  # Penalty of the violation
    # Poles (soft constraints)
    pole_radius: float = 0.2  # Minimum horizontal distance from the pole axes
    pole_slack_w: float = 1e6  # Penalty of the violation
    # Real-time iterations per control step (more right after a path change, when the previous
    # solution is a poor initialization)
    rti_iters: int = 3
    rti_iters_change: int = 4
    # Stuck detection (e.g. a pole revealed on the path): the progress jumps forward
    stuck_speed: float = 0.3
    stuck_time: float = 0.4  # [s]
    stuck_skip: float = 0.25
    stuck_hold: bool = False  # The progress never falls back behind a jump (else it may)
    # Gate tunnel (soft constraint on the offset from the gate center in the gate plane)
    tunnel_radius: float = 0.08
    tunnel_half_len: float = 0.3  # Arc length before the gate center in which it is active
    tunnel_after: float = 0.3  # ... and after it (the path may already turn there)
    tunnel_slack_w: float = 1e5
    slack_lin: float = 0.1  # Linear slack penalty relative to the quadratic one
    # Path: samples are pushed out of the clearance radius of the poles (0: off), smoothly along
    # the arc length and not near the gate centers
    pole_clear: float = 0.25
    pole_smooth: float = 0.1  # Arc length scale of the smoothing of the push
    pole_push_gain: float = 1.5  # Compensation of the smoothing
    pole_gate_free: float = 0.2  # No push within this arc length from the gate centers
    pole_push_iters: int = 2  # Repetitions (the smoothing may leave the path too close)
    gate_axis_d: float = 0.25  # Distance of the exit points from the gate centers
    gate_pre_d: float = 0.15  # Distance of the approach points from the gate centers
    last_pre_d: float = 0.0  # Extra approach point on the axis of the last gate (0: none)
    finish_d: float = 1.0  # Distance of the path end after the last gate
    # Route shape: offsets in the frame of the named gate (x: through the gate); slopes: lateral
    # offset per distance along the gate axis of the approach/exit. Several gates have a pole on
    # their axis.
    start_via: tuple[float, ...] = (-1.0, 0.3, 0.4)  # Start detour (world) that reveals gate 2
    first_via: tuple[float, ...] = (-0.6, -0.4, -0.1)  # Before gate 0, beside its axis pole
    g0_in: float = 0.5  # Approach slope of gate 0
    g0_via: tuple[float, ...] = (0.36, -0.04, 0.1191)  # After gate 0, clear of pole 1
    g1_via: tuple[float, ...] = (-0.8605, 0.2831, -0.13)  # Before gate 1
    g2_out: float = 0.55  # Exit slope of gate 2 (pole on its axis behind it)
    via2: tuple[float, ...] = (0.5, 0.5, 0.17)  # U-turn after gate 2 around the pole behind it
    via2b: tuple[float, ...] = (0.15, 0.85, 0.3)
    via3_x: float = -0.6  # Via point before gate 3, beside the pole on its axis
    via3_z: float = -0.06
    g3_via_out: tuple[float, ...] = (0.749, -0.094, -0.054)  # After gate 3
    # Gate 3: the pole is passed on the -y side; if the revealed pole lies on that side of the gate
    # axis (lateral offset < g3_flip_y), it is passed on the +y side instead. Pairs: (-y, +y) side.
    g3_in: tuple[float, ...] = (0.5, -0.3)  # Approach slope
    via3_y: tuple[float, ...] = (-0.3, 0.2)  # Lateral offset of the via point before the gate
    g3_bias: tuple[float, ...] = (0.18, 0.0)  # Lateral offset of the crossing (tracking error)
    g3_flip_y: float = -0.1
    # Thrust scale estimation (randomized mass and model mismatch)
    thrust_est_tau: float = 0.3  # Time constant [s] (0: disabled)
    thrust_est_limits: tuple[float, ...] = (0.75, 1.35)


class Passage(NamedTuple):
    """Gate passage: lateral slopes of the approach and exit, lateral offset of the crossing."""

    slope_in: float = 0.0
    slope_out: float = 0.0
    bias: float = 0.0


def _route(t: MPCCTuning, flip3: bool) -> list[tuple[int, Passage | tuple[float, float, float]]]:
    """Route as (gate ID, element) pairs; an element is a gate passage or a via point.

    Via points are offsets in the frame of the gate. They come from an offline optimization of the
    racing line on randomly perturbed tracks.
    """
    side = int(flip3)
    return [
        (0, t.first_via),
        (0, Passage(slope_in=t.g0_in)),
        (0, t.g0_via),
        (1, t.g1_via),
        (1, Passage()),
        (2, Passage(slope_out=t.g2_out)),
        (2, t.via2),
        (2, t.via2b),
        (3, (t.via3_x, t.via3_y[side], t.via3_z)),
        (3, Passage(slope_in=t.g3_in[side], bias=t.g3_bias[side])),
        (3, t.g3_via_out),
        (1, Passage()),
    ]


_SOLVERS: dict = {}
NP = 24  # Number of stage parameters


def _push_from_poles(
    dense: NDArray[np.floating],
    s: NDArray[np.floating],
    poles: NDArray,
    gate_s: NDArray,
    t: MPCCTuning,
) -> NDArray[np.floating]:
    """Push path samples horizontally out of the clearance radius of the poles (smoothed)."""
    offset = np.zeros_like(dense)
    for pole in poles[:, :2]:
        delta = dense[:, :2] - pole
        d = np.maximum(np.linalg.norm(delta, axis=1), 1e-6)
        push = np.maximum(t.pole_clear - d, 0.0)
        offset[:, :2] += (delta / d[:, None]) * push[:, None]
    near_gate = np.min(np.abs(s[:, None] - gate_s[None, :]), axis=1) < t.pole_gate_free
    offset[near_gate] = 0.0
    ds = float(np.mean(np.diff(s)))
    offset = gaussian_filter1d(offset, t.pole_smooth / ds, axis=0, mode="nearest")
    # The smoothing shrinks the push: rescale so that the clearance is reached
    return dense + t.pole_push_gain * offset


class Path:
    """Arc length parameterized cubic spline path with the arc lengths of the gate centers."""

    def __init__(
        self,
        points: NDArray[np.floating],
        gate_idx: list[int],
        poles: NDArray | None = None,
        t: MPCCTuning | None = None,
    ):
        """Build the path through the points; gate_idx are the indices of the gate centers.

        Path samples closer than t.pole_clear to a pole are pushed away horizontally (smoothly, and
        not near the gate centers) so that the path never runs into a pole revealed late.
        """
        chord = np.linalg.norm(np.diff(points, axis=0), axis=1)
        u = np.concatenate(([0.0], np.cumsum(chord)))
        spline = CubicSpline(u, points, bc_type="natural")
        uu = np.arange(0.0, u[-1], 0.005)
        dense = spline(uu)
        s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))))
        gate_s = np.interp(u[gate_idx], uu, s[: len(uu)])
        t = t or MPCCTuning()
        for _ in range(int(t.pole_push_iters) if poles is not None and t.pole_clear > 0 else 0):
            dense = _push_from_poles(dense, s, poles, gate_s, t)
            s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))))
            gate_s = np.interp(u[gate_idx], uu, s[: len(uu)])
        self.s = np.arange(0.0, s[-1], 0.01)
        self.pos = np.stack([np.interp(self.s, s, dense[:, k]) for k in range(3)], axis=1)
        tan = np.gradient(self.pos, self.s, axis=0)
        self.tan = tan / np.linalg.norm(tan, axis=1, keepdims=True)
        self.length = float(self.s[-1])
        self.gate_s = gate_s

    def __call__(self, theta: NDArray[np.floating]) -> tuple[NDArray, NDArray]:
        """Position and unit tangent at the arc lengths theta."""
        theta = np.clip(theta, 0.0, self.length)
        p = np.stack([np.interp(theta, self.s, self.pos[:, k]) for k in range(3)], axis=-1)
        t = np.stack([np.interp(theta, self.s, self.tan[:, k]) for k in range(3)], axis=-1)
        return p, t / np.linalg.norm(t, axis=-1, keepdims=True)

    def project(self, pos: NDArray[np.floating], theta: float, back: float, ahead: float) -> float:
        """Arc length of the closest path point to pos in a window around theta."""
        m = (self.s >= theta - back) & (self.s <= theta + ahead)
        if not np.any(m):
            return float(np.clip(theta, 0.0, self.length))
        idx = np.nonzero(m)[0]
        d = np.linalg.norm(self.pos[idx] - pos, axis=1)
        return float(self.s[idx[int(np.argmin(d))]])


def _create_solver(params: dict, t: MPCCTuning) -> AcadosOcpSolver:
    """Create the acados solver (cached per configuration of the build-time parameters)."""
    key = (
        t.n_horizon,
        t.t_horizon,
        t.max_tilt,
        t.v_max,
        t.a_theta_max,
        t.d_rpy_max,
        t.d_thrust_max,
    )
    key += (t.pole_radius, t.pole_slack_w, t.tunnel_slack_w, t.slack_lin, t.speed_slack_w)
    if key in _SOLVERS:
        return _SOLVERS[key]
    common = {
        k: params[k]
        for k in ("mass", "gravity_vec", "acc_coef", "cmd_f_coef", "rpy_coef", "rpy_rates_coef")
    }
    X_dot, X, U, _ = symbolic_dynamics_euler(
        True,
        J=params["J"],
        J_inv=params["J_inv"],
        thrust_dyn_coef=params["thrust_dyn_coef"],
        drag_matrix=params["drag_matrix"],
        cmd_rpy_coef=params["cmd_rpy_coef"],
        **common,
    )
    theta, v_theta, a_theta = cs.MX.sym("theta"), cs.MX.sym("v_theta"), cs.MX.sym("a_theta")
    # The attitude and thrust commands are states; the inputs are their rates (smooth commands)
    cmd = cs.MX.sym("cmd", 4)
    d_cmd = cs.MX.sym("d_cmd", 4)
    X_dot = cs.substitute(X_dot, U, cmd)
    x = cs.vertcat(X, theta, v_theta, cmd)
    u = cs.vertcat(d_cmd, a_theta)
    U = cmd
    p = cs.MX.sym("p", NP)  # path point, tangent, theta_bar, pole xy (8), gate center/side/up
    pd, tg, th_bar = p[0:3], p[3:6], p[6]
    e = X[0:3] - pd - tg * (theta - th_bar)
    e_lag = cs.dot(e, tg)
    e_con = e - e_lag * tg
    hover = (params["mass"] * 9.81 - params["acc_coef"]) / params["cmd_f_coef"]
    y = cs.vertcat(
        e_con,
        e_lag,
        v_theta,
        X[9:12],
        U[0:3],
        U[3] - hover,
        a_theta,
        X[3:6],
        U[0:3] - X[3:6],
        d_cmd,
    )
    y_e = cs.vertcat(e_con, e_lag)
    poles = [X[0:2] - p[7 + 2 * i : 9 + 2 * i] for i in range(4)]
    rel = X[0:3] - p[15:18]
    lat, up = cs.dot(rel, p[18:21]), cs.dot(rel, p[21:24])
    h = cs.vertcat(*[cs.dot(d, d) for d in poles], lat**2 + up**2, cs.dot(X[6:9], X[6:9]))

    model = AcadosModel()
    model.name = "racing_mpcc"
    model.x, model.u, model.p = x, u, p
    model.f_expl_expr = cs.vertcat(X_dot, v_theta, a_theta, d_cmd)
    model.cost_y_expr, model.cost_y_expr_e = y, y_e
    model.con_h_expr = h

    ocp = AcadosOcp()
    ocp.model = model
    nx, nu = x.rows(), u.rows()
    ocp.solver_options.N_horizon = t.n_horizon
    ocp.solver_options.tf = t.t_horizon
    ocp.cost.cost_type, ocp.cost.cost_type_e = "NONLINEAR_LS", "NONLINEAR_LS"
    ocp.cost.W = np.eye(y.rows())
    ocp.cost.W_e = np.eye(y_e.rows())
    ocp.cost.yref, ocp.cost.yref_e = np.zeros(y.rows()), np.zeros(y_e.rows())
    ocp.parameter_values = np.zeros(NP)
    t_min, t_max = 4 * params["thrust_min"], 4 * params["thrust_max"]
    ocp.constraints.idxbx = np.array([3, 4, 17, 18, 19, 20, 21])
    ocp.constraints.lbx = np.array(
        [-t.max_tilt, -t.max_tilt, 0.0, -t.max_tilt, -t.max_tilt, -0.3, t_min]
    )
    ocp.constraints.ubx = np.array(
        [t.max_tilt, t.max_tilt, t.v_max, t.max_tilt, t.max_tilt, 0.3, t_max]
    )
    ocp.constraints.idxbu = np.arange(nu)
    ocp.constraints.lbu = -np.array([*[t.d_rpy_max] * 3, t.d_thrust_max, t.a_theta_max])
    ocp.constraints.ubu = np.array([*[t.d_rpy_max] * 3, t.d_thrust_max, t.a_theta_max])
    ocp.constraints.x0 = np.zeros(nx)
    ocp.constraints.lh = np.array([t.pole_radius**2] * 4 + [0.0, 0.0])
    ocp.constraints.uh = np.full(6, 1e3)
    ocp.constraints.idxsh = np.arange(6)
    ocp.cost.Zl = np.array([t.pole_slack_w] * 4 + [0.0, 0.0])
    ocp.cost.Zu = np.array([0.0] * 4 + [t.tunnel_slack_w, t.speed_slack_w])
    ocp.cost.zl = t.slack_lin * ocp.cost.Zl
    ocp.cost.zu = t.slack_lin * ocp.cost.Zu
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 30
    solver = AcadosOcpSolver(
        ocp, json_file="c_generated_code/racing_mpcc.json", verbose=False, build=True, generate=True
    )
    _SOLVERS[key] = solver
    return solver


class RacingMPCC(Controller):
    """Model predictive contouring control racing controller (attitude interface)."""

    TUNING: ClassVar[MPCCTuning] = MPCCTuning()  # Replaced by the variants of the controller

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Build the path and the solver."""
        super().__init__(obs, info, config)
        self._t = self.TUNING
        self._dt = 1 / config.env.freq
        self._params = load_params(Dynamics.so_rpy_rotor_drag, config.sim.drone)
        self._solver = _create_solver(self._params, self._t)
        self._solver.reset()
        self._nx = 22
        p = self._params
        self._hover = (p["mass"] * 9.81 - p["acc_coef"]) / p["cmd_f_coef"]
        self._cmd = np.array([0.0, 0.0, 0.0, self._hover])
        self._start = np.asarray(obs["pos"], float).copy()
        self._sequence = np.asarray(obs["gate_sequence"], int)
        self._update_track(obs, force=True)
        self._theta = 0.0
        self._theta_pred = None
        self._u_pred = None
        self._fail_count = 0
        self._iters_left = 0
        self._slow_time = 0.0
        self._theta_floor = 0.0
        self._thrust_state = 0.0
        self._thrust_scale = 1.0
        self._last_vel = self._last_quat = None
        self._tick = 0
        self._finished = False
        self._init_guess()

    # region Path
    def _update_track(self, obs: dict, force: bool = False) -> bool:
        gp = np.asarray(obs["gates_pos"], float)
        gq = np.asarray(obs["gates_quat"], float)
        op = np.asarray(obs["obstacles_pos"], float)
        if not force and (
            np.abs(gp - self._gates_pos).max() < 1e-4
            and np.abs(gq - self._gates_quat).max() < 1e-4
            and np.abs(op - self._obstacles_pos).max() < 1e-4
        ):
            return False
        self._gates_pos, self._gates_quat, self._obstacles_pos = gp.copy(), gq.copy(), op.copy()
        points, gate_idx, gate_ids = [self._start, np.array(self._t.start_via)], [], []
        flip3 = self._pole_local(3, 3)[1] < self._t.g3_flip_y
        route = _route(self._t, flip3)
        for k, (gate_id, element) in enumerate(route):
            rot = R.from_quat(gq[gate_id])
            if isinstance(element, Passage):  # Approach point, gate center and exit point
                bias = np.array([0.0, element.bias, 0.0])
                if k == len(route) - 1 and self._t.last_pre_d > 0:  # Straight final approach
                    points.append(gp[gate_id] + rot.apply([-self._t.last_pre_d, 0.0, 0.0]))
                pre = [-self._t.gate_pre_d, -element.slope_in * self._t.gate_pre_d, 0.0]
                post = [self._t.gate_axis_d, element.slope_out * self._t.gate_axis_d, 0.0]
                points.append(gp[gate_id] + rot.apply(np.add(pre, bias)))
                gate_idx.append(len(points))
                gate_ids.append(gate_id)
                points.append(gp[gate_id] + rot.apply(bias))
                points.append(gp[gate_id] + rot.apply(post))
            else:
                points.append(gp[gate_id] + rot.apply(element))
        last = self._sequence[-1]
        points.append(gp[last] + R.from_quat(gq[last]).apply([self._t.finish_d, 0.0, 0.0]))
        self._path = Path(np.array(points), gate_idx, op, self._t)
        self._gate_of_center = np.array(gate_ids)
        return True

    def _pole_local(self, gate_id: int, pole_id: int) -> NDArray[np.floating]:
        """Position of a pole in the frame of a gate (only the horizontal components matter)."""
        rot = R.from_quat(self._gates_quat[gate_id])
        return rot.apply(self._obstacles_pos[pole_id] - self._gates_pos[gate_id], inverse=True)

    def _stage_params(self, thetas: NDArray[np.floating]) -> NDArray[np.floating]:
        pd, tg = self._path(thetas)
        poles = np.tile(self._obstacles_pos[:, :2].reshape(-1), (len(thetas), 1))
        k = np.argmin(np.abs(thetas[:, None] - self._path.gate_s[None, :]), axis=1)
        gate_ids = self._gate_of_center[k]
        rot = R.from_quat(self._gates_quat[gate_ids]).as_matrix()  # (n, 3, 3)
        gates = np.concatenate((self._gates_pos[gate_ids], rot[:, :, 1], rot[:, :, 2]), axis=1)
        rel = thetas - self._path.gate_s[k]
        self._tunnel_active = (rel > -self._t.tunnel_half_len) & (rel < self._t.tunnel_after)
        return np.concatenate((pd, tg, thetas[:, None], poles, gates), axis=1)

    def _slow_stages(self, thetas: NDArray[np.floating], visited: NDArray[np.bool_]) -> NDArray:
        """Squared speed limit per stage.

        The speed is low just before the centers of gates that are unknown, and before the last
        gate (v_last) where the drone would otherwise arrive at its top speed.
        """
        t = self._t
        rel = self._path.gate_s[None, :] - thetas[:, None]  # (stages, gate passages)
        unknown = ~visited[self._gate_of_center]
        slow = (rel > -0.05) & (rel < t.slow_before) & unknown[None, :]
        v_max = np.choose(
            self._gate_of_center, [t.v_unknown0, t.v_unknown, t.v_unknown, t.v_unknown3]
        )
        limit = np.min(np.where(slow, v_max[None, :] ** 2, 1e3), axis=1)
        last = (rel[:, -1] > -0.05) & (rel[:, -1] < t.last_before)
        return np.where(last, np.minimum(limit, t.v_last**2), limit)

    def _contour_weight(self, thetas: NDArray[np.floating]) -> NDArray[np.floating]:
        rel = thetas[:, None] - self._path.gate_s[None, :]
        d = np.maximum(np.maximum(-rel - self._t.gate_before, rel - self._t.gate_after), 0.0).min(
            axis=1
        )
        return self._t.q_contour + self._t.q_contour_gate * np.exp(
            -0.5 * (d / self._t.gate_sigma) ** 2
        )

    # endregion

    def _init_guess(self):
        x = np.zeros(self._nx)
        x[:3] = self._start
        x[18:22] = self._cmd
        for j in range(self._t.n_horizon + 1):
            self._solver.set(j, "x", x)

    def _update_thrust_scale(self, obs: dict):
        vel = np.asarray(obs["vel"], float)
        if (
            self._t.thrust_est_tau > 0
            and self._last_vel is not None
            and float(obs["pos"][2]) > 0.05
        ):
            p = self._params
            acc = (vel - self._last_vel) / self._dt
            b3 = R.from_quat(self._last_quat).apply([0.0, 0.0, 1.0])
            drag = np.diag(np.asarray(p["drag_matrix"], float)) * self._last_vel / p["mass"]
            measured = float((acc - np.asarray(p["gravity_vec"]) - drag) @ b3)
            predicted = (p["acc_coef"] + p["cmd_f_coef"] * self._thrust_state) / p["mass"]
            if predicted > 1.0:
                ratio = np.clip(measured / predicted, *self._t.thrust_est_limits)
                self._thrust_scale += (
                    self._dt / self._t.thrust_est_tau * (ratio - self._thrust_scale)
                )
        self._last_vel = vel
        self._last_quat = np.asarray(obs["quat"], float)

    def _command(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        p = self._params
        alpha = 1.0 - np.exp(-float(p["thrust_dyn_coef"]) * self._dt)
        self._thrust_state += alpha * (float(u[3]) - self._thrust_state)
        cmd = np.array(u[:4], dtype=np.float32)
        force = p["acc_coef"] + p["cmd_f_coef"] * cmd[3]
        cmd[3] = (force / self._thrust_scale - p["acc_coef"]) / p["cmd_f_coef"]
        cmd[3] = np.clip(cmd[3], 4 * p["thrust_min"], 4 * p["thrust_max"])
        return cmd

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the attitude and collective thrust command [roll, pitch, yaw, thrust]."""
        if int(obs["n_gates_passed"]) >= len(self._sequence) or int(obs["n_gates_passed"]) < 0:
            self._finished = True
        pos = np.asarray(obs["pos"], float)
        vel = np.asarray(obs["vel"], float)
        changed = self._update_track(obs)
        if changed:
            self._iters_left = int(self._t.rti_iters_change)
        self._update_thrust_scale(obs)
        # Stuck detection (e.g. a pole revealed on the path): push the progress forward
        t = self._tick * self._dt
        self._slow_time = (
            self._slow_time + self._dt
            if (t > 1.0 and np.linalg.norm(vel) < self._t.stuck_speed)
            else 0.0
        )
        if self._slow_time > self._t.stuck_time:
            self._theta = min(self._theta + self._t.stuck_skip, self._path.length)
            self._slow_time = 0.0
            if self._t.stuck_hold:
                self._theta_floor = self._theta
        theta = self._path.project(pos, self._theta, 0.3, 0.6)
        self._theta = max(theta, self._theta - 0.05, self._theta_floor)
        _, tg0 = self._path(np.array([self._theta]))
        v_theta = float(np.clip(vel @ tg0[0], 0.0, self._t.v_max))
        dt = self._t.t_horizon / self._t.n_horizon
        if self._theta_pred is None:
            thetas = self._theta + v_theta * dt * np.arange(self._t.n_horizon + 1)
        else:
            shifted = np.concatenate(
                (self._theta_pred[1:], [2 * self._theta_pred[-1] - self._theta_pred[-2]])
            )
            thetas = shifted + (self._theta - shifted[0])
        thetas = np.clip(thetas, 0.0, self._path.length)
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate(
            (pos, rpy, vel, drpy, [self._thrust_state] * 4, [self._theta, v_theta], self._cmd)
        )
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)
        prm = self._stage_params(thetas)
        slow = self._slow_stages(thetas, np.asarray(obs["gates_visited"], bool))
        qc = self._contour_weight(thetas)
        yref = np.zeros(23)
        yref[4] = self._t.v_ref
        for j in range(self._t.n_horizon + 1):
            self._solver.set(j, "p", prm[j])
            if 0 < j < self._t.n_horizon:
                uh = self._t.tunnel_radius**2 if self._tunnel_active[j] else 1e3
                vh = slow[j]
                self._solver.constraints_set(j, "uh", np.array([1e3] * 4 + [uh, vh]))
            w_c = [qc[j]] * 3
            if j < self._t.n_horizon:
                w = [
                    *w_c,
                    self._t.q_lag,
                    self._t.q_progress,
                    *[self._t.q_drpy] * 3,
                    *[self._t.r_rpy] * 3,
                    self._t.r_thrust,
                    self._t.r_a_theta,
                ]
                w += [
                    self._t.q_rp,
                    self._t.q_rp,
                    self._t.q_yaw,
                    *[self._t.r_du] * 3,
                    *[self._t.r_d_rpy] * 3,
                    self._t.r_d_thrust,
                ]
                self._solver.cost_set(j, "W", np.diag(w))
                self._solver.set(j, "yref", yref)
            else:
                self._solver.cost_set(j, "W", np.diag([*w_c, self._t.q_lag]))
        n_iter = max(int(self._t.rti_iters), 1)
        if self._iters_left > 0:
            n_iter = max(n_iter, int(self._t.rti_iters_change))
            self._iters_left -= 1
        for _ in range(n_iter):
            status = self._solver.solve()
            if status not in (0, 2):
                break
        if status not in (0, 2):
            # The real-time iteration failed: replay the previous prediction and restart the solver
            self._solver.reset()
            for j in range(self._t.n_horizon + 1):
                self._solver.set(j, "x", x0)
            self._theta_pred = None
            self._fail_count += 1
            if self._u_pred is not None and self._fail_count < len(self._u_pred):
                self._cmd = self._u_pred[self._fail_count].copy()
                return self._command(self._cmd)
            self._cmd = np.array([0.0, 0.0, 0.0, self._hover])
            return self._command(self._cmd)
        self._fail_count = 0
        self._u_pred = np.array(
            [self._solver.get(j, "x")[18:22] for j in range(1, self._t.n_horizon + 1)]
        )
        self._theta_pred = np.array(
            [self._solver.get(j, "x")[16] for j in range(self._t.n_horizon + 1)]
        )
        self._cmd = self._solver.get(1, "x")[18:22].copy()
        return self._command(self._cmd)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the time."""
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the episode state."""
        self._tick = 0
