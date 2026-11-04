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

from typing import TYPE_CHECKING, NamedTuple

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

# Horizon
N_HORIZON = 30  # Number of prediction steps
T_HORIZON = 0.6  # Prediction horizon [s]
MAX_TILT = 1.0  # Maximum roll and pitch [rad]
# Cost weights
Q_CONTOUR = 4000.0  # Contour error weight away from the gates
Q_CONTOUR_GATE = 3000.0  # Additional contour error weight at the gate centers
GATE_SIGMA = 0.1  # Arc length scale of the decay of the gate emphasis [m]
GATE_BEFORE = 0.5  # Full emphasis from this arc length before the gate center [m]
GATE_AFTER = 0.25  # ... to this arc length after it [m]
Q_LAG = 300.0  # Lag error weight (keeps theta at the drone position)
Q_PROGRESS = 6  # Weight pulling v_theta towards V_REF (progress reward)
V_REF = 10.0  # Reference progress speed [m/s]
V_MAX = 6.0  # Maximum progress speed [m/s]
A_THETA_MAX = 30.0  # Maximum progress acceleration [m/s^2]
# Gates are revealed only 0.7 m before them: approaching a gate whose exact pose is unknown, the
# speed is limited so that the drone can still correct the offset after the observation.
V_UNKNOWN = 2.2  # Speed limit before unknown gates [m/s]
V_UNKNOWN0 = 2.2  # Speed limit before gate 0 while unknown [m/s]
V_UNKNOWN3 = 2.2  # Speed limit before gate 3 while unknown (tight approach around its pole) [m/s]
SLOW_BEFORE = 1.2  # Arc length before the gate center where the limit applies [m]
SPEED_SLACK_W = 1e3  # Penalty of the (squared) speed limit violation
Q_DRPY = 5.0  # Attitude rate weight
R_RPY = 20.0  # Attitude command weight
R_THRUST = 5.0  # Weight of the thrust command deviation from hover
R_A_THETA = 0.01  # Progress acceleration weight
Q_RP = 0.0  # Roll and pitch weight
Q_YAW = 50.0  # Yaw weight (zero yaw)
R_DU = 5.0  # Weight of the difference between the attitude command and the attitude
R_D_RPY = 0.01  # Weight of the attitude command rate
R_D_THRUST = 0.1  # Weight of the thrust command rate
D_RPY_MAX = 30.0  # Maximum attitude command rate [rad/s]
D_THRUST_MAX = 15.0  # Maximum thrust command rate [N/s]
# Poles (soft constraints)
POLE_RADIUS = 0.2  # Minimum horizontal distance from the pole axes [m]
POLE_SLACK_W = 1e6  # Penalty of the constraint violation
# Real-time iterations per control step (more right after a path change, when the previous
# solution is a poor initialization)
RTI_ITERS = 3
RTI_ITERS_CHANGE = 4  # Iterations during the steps after a path change
# Stuck detection
STUCK_SPEED = 0.3  # [m/s]
STUCK_TIME = 0.4  # [s]
STUCK_SKIP = 0.25  # Progress jump [m]
# Gate tunnel (soft constraint on the offset from the gate center in the gate plane)
TUNNEL_RADIUS = 0.08  # [m]
TUNNEL_HALF_LEN = 0.3  # Arc length around the gate center in which it is active [m]
TUNNEL_SLACK_W = 1e5
SLACK_LIN = 0.1  # Linear slack penalty relative to the quadratic one
# Path
POLE_CLEAR = 0.25  # The path is pushed out of this horizontal distance from the poles [m] (0: off)
POLE_SMOOTH = 0.1  # Arc length scale of the smoothing of the push [m]
POLE_PUSH_GAIN = 1.5  # Compensation of the smoothing
POLE_GATE_FREE = 0.2  # No push within this arc length from the gate centers [m]
POLE_PUSH_ITERS = 2  # Repetitions of the push (the smoothing may leave the path too close)
GATE_AXIS_D = 0.25  # Distance of the exit points from the gate centers [m]
GATE_PRE_D = 0.15  # Distance of the approach points from the gate centers [m]
FINISH_D = 1.0  # Distance of the path end after the last gate [m]
# Route shape (offsets in the frame of the named gate, x: through the gate; slopes: lateral offset
# per distance along the gate axis of the approach/exit). Several gates have a pole on their axis.
START_VIA = (-1.0, 0.3, 0.4)  # Start detour (world frame) within the sensor range of gate 2 [m]
FIRST_VIA = (-0.6, -0.4, -0.1)  # Via point before gate 0, beside the pole on its axis [m]
G0_IN = 0.5  # Approach slope of gate 0
G0_VIA = (0.36, -0.04, 0.1191)  # Via point after gate 0 (clear of pole 1) [m]
G1_VIA = (-0.8605, 0.2831, -0.13)  # Via point before gate 1 [m]
G2_OUT = 0.55  # Exit slope of gate 2 (pole on its axis behind it)
VIA2 = (0.5, 0.5, 0.17)  # U-turn after gate 2 around the pole behind it [m]
VIA2B = (0.15, 0.85, 0.3)  # [m]
VIA3_X = -0.6  # Via point before gate 3, beside the pole on its axis [m]
VIA3_Z = -0.06  # [m]
G3_VIA_OUT = (0.749, -0.094, -0.054)  # Via point after gate 3 [m]
# Gate 3: the pole is passed on the -y side; if the revealed pole lies on that side of the gate axis
# (lateral offset < G3_FLIP_Y), it is passed on the +y side instead
G3_IN = (0.5, -0.3)  # Approach slope of gate 3 for a pass on the -y / +y side
VIA3_Y = (-0.3, 0.2)  # Lateral offset of the via point before gate 3 [m]
G3_BIAS = (0.18, 0.0)  # Lateral offset of the gate 3 crossing (compensates a tracking error) [m]
G3_FLIP_Y = -0.1  # [m]
# Thrust scale estimation (randomized mass and model mismatch)
THRUST_EST_TAU = 0.3  # Time constant [s] (0: disabled)
THRUST_EST_LIMITS = (0.75, 1.35)


class Passage(NamedTuple):
    """Gate passage: lateral slopes of the approach and exit, lateral offset of the crossing."""

    slope_in: float = 0.0
    slope_out: float = 0.0
    bias: float = 0.0


def _route(flip3: bool) -> list[tuple[int, Passage | tuple[float, float, float]]]:
    """Route as (gate ID, element) pairs; an element is a gate passage or a via point.

    Via points are offsets in the frame of the gate. They come from an offline optimization of the
    racing line on randomly perturbed tracks. The route is built on demand so that the module
    parameters can be changed (parameter studies).
    """
    side = int(flip3)
    return [
        (0, FIRST_VIA),
        (0, Passage(slope_in=G0_IN)),
        (0, G0_VIA),
        (1, G1_VIA),
        (1, Passage()),
        (2, Passage(slope_out=G2_OUT)),
        (2, VIA2),
        (2, VIA2B),
        (3, (VIA3_X, VIA3_Y[side], VIA3_Z)),
        (3, Passage(slope_in=G3_IN[side], bias=G3_BIAS[side])),
        (3, G3_VIA_OUT),
        (1, Passage()),
    ]


_SOLVERS: dict = {}
NP = 24  # Number of stage parameters


def _push_from_poles(
    dense: NDArray[np.floating], s: NDArray[np.floating], poles: NDArray, gate_s: NDArray
) -> NDArray[np.floating]:
    """Push path samples horizontally out of the clearance radius of the poles (smoothed)."""
    offset = np.zeros_like(dense)
    for pole in poles[:, :2]:
        delta = dense[:, :2] - pole
        d = np.maximum(np.linalg.norm(delta, axis=1), 1e-6)
        push = np.maximum(POLE_CLEAR - d, 0.0)
        offset[:, :2] += (delta / d[:, None]) * push[:, None]
    near_gate = np.min(np.abs(s[:, None] - gate_s[None, :]), axis=1) < POLE_GATE_FREE
    offset[near_gate] = 0.0
    ds = float(np.mean(np.diff(s)))
    offset = gaussian_filter1d(offset, POLE_SMOOTH / ds, axis=0, mode="nearest")
    # The smoothing shrinks the push: rescale so that the clearance is reached
    return dense + POLE_PUSH_GAIN * offset


class Path:
    """Arc length parameterized cubic spline path with the arc lengths of the gate centers."""

    def __init__(
        self, points: NDArray[np.floating], gate_idx: list[int], poles: NDArray | None = None
    ):
        """Build the path through the points; gate_idx are the indices of the gate centers.

        Path samples closer than POLE_CLEAR to a pole are pushed away horizontally (smoothly, and
        not near the gate centers) so that the path never runs into a pole revealed late.
        """
        chord = np.linalg.norm(np.diff(points, axis=0), axis=1)
        u = np.concatenate(([0.0], np.cumsum(chord)))
        spline = CubicSpline(u, points, bc_type="natural")
        uu = np.arange(0.0, u[-1], 0.005)
        dense = spline(uu)
        s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))))
        gate_s = np.interp(u[gate_idx], uu, s[: len(uu)])
        for _ in range(int(POLE_PUSH_ITERS) if poles is not None and POLE_CLEAR > 0 else 0):
            dense = _push_from_poles(dense, s, poles, gate_s)
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


def _create_solver(params: dict) -> AcadosOcpSolver:
    """Create the acados solver (cached per configuration of the build-time parameters)."""
    key = (N_HORIZON, T_HORIZON, MAX_TILT, V_MAX, A_THETA_MAX, D_RPY_MAX, D_THRUST_MAX)
    key += (POLE_RADIUS, POLE_SLACK_W, TUNNEL_SLACK_W, SLACK_LIN, SPEED_SLACK_W)
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
    ocp.solver_options.N_horizon = N_HORIZON
    ocp.solver_options.tf = T_HORIZON
    ocp.cost.cost_type, ocp.cost.cost_type_e = "NONLINEAR_LS", "NONLINEAR_LS"
    ocp.cost.W = np.eye(y.rows())
    ocp.cost.W_e = np.eye(y_e.rows())
    ocp.cost.yref, ocp.cost.yref_e = np.zeros(y.rows()), np.zeros(y_e.rows())
    ocp.parameter_values = np.zeros(NP)
    t_min, t_max = 4 * params["thrust_min"], 4 * params["thrust_max"]
    ocp.constraints.idxbx = np.array([3, 4, 17, 18, 19, 20, 21])
    ocp.constraints.lbx = np.array([-MAX_TILT, -MAX_TILT, 0.0, -MAX_TILT, -MAX_TILT, -0.3, t_min])
    ocp.constraints.ubx = np.array([MAX_TILT, MAX_TILT, V_MAX, MAX_TILT, MAX_TILT, 0.3, t_max])
    ocp.constraints.idxbu = np.arange(nu)
    ocp.constraints.lbu = -np.array([*[D_RPY_MAX] * 3, D_THRUST_MAX, A_THETA_MAX])
    ocp.constraints.ubu = np.array([*[D_RPY_MAX] * 3, D_THRUST_MAX, A_THETA_MAX])
    ocp.constraints.x0 = np.zeros(nx)
    ocp.constraints.lh = np.array([POLE_RADIUS**2] * 4 + [0.0, 0.0])
    ocp.constraints.uh = np.full(6, 1e3)
    ocp.constraints.idxsh = np.arange(6)
    ocp.cost.Zl = np.array([POLE_SLACK_W] * 4 + [0.0, 0.0])
    ocp.cost.Zu = np.array([0.0] * 4 + [TUNNEL_SLACK_W, SPEED_SLACK_W])
    ocp.cost.zl = SLACK_LIN * ocp.cost.Zl
    ocp.cost.zu = SLACK_LIN * ocp.cost.Zu
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

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Build the path and the solver."""
        super().__init__(obs, info, config)
        self._dt = 1 / config.env.freq
        self._params = load_params(Dynamics.so_rpy_rotor_drag, config.sim.drone)
        self._solver = _create_solver(self._params)
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
        points, gate_idx, gate_ids = [self._start, np.array(START_VIA)], [], []
        flip3 = self._pole_local(3, 3)[1] < G3_FLIP_Y
        for gate_id, element in _route(flip3):
            rot = R.from_quat(gq[gate_id])
            if isinstance(element, Passage):  # Approach point, gate center and exit point
                bias = np.array([0.0, element.bias, 0.0])
                pre = [-GATE_PRE_D, -element.slope_in * GATE_PRE_D, 0.0]
                post = [GATE_AXIS_D, element.slope_out * GATE_AXIS_D, 0.0]
                points.append(gp[gate_id] + rot.apply(np.add(pre, bias)))
                gate_idx.append(len(points))
                gate_ids.append(gate_id)
                points.append(gp[gate_id] + rot.apply(bias))
                points.append(gp[gate_id] + rot.apply(post))
            else:
                points.append(gp[gate_id] + rot.apply(element))
        last = self._sequence[-1]
        points.append(gp[last] + R.from_quat(gq[last]).apply([FINISH_D, 0.0, 0.0]))
        self._path = Path(np.array(points), gate_idx, op)
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
        self._tunnel_active = np.abs(thetas - self._path.gate_s[k]) < TUNNEL_HALF_LEN
        return np.concatenate((pd, tg, thetas[:, None], poles, gates), axis=1)

    def _slow_stages(self, thetas: NDArray[np.floating], visited: NDArray[np.bool_]) -> NDArray:
        """Squared speed limit per stage: low just before the centers of gates that are unknown."""
        rel = self._path.gate_s[None, :] - thetas[:, None]  # (stages, gate passages)
        unknown = ~visited[self._gate_of_center]
        before = (rel > -0.05) & (rel < SLOW_BEFORE)
        v_max = np.choose(self._gate_of_center, [V_UNKNOWN0, V_UNKNOWN, V_UNKNOWN, V_UNKNOWN3])
        slow = before & unknown[None, :]
        return np.min(np.where(slow, v_max[None, :] ** 2, 1e3), axis=1)

    def _contour_weight(self, thetas: NDArray[np.floating]) -> NDArray[np.floating]:
        rel = thetas[:, None] - self._path.gate_s[None, :]
        d = np.maximum(np.maximum(-rel - GATE_BEFORE, rel - GATE_AFTER), 0.0).min(axis=1)
        return Q_CONTOUR + Q_CONTOUR_GATE * np.exp(-0.5 * (d / GATE_SIGMA) ** 2)

    # endregion

    def _init_guess(self):
        x = np.zeros(self._nx)
        x[:3] = self._start
        x[18:22] = self._cmd
        for j in range(N_HORIZON + 1):
            self._solver.set(j, "x", x)

    def _update_thrust_scale(self, obs: dict):
        vel = np.asarray(obs["vel"], float)
        if THRUST_EST_TAU > 0 and self._last_vel is not None and float(obs["pos"][2]) > 0.05:
            p = self._params
            acc = (vel - self._last_vel) / self._dt
            b3 = R.from_quat(self._last_quat).apply([0.0, 0.0, 1.0])
            drag = np.diag(np.asarray(p["drag_matrix"], float)) * self._last_vel / p["mass"]
            measured = float((acc - np.asarray(p["gravity_vec"]) - drag) @ b3)
            predicted = (p["acc_coef"] + p["cmd_f_coef"] * self._thrust_state) / p["mass"]
            if predicted > 1.0:
                ratio = np.clip(measured / predicted, *THRUST_EST_LIMITS)
                self._thrust_scale += self._dt / THRUST_EST_TAU * (ratio - self._thrust_scale)
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
            self._iters_left = int(RTI_ITERS_CHANGE)
        self._update_thrust_scale(obs)
        # Stuck detection (e.g. a pole revealed on the path): push the progress forward
        t = self._tick * self._dt
        self._slow_time = (
            self._slow_time + self._dt if (t > 1.0 and np.linalg.norm(vel) < STUCK_SPEED) else 0.0
        )
        if self._slow_time > STUCK_TIME:
            self._theta = min(self._theta + STUCK_SKIP, self._path.length)
            self._slow_time = 0.0
        theta = self._path.project(pos, self._theta, 0.3, 0.6)
        self._theta = max(theta, self._theta - 0.05)
        _, tg0 = self._path(np.array([self._theta]))
        v_theta = float(np.clip(vel @ tg0[0], 0.0, V_MAX))
        dt = T_HORIZON / N_HORIZON
        if self._theta_pred is None:
            thetas = self._theta + v_theta * dt * np.arange(N_HORIZON + 1)
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
        yref[4] = V_REF
        for j in range(N_HORIZON + 1):
            self._solver.set(j, "p", prm[j])
            if 0 < j < N_HORIZON:
                uh = TUNNEL_RADIUS**2 if self._tunnel_active[j] else 1e3
                vh = slow[j]
                self._solver.constraints_set(j, "uh", np.array([1e3] * 4 + [uh, vh]))
            w_c = [qc[j]] * 3
            if j < N_HORIZON:
                w = [*w_c, Q_LAG, Q_PROGRESS, *[Q_DRPY] * 3, *[R_RPY] * 3, R_THRUST, R_A_THETA]
                w += [Q_RP, Q_RP, Q_YAW, *[R_DU] * 3, *[R_D_RPY] * 3, R_D_THRUST]
                self._solver.cost_set(j, "W", np.diag(w))
                self._solver.set(j, "yref", yref)
            else:
                self._solver.cost_set(j, "W", np.diag([*w_c, Q_LAG]))
        n_iter = max(int(RTI_ITERS), 1)
        if self._iters_left > 0:
            n_iter = max(n_iter, int(RTI_ITERS_CHANGE))
            self._iters_left -= 1
        for _ in range(n_iter):
            status = self._solver.solve()
            if status not in (0, 2):
                break
        if status not in (0, 2):
            # The real-time iteration failed: replay the previous prediction and restart the solver
            self._solver.reset()
            for j in range(N_HORIZON + 1):
                self._solver.set(j, "x", x0)
            self._theta_pred = None
            self._fail_count += 1
            if self._u_pred is not None and self._fail_count < len(self._u_pred):
                self._cmd = self._u_pred[self._fail_count].copy()
                return self._command(self._cmd)
            self._cmd = np.array([0.0, 0.0, 0.0, self._hover])
            return self._command(self._cmd)
        self._fail_count = 0
        self._u_pred = np.array([self._solver.get(j, "x")[18:22] for j in range(1, N_HORIZON + 1)])
        self._theta_pred = np.array([self._solver.get(j, "x")[16] for j in range(N_HORIZON + 1)])
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
