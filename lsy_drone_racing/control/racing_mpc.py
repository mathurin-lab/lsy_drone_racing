"""Racing controller that tracks the planned racing trajectory with a nonlinear MPC.

The trajectory planning (gate waypoints, collision avoidance, re-timing and online re-planning) is
inherited from :class:`~lsy_drone_racing.control.racing_controller.RacingController`. Instead of
sending the reference to the onboard state controller, whose low position gains make the drone lag
behind fast trajectories, this controller computes attitude and collective thrust commands with a
nonlinear MPC (acados). The MPC uses the identified drone dynamics of the simulator and predicts
0.5 s ahead, so it starts tilting before a turn instead of reacting to the tracking error.

The reference for the MPC contains the planned position and velocity, as well as the attitude and
thrust that realize the planned acceleration (differential flatness with zero yaw). An outer
integral on the position reference compensates model errors such as the randomized drone mass.

The controller requires ``control_mode = "attitude"`` in the config.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from acados_template.builders import ocp_get_default_cmake_builder
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.so_rpy import symbolic_dynamics_euler
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.racing_controller import RacingController

if TYPE_CHECKING:
    from numpy.typing import NDArray

# MPC parameters
N_HORIZON = 25  # Number of prediction steps (one control period each)
MAX_TILT = 0.8  # Maximum roll and pitch [rad]
Q_POS = 200.0  # Weight of the horizontal position error
Q_POS_Z = 400.0  # Weight of the vertical position error
Q_VEL = 10.0  # Weight of the velocity error
Q_RPY = 1.0  # Weight of the attitude error
Q_DRPY = 1.0  # Weight of the attitude rate error
R_RPY = 5.0  # Weight of the attitude command error
R_THRUST = 50.0  # Weight of the thrust command error
# Limits of the velocity profile (replace the ones of the planner)
MPC_MAX_SPEED = 4.0  # [m/s]
MPC_MAX_LAT_ACC = 5.0  # [m/s^2]
MPC_MAX_LON_ACC = 4.0  # [m/s^2]
MPC_UNKNOWN_GATE_SPEED = 1.2  # Speed near gates whose exact pose is not known yet [m/s]
# Outer integral on the reference position (model errors, randomized mass)
MPC_INTEGRAL_GAIN = 4.0  # [1/s]
MPC_INTEGRAL_MAX = 0.3  # [m]
MPC_INTEGRAL_XY = 0.0  # Scale of the integral gain for the horizontal axes (z is mostly affected)

_SOLVERS: dict = {}  # Compiled solvers, reused across episodes (code generation takes seconds)


def _create_solver(dt: float, params: dict, weights: tuple) -> tuple[AcadosOcpSolver, int, int]:
    """Create (or reuse) the acados OCP solver for trajectory tracking."""
    key = (dt, N_HORIZON, MAX_TILT, weights)
    if key in _SOLVERS:
        return _SOLVERS[key]
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=params["mass"],
        gravity_vec=params["gravity_vec"],
        acc_coef=params["acc_coef"],
        cmd_f_coef=params["cmd_f_coef"],
        rpy_coef=params["rpy_coef"],
        rpy_rates_coef=params["rpy_rates_coef"],
        cmd_rpy_coef=params["cmd_rpy_coef"],
    )
    model = AcadosModel()
    model.name = "racing_mpc"
    model.f_expl_expr = X_dot
    model.x = X
    model.u = U

    ocp = AcadosOcp()
    ocp.model = model
    nx, nu = X.rows(), U.rows()
    ny = nx + nu
    ocp.solver_options.N_horizon = N_HORIZON
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    q_pos, q_pos_z, q_vel, q_rpy, q_drpy, r_rpy, r_thrust = weights
    Q = np.diag([q_pos, q_pos, q_pos_z, *[q_rpy] * 3, *[q_vel] * 3, *[q_drpy] * 3])
    W_u = np.diag([r_rpy, r_rpy, r_rpy, r_thrust])
    ocp.cost.W = scipy.linalg.block_diag(Q, W_u)
    ocp.cost.W_e = Q.copy()
    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[nx:, :] = np.eye(nu)
    ocp.cost.Vx, ocp.cost.Vu, ocp.cost.Vx_e = Vx, Vu, np.eye(nx)
    ocp.cost.yref, ocp.cost.yref_e = np.zeros(ny), np.zeros(nx)

    ocp.constraints.lbx = np.array([-MAX_TILT, -MAX_TILT, -0.5])
    ocp.constraints.ubx = np.array([MAX_TILT, MAX_TILT, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])
    thrust_min, thrust_max = params["thrust_min"] * 4, params["thrust_max"] * 4
    ocp.constraints.lbu = np.array([-MAX_TILT, -MAX_TILT, -0.5, thrust_min])
    ocp.constraints.ubu = np.array([MAX_TILT, MAX_TILT, 0.5, thrust_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.zeros(nx)

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.qp_solver_cond_N = N_HORIZON
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.tf = N_HORIZON * dt

    cmake_builder = None
    if os.name == "nt":  # On Windows, build the generated solver with CMake and MinGW
        cmake_builder = ocp_get_default_cmake_builder()
        cmake_builder.generator = "MinGW Makefiles"
        cmake_builder.build_dir = "build-mingw"
    solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/racing_mpc.json",
        verbose=False,
        build=True,
        generate=True,
        cmake_builder=cmake_builder,
    )
    _SOLVERS[key] = (solver, nx, nu)
    return _SOLVERS[key]


class RacingMPC(RacingController):
    """Racing controller with trajectory tracking by a nonlinear MPC (attitude interface)."""

    # The MPC tracks much better than the onboard state controller, so the trajectory can be faster
    max_speed = MPC_MAX_SPEED
    max_lat_acc = MPC_MAX_LAT_ACC
    max_lon_acc = MPC_MAX_LON_ACC
    unknown_gate_speed = MPC_UNKNOWN_GATE_SPEED

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the planner and the MPC.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        self._dt = 1 / config.env.freq
        self._params = load_params(Dynamics.so_rpy, config.sim.drone)
        self._gravity = np.asarray(self._params["gravity_vec"], dtype=float)
        weights = (Q_POS, Q_POS_Z, Q_VEL, Q_RPY, Q_DRPY, R_RPY, R_THRUST)
        self._solver, self._nx, self._nu = _create_solver(self._dt, self._params, weights)
        self._solver.reset()

    def _reference(self, t: NDArray[np.floating]) -> NDArray[np.floating]:
        """MPC references (state and input) at the times t, shape (len(t), nx + nu)."""
        t = np.minimum(t, self._t_end)
        moving = (t < self._t_end)[:, None]
        pos = self._spline(t)
        vel = np.where(moving, self._spline(t, 1), 0.0)
        acc = np.where(moving, self._spline(t, 2), 0.0)
        # Attitude and thrust that realize the planned acceleration (zero yaw)
        force = self._params["mass"] * (acc - self._gravity)
        thrust = np.linalg.norm(force, axis=1)
        z_axis = force / thrust[:, None]
        roll = np.arcsin(np.clip(-z_axis[:, 1], -1.0, 1.0))
        pitch = np.arctan2(z_axis[:, 0], z_axis[:, 2])
        rpy = np.clip(np.stack((roll, pitch, np.zeros_like(roll)), axis=1), -MAX_TILT, MAX_TILT)
        ref = np.zeros((len(t), self._nx + self._nu))
        ref[:, 0:3] = pos + self._pos_integral
        ref[:, 3:6] = rpy
        ref[:, 6:9] = vel
        ref[:, self._nx : self._nx + 3] = rpy
        ref[:, self._nx + 3] = thrust
        return ref

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the attitude and collective thrust command.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            The command [roll, pitch, yaw, collective thrust].
        """
        t = self._tick / self._freq
        if int(obs["n_gates_passed"]) >= len(self._gate_sequence):
            self._finished = True
        else:
            self._maybe_replan(obs, t)
        pos = np.asarray(obs["pos"], dtype=float)
        if t > 0.5:  # Skip the takeoff, where the ground contact biases the error
            err = self._spline(min(t, self._t_end)) - pos
            gain = MPC_INTEGRAL_GAIN * np.array([MPC_INTEGRAL_XY, MPC_INTEGRAL_XY, 1.0])
            self._pos_integral += gain * err * self._dt
            self._pos_integral = np.clip(self._pos_integral, -MPC_INTEGRAL_MAX, MPC_INTEGRAL_MAX)

        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((pos, rpy, obs["vel"], drpy))
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)
        ref = self._reference(t + self._dt * np.arange(N_HORIZON + 1))
        for j in range(N_HORIZON):
            self._solver.set(j, "yref", ref[j])
        self._solver.set(N_HORIZON, "yref", ref[N_HORIZON, : self._nx])
        if self._solver.solve() != 0:
            # The real-time iteration failed (e.g. after a large jump of the reference). Restart
            # the solver from scratch and use the feed-forward command for this step.
            self._solver.reset()
            return ref[0, self._nx :].astype(np.float32)
        return np.asarray(self._solver.get(0, "u"), dtype=np.float32)
