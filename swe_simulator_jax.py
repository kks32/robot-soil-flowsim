from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp


class State(NamedTuple):
    h: jnp.ndarray
    hu: jnp.ndarray
    hv: jnp.ndarray


@dataclass(frozen=True)
class SimParams:
    nx: int
    ny: int
    dx: float
    dt: float
    g: float = 9.81
    rho_bulk: float = 1500.0
    rho_grain: float = 2650.0
    grain_d: float = 0.002
    mu_s: float = 0.38
    mu_2: float = 0.64
    I0: float = 0.279
    h_min: float = 1e-4
    v_min: float = 1e-4
    tool_width: float = 0.12
    tool_length: float = 0.06
    tool_drag: float = 2.0e4
    tool_splash: float = 1.0
    tool_softness: float = 0.01
    tool_speed_eps: float = 1e-3
    slope_eps: float = 0.02
    speed_cap: float = 5.0


def make_grid(params: SimParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.arange(params.nx) * params.dx
    y = jnp.arange(params.ny) * params.dx
    return jnp.meshgrid(x, y)


def init_flat(params: SimParams, height: float) -> State:
    h = jnp.full((params.ny, params.nx), height)
    hu = jnp.zeros_like(h)
    hv = jnp.zeros_like(h)
    return State(h=h, hu=hu, hv=hv)


def _grad_x(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    f_pad = jnp.pad(f, ((0, 0), (1, 1)), mode="edge")
    return (f_pad[:, 2:] - f_pad[:, :-2]) / (2.0 * dx)


def _grad_y(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    f_pad = jnp.pad(f, ((1, 1), (0, 0)), mode="edge")
    return (f_pad[2:, :] - f_pad[:-2, :]) / (2.0 * dx)


def _flux_x(q: jnp.ndarray, g: float, h_min: float) -> jnp.ndarray:
    h = q[0]
    hu = q[1]
    hv = q[2]
    h_safe = jnp.maximum(h, h_min)
    u = hu / h_safe
    v = hv / h_safe
    return jnp.stack(
        [hu, hu * u + 0.5 * g * h * h, hu * v],
        axis=0,
    )


def _flux_y(q: jnp.ndarray, g: float, h_min: float) -> jnp.ndarray:
    h = q[0]
    hu = q[1]
    hv = q[2]
    h_safe = jnp.maximum(h, h_min)
    u = hu / h_safe
    v = hv / h_safe
    return jnp.stack(
        [hv, hv * u, hv * v + 0.5 * g * h * h],
        axis=0,
    )


def _rusanov_flux_x(q: jnp.ndarray, params: SimParams) -> jnp.ndarray:
    q_pad = jnp.pad(q, ((0, 0), (0, 0), (1, 1)), mode="edge")
    flux_pad = jnp.pad(_flux_x(q, params.g, params.h_min), ((0, 0), (0, 0), (1, 1)), mode="edge")
    q_l = q_pad[:, :, :-1]
    q_r = q_pad[:, :, 1:]
    f_l = flux_pad[:, :, :-1]
    f_r = flux_pad[:, :, 1:]

    h_l = jnp.maximum(q_l[0], params.h_min)
    h_r = jnp.maximum(q_r[0], params.h_min)
    u_l = q_l[1] / h_l
    u_r = q_r[1] / h_r
    c_l = jnp.sqrt(params.g * h_l)
    c_r = jnp.sqrt(params.g * h_r)
    a = jnp.maximum(jnp.abs(u_l) + c_l, jnp.abs(u_r) + c_r)
    return 0.5 * (f_l + f_r) - 0.5 * a * (q_r - q_l)


def _rusanov_flux_y(q: jnp.ndarray, params: SimParams) -> jnp.ndarray:
    q_pad = jnp.pad(q, ((0, 0), (1, 1), (0, 0)), mode="edge")
    flux_pad = jnp.pad(_flux_y(q, params.g, params.h_min), ((0, 0), (1, 1), (0, 0)), mode="edge")
    q_l = q_pad[:, :-1, :]
    q_r = q_pad[:, 1:, :]
    f_l = flux_pad[:, :-1, :]
    f_r = flux_pad[:, 1:, :]

    h_l = jnp.maximum(q_l[0], params.h_min)
    h_r = jnp.maximum(q_r[0], params.h_min)
    v_l = q_l[2] / h_l
    v_r = q_r[2] / h_r
    c_l = jnp.sqrt(params.g * h_l)
    c_r = jnp.sqrt(params.g * h_r)
    a = jnp.maximum(jnp.abs(v_l) + c_l, jnp.abs(v_r) + c_r)
    return 0.5 * (f_l + f_r) - 0.5 * a * (q_r - q_l)


def _flux_divergence(q: jnp.ndarray, params: SimParams) -> jnp.ndarray:
    fx = _rusanov_flux_x(q, params)
    fy = _rusanov_flux_y(q, params)
    div_x = (fx[:, :, 1:] - fx[:, :, :-1]) / params.dx
    div_y = (fy[:, 1:, :] - fy[:, :-1, :]) / params.dx
    return div_x + div_y


def _mu_I(u: jnp.ndarray, v: jnp.ndarray, h: jnp.ndarray, params: SimParams) -> jnp.ndarray:
    du_dx = _grad_x(u, params.dx)
    du_dy = _grad_y(u, params.dx)
    dv_dx = _grad_x(v, params.dx)
    dv_dy = _grad_y(v, params.dx)
    strain = jnp.sqrt(du_dx**2 + dv_dy**2 + 0.5 * (du_dy + dv_dx) ** 2 + 1e-8)
    pressure = params.rho_bulk * params.g * h
    denom = jnp.sqrt(pressure / params.rho_grain + 1e-8)
    inertial = params.grain_d * strain / denom
    inertial = jnp.clip(inertial, 1e-6, 1e3)
    return params.mu_s + (params.mu_2 - params.mu_s) / (params.I0 / (inertial + 1e-8) + 1.0)


def _friction_accel(u: jnp.ndarray, v: jnp.ndarray, mu: jnp.ndarray, params: SimParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
    speed = jnp.sqrt(u**2 + v**2 + 1e-8)
    fric_mag = params.g * mu
    ax = -fric_mag * u / speed
    ay = -fric_mag * v / speed
    stop = fric_mag * params.dt > speed
    ax = jnp.where(stop, -u / params.dt, ax)
    ay = jnp.where(stop, -v / params.dt, ay)
    return ax, ay


def _tool_accel(
    h: jnp.ndarray,
    grid_x: jnp.ndarray,
    grid_y: jnp.ndarray,
    tool_pos: jnp.ndarray,
    tool_vel: jnp.ndarray,
    params: SimParams,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    tx, ty, tz = tool_pos
    tvx, tvy = tool_vel
    speed = jnp.sqrt(tvx**2 + tvy**2)
    speed_safe = jnp.maximum(speed, params.tool_speed_eps)
    dir_x = tvx / speed_safe
    dir_y = tvy / speed_safe
    dir_x = jnp.where(speed > params.tool_speed_eps, dir_x, 0.0)
    dir_y = jnp.where(speed > params.tool_speed_eps, dir_y, 0.0)

    rx = grid_x - tx
    ry = grid_y - ty
    perp_x = -dir_y
    perp_y = dir_x
    long = rx * dir_x + ry * dir_y
    lat = rx * perp_x + ry * perp_y

    half_len = 0.5 * params.tool_length
    half_wid = 0.5 * params.tool_width
    softness = params.tool_softness
    mask_long = jax.nn.sigmoid((half_len - jnp.abs(long)) / softness)
    mask_lat = jax.nn.sigmoid((half_wid - jnp.abs(lat)) / softness)
    mask = mask_long * mask_lat

    z_imm = jnp.clip(h - tz, 0.0, h)
    tau = z_imm * (params.tool_drag + params.tool_splash * params.rho_bulk * speed * speed)
    h_safe = jnp.maximum(h, params.h_min)
    accel = tau / (params.rho_bulk * h_safe)
    return accel * dir_x * mask, accel * dir_y * mask


def _mass_flux_divergence(h: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, params: SimParams) -> jnp.ndarray:
    h_pad_x = jnp.pad(h, ((0, 0), (1, 1)), mode="edge")
    u_pad = jnp.pad(u, ((0, 0), (1, 1)), mode="edge")
    h_l = h_pad_x[:, :-1]
    h_r = h_pad_x[:, 1:]
    u_l = u_pad[:, :-1]
    u_r = u_pad[:, 1:]
    f_l = u_l * h_l
    f_r = u_r * h_r
    a = jnp.maximum(jnp.abs(u_l), jnp.abs(u_r))
    flux_x = 0.5 * (f_l + f_r) - 0.5 * a * (h_r - h_l)
    div_x = (flux_x[:, 1:] - flux_x[:, :-1]) / params.dx

    h_pad_y = jnp.pad(h, ((1, 1), (0, 0)), mode="edge")
    v_pad = jnp.pad(v, ((1, 1), (0, 0)), mode="edge")
    h_b = h_pad_y[:-1, :]
    h_t = h_pad_y[1:, :]
    v_b = v_pad[:-1, :]
    v_t = v_pad[1:, :]
    f_b = v_b * h_b
    f_t = v_t * h_t
    a = jnp.maximum(jnp.abs(v_b), jnp.abs(v_t))
    flux_y = 0.5 * (f_b + f_t) - 0.5 * a * (h_t - h_b)
    div_y = (flux_y[1:, :] - flux_y[:-1, :]) / params.dx

    return div_x + div_y


def step(
    state: State,
    tool_pos: jnp.ndarray,
    tool_vel: jnp.ndarray,
    grid_x: jnp.ndarray,
    grid_y: jnp.ndarray,
    params: SimParams,
) -> State:
    h = state.h
    h_safe = jnp.maximum(h, params.h_min)
    u = state.hu / h_safe
    v = state.hv / h_safe

    h_new = h - params.dt * _mass_flux_divergence(h, u, v, params)
    h_new = jnp.maximum(h_new, 0.0)

    dh_dx = _grad_x(h_new, params.dx)
    dh_dy = _grad_y(h_new, params.dx)
    du_dx = _grad_x(u, params.dx)
    du_dy = _grad_y(u, params.dx)
    dv_dx = _grad_x(v, params.dx)
    dv_dy = _grad_y(v, params.dx)

    adv_u = u * du_dx + v * du_dy
    adv_v = u * dv_dx + v * dv_dy

    slope_mag = jnp.sqrt(dh_dx**2 + dh_dy**2 + 1e-8)
    slope_gate = jax.nn.sigmoid((slope_mag - params.mu_s) / params.slope_eps)
    gdx = -params.g * dh_dx * slope_gate
    gdy = -params.g * dh_dy * slope_gate

    mu = _mu_I(u, v, h_new, params)
    ax_fric, ay_fric = _friction_accel(u, v, mu, params)
    ax_tool, ay_tool = _tool_accel(h_new, grid_x, grid_y, tool_pos, tool_vel, params)

    u_new = u + params.dt * (-adv_u + gdx + ax_fric + ax_tool)
    v_new = v + params.dt * (-adv_v + gdy + ay_fric + ay_tool)
    speed = jnp.sqrt(u_new**2 + v_new**2 + 1e-8)
    scale = jnp.minimum(1.0, params.speed_cap / speed)
    u_new = u_new * scale
    v_new = v_new * scale

    dry = h_new <= params.h_min
    u_new = jnp.where(dry, 0.0, u_new)
    v_new = jnp.where(dry, 0.0, v_new)
    hu_new = u_new * h_new
    hv_new = v_new * h_new
    return State(h=h_new, hu=hu_new, hv=hv_new)


def rollout(
    state: State,
    tool_pos: jnp.ndarray,
    tool_vel: jnp.ndarray,
    grid_x: jnp.ndarray,
    grid_y: jnp.ndarray,
    params: SimParams,
) -> Tuple[State, State]:
    def body(curr_state, inputs):
        pos, vel = inputs
        next_state = step(curr_state, pos, vel, grid_x, grid_y, params)
        return next_state, next_state

    return jax.lax.scan(body, state, (tool_pos, tool_vel))
