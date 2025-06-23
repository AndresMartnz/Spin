import numpy as np
import tqdm

from spin_distributions import (
    prepare_linecoupling_dist,
)
from pulse_construction import power_term, square_amplitude, Pulse
from ode import solve_rkf45
from utils import update_file


def motion_equations_PRL_inh(
    t,
    z,
    omega_c,
    omega,
    kappa,
    kappa_ext,
    Gamma,
    gamma,
    dist_freqs,
    N_disc,
    dist_rescoup,
    N_coup,
    r_line,
    t0,
    tf,
    w_coup,
    N_spins,
    pulse_power,
):

    # Get variables from z
    N_total = int(N_disc * N_coup)
    x = z[0]
    p = z[1]
    sx = z[2 : N_total + 2]
    sy = z[N_total + 2 : 2 * N_total + 2]
    sz = z[2 * N_total + 2 :]

    # Parameters
    Dc = omega_c - omega
    Ds = dist_freqs[0] - omega
    weights = dist_freqs[1]

    p_square_line = square_amplitude(t, t0, tf, pulse_power, rise=4.0)
    dist_linecoup = prepare_linecoupling_dist(r_line, p_square_line)
    line_couplings = dist_linecoup * 2 * np.pi * 28.0

    p_square_res = power_term(pulse_power, kappa_ext, omega_c)
    bin_resonator = square_amplitude(t, t0, tf, p_square_res, rise=4.0)

    sqrt_N_spins = np.sqrt(N_spins)
    sqrt_2 = np.sqrt(2)

    # Compute common terms for the resonator
    common_factor = -dist_rescoup * sqrt_N_spins * weights * w_coup / sqrt_2

    # Update dx y dp
    dx = Dc * p - kappa * x - bin_resonator / np.sqrt(2 * N_spins)
    dp = -Dc * x - kappa * p

    dx += np.sum(common_factor * sy)
    dp += np.sum(common_factor * sx)

    # Compute dsx, dsy, dsz
    dsx = (
        -Ds * sy
        - Gamma * sx
        - sqrt_2 * dist_rescoup * sqrt_N_spins * sz * p
        + line_couplings * (sz - sy)
    )

    dsy = (
        Ds * sx
        - Gamma * sy
        - sqrt_2 * dist_rescoup * sqrt_N_spins * sz * x
        + line_couplings * (-sz + sx)
    )

    dsz = (
        sqrt_2 * dist_rescoup * sqrt_N_spins * (sx * p + sy * x)
        - gamma * (sz + 1 / (N_spins * weights * w_coup))
        + line_couplings * (sy - sx)
    )

    # Construct derivative vector
    derivadas = np.zeros_like(z)
    derivadas[0] = dx
    derivadas[1] = dp
    derivadas[2 : N_total + 2] = dsx
    derivadas[N_total + 2 : 2 * N_total + 2] = dsy
    derivadas[2 * N_total + 2 :] = dsz

    return derivadas


def compute_shift(evol_sz, dist_freqs, dist_coup, omega_c, Gamma):

    freqs, weights = dist_freqs[0], dist_freqs[1]
    couplings, w_g, N_spins = dist_coup[0], dist_coup[1], dist_coup[2]

    D_i = freqs - omega_c
    g_j = couplings * np.sqrt(N_spins * weights * w_g)
    G_J = g_j * g_j * (evol_sz + 1.0)
    spin_comp = G_J * D_i / (D_i * D_i + Gamma * Gamma)
    return spin_comp.sum()


def compute_refshift(dist_freqs, dist_coup, omega_c, Gamma):

    freqs, weights = dist_freqs[0], dist_freqs[1]
    couplings, w_g, N_spins = dist_coup[0], dist_coup[1], dist_coup[2]

    D_i = freqs - omega_c
    g_j = couplings * np.sqrt(N_spins * weights * w_g)
    G_J = g_j * g_j * (-1.0)
    spin_comp = G_J * D_i / (D_i * D_i + Gamma * Gamma)
    return spin_comp.sum()


def solve_dynamics(
    tp_list,
    tol,
    delta_t,
    omega_c,
    omega_driving,
    kappa,
    kappa_ext,
    Gamma,
    gamma,
    dist_freqs,
    N_disc,
    dist_rescoup,
    N_coup,
    r_line,
    pot,
    w_coup,
    N_spins,
    N_total,
    filename,
):
    for ind_tp, tp in enumerate(tp_list):
        t1 = tp
        t_delay = 50
        t_final = t1 + t_delay
        tspan = (0.0, t_final)

        t_eval = np.linspace(tspan[0], tspan[1], int(t_final / 4 + 1))
        train_pulse = [
            Pulse(0, t1, pot, 0.0),
            Pulse(t1, t_final, -np.inf, 0.0),
        ]

        ini_conditions = np.zeros((int(3 * N_total + 2)))
        ini_conditions[2 * N_total + 2 :] = -1.0

        t = tspan[0]
        y = ini_conditions
        ind_pulse = 0
        pulse = train_pulse[ind_pulse]
        print(f"Evolution for tp = {tp} ns")
        for tind, ts in enumerate(tqdm.tqdm(t_eval)):
            if ts != t:
                if ts > pulse.tf:
                    ind_pulse += 1
                    pulse = train_pulse[ind_pulse]

                pulse_power = pulse.pot
                if pulse_power * 1e9 > 1e-3:
                    dt = delta_t
                else:
                    dt = 500.0
                result = solve_rkf45(
                    motion_equations_PRL_inh,
                    y,
                    (t, ts),
                    tol,
                    dt,
                    omega_c,
                    omega_driving,
                    kappa,
                    kappa_ext,
                    Gamma,
                    gamma,
                    dist_freqs,
                    N_disc,
                    dist_rescoup,
                    N_coup,
                    r_line,
                    pulse.t0,
                    pulse.tf,
                    w_coup,
                    N_spins,
                    pulse_power,
                )
                y = result[-1, :]
                t = ts

        mv_sx = np.mean(y[2 : N_total + 2], axis=0)
        mv_sy = np.mean(y[N_total + 2 : 2 * N_total + 2], axis=0)
        mv_sz = np.mean(y[2 * N_total + 2 :], axis=0)
        mv_x = y[0]
        mv_p = y[1]

        shift = compute_shift(
            y[2 * N_total + 2 :],
            dist_freqs,
            [dist_rescoup, w_coup, N_spins],
            omega_c,
            Gamma,
        )

        update_file(
            filename, omega_driving, mv_x, mv_p, mv_sx, mv_sy, mv_sz, shift, ind_tp
        )
        print(mv_x, mv_p, mv_sx, mv_sy, mv_sz, (shift) / 2 / np.pi * 1e6, flush=True)
