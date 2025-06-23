import numpy as np
import scipy.integrate as integrate


def freq_distr_lorentz(x, omega_s, g_s):
    numerator = g_s
    denominator = np.pi * (g_s * g_s + (x - omega_s) ** 2)
    return numerator / denominator


def freq_distr(x, omega_s, g_s):
    prefactor = 1 / np.sqrt(2 * np.pi * g_s * g_s)
    exponential = np.exp(-((x - omega_s) ** 2) / (2 * g_s * g_s))
    return prefactor * exponential


def prepare_spindist(N_disc, g_s, omega_s, nl=8):
    freqs = np.linspace(omega_s - nl * g_s, omega_s + nl * g_s, N_disc + 2)
    delta_omega = freqs[1] - freqs[0]
    freqs = freqs[1:-1]

    weights = []
    for n in range(1, N_disc + 1):
        lim_inf = omega_s - nl * g_s + (n - 1) * delta_omega
        lim_sup = omega_s - nl * g_s + (n + 1) * delta_omega
        # Necesitamos usar SciPy para la integración
        weight = integrate.quad(freq_distr, lim_inf, lim_sup, args=(omega_s, g_s))[0]
        weights.append(weight)

    weights = np.array(weights)
    weights = weights / np.sum(weights)
    dist = [freqs, weights]

    return dist


def g_distr(x, tau, epsilon, norm):
    return np.power(x + epsilon, -tau) / norm


def prepare_couplingdist(
    couplings, g_min=1e-3, g_max=1.6e5, g_int=1.3, epsilon=1e-12, tau=2.93
):
    N_coup = len(couplings)
    norm_factor = (
        integrate.quad(g_distr, g_min, g_int, args=(tau, epsilon, 1.0))[0]
        + integrate.quad(g_distr, g_int, g_max, args=(tau, epsilon, 1.0))[0]
    )
    weights = []
    for i in range(N_coup):
        if i == N_coup - 1:
            lim_sup = g_max
        else:
            lim_sup = couplings[i + 1]
        lim_inf = couplings[i]
        weight = integrate.quad(
            g_distr, lim_inf, lim_sup, args=(tau, epsilon, norm_factor)
        )[0]
        weights.append(weight)

    weights = np.array(weights)
    weights = weights / np.sum(weights)
    couplings = np.array(couplings)
    dist = [couplings, weights]

    return dist


def _get_resonator_positions(Nx, Nz, xpos, zpos, x_res, r_min_res):
    delta_x = xpos - x_res
    res_bound = np.abs(delta_x) < r_min_res
    delta_x[res_bound] = r_min_res
    dx_exp = np.repeat(delta_x, Nz)
    z_exp = np.tile(zpos, Nx)
    r = np.sqrt(np.power(dx_exp, 2) + np.power(z_exp, 2))
    return r


def _get_transline_positions(Nx, Nz, xpos, zpos):
    x_exp = np.repeat(xpos, Nz)
    z_exp = np.tile(zpos, Nx)
    r = np.sqrt(np.power(x_exp, 2) + np.power(z_exp, 2))
    return r


def get_transline_distances(x_data, Nx, z_data, Nz):
    zpos = get_1d_mesh(*z_data, Nz)
    n_distr = len(Nx)
    line_dist = []
    for i in range(n_distr):
        xpos = get_1d_mesh(*x_data[i], Nx[i])
        line_dist.append(_get_transline_positions(Nx[i], Nz, xpos, zpos))
    line_dist = np.concatenate(line_dist)
    return line_dist


def get_1d_mesh(r_min, r_max, N):
    r_grid, dr = np.linspace(r_min, r_max, N + 1, retstep=True)
    r_pos = np.asarray(r_grid[:-1] + dr / 2.0)
    return r_pos


def construct_resonator_mesh(x_data, x_res, r_min, Nx, z_data, Nz, brms_max):
    n_distr = len(Nx)
    zpos = get_1d_mesh(*z_data, Nz)
    field_distr = []
    for i in range(n_distr):
        xpos = get_1d_mesh(*x_data[i], Nx[i])
        r_res = _get_resonator_positions(Nx[i], Nz, xpos, zpos, x_res, r_min[i])
        field_distr.append(brms_max[i] * r_min[i] / r_res)
    field_distr = np.concatenate(field_distr)
    return field_distr


def prepare_linecoupling_dist(r, pulse_power):

    irms = np.sqrt(pulse_power / 50)
    b_line = 2 * 1e-7 * irms
    dist_line = b_line / r

    return dist_line


def compute_spin_density(x_data, Nx, z_data, Nz, N_disc, N_spins, factors):
    n_distr = len(Nx)
    x = [get_1d_mesh(*x_data[i], Nx[i]) for i in range(n_distr)]
    z = get_1d_mesh(*z_data, Nz)
    dz = z[1] - z[0]
    dx = [x[i][1] - x[i][0] for i in range(n_distr)]

    factor1, factor2, factor3 = factors

    area1 = factor1 * dx[0] * dz
    area2 = factor2 * dx[1] * dz
    area3 = factor3 * dx[2] * dz
    total_area = (
        factor1 * Nx[0] * dx[0] + factor2 * Nx[1] * dx[1] + factor3 * Nx[2] * dx[2]
    ) * (Nz * dz)

    weight1 = area1 / total_area
    weight2 = area2 / total_area
    weight3 = area3 / total_area

    weight1 = np.asarray([weight1] * Nx[0]).repeat(Nz)
    weight2 = np.asarray([weight2] * Nx[1]).repeat(Nz)
    weight3 = np.asarray([weight3] * Nx[2]).repeat(Nz)

    print("Number of spins in each region")
    print(f"Wire: {np.sum(weight1)*N_spins:.2E}")
    print(f"Constriction: {np.sum(weight2)*N_spins:.2E}")
    print(f"Nano-constriction: {np.sum(weight3)*N_spins:.2E}")

    rho_spin = np.concatenate([weight1, weight2, weight3])
    return rho_spin
