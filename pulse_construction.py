import numpy as np


class Pulse:
    def __init__(self, t0, tf, pulse_power, phi):
        self.t0 = t0
        self.tf = tf
        self.pot = 10 ** (pulse_power / 10) / 1000
        self.phi = phi


def gaussian_amplitude(t, t0, tf, pulse_power):

    n_sigma = 4
    tp = tf - t0
    sigma = tp / 2 / n_sigma
    t_mean = (tf + t0) / 2
    amplitude = pulse_power * np.exp(-((t - t_mean) ** 2) / (2 * sigma**2))

    return amplitude


def square_amplitude(t, t0, tf, pulse_power, rise=10.0):
    if t < t0:
        return 0
    elif t >= t0 and t < t0 + rise:
        return pulse_power * (t - t0) / rise
    elif t >= t0 + rise and t < tf - rise:
        return pulse_power
    elif t >= tf - rise and t < tf:
        return pulse_power * (tf - t) / rise
    else:
        return 0


def power_term(P, kappa_ext, omega_c):
    Energy = 1.054571817e-34 * omega_c * 1e9  # Joules
    return np.sqrt(2 * kappa_ext * P / Energy / 1e9)
