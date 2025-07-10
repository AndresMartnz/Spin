import numpy as np

from spin_distributions import (
    prepare_spindist,
    prepare_couplingdist,
)
from eom import compute_refshift


### Parameters describing the resonator ###

kappa = 0.055 * 1e-3  # Cavity dissipation rate
kappa_c = 1.0e-3  # Scaling factor
kappa_ext = kappa_c * kappa  # Cavity-line coupling
omega_c = 2 * np.pi * (2.7340526876318556 - 0.0e-3)  # Resonator frequency
r_line = 75.0e-6

### Parameters describing the spin ensemble ###

omega_s = 2 * np.pi * 2.00028576 * 13.998 * 98.36e-3  # Central spin frequency
N_disc = 1  # Discretization of spin frequency distribution
g_s = 2 * np.pi * 10.0 * 1e-3  # Width of the spin frequency distribution
dist_freqs = prepare_spindist(N_disc, g_s, omega_s, nl=3)  # Frequency distribution
T2 = 400.0  # Decoherence time
T1 = 15.0 * 1e9  # Relaxation time
Gamma = 1 / T2  # Decoherence rate
gamma = 1 / T1  # Relaxation rate

### Parameters describing spin-resonator interaction ###

Delta = omega_s - omega_c  # Frequency detuning between subsystems
print(f"Detuning: {Delta/2/np.pi*1e3:.3f} MHz")
N_coup = 1  # Number of different spin-photon couplings
# couplings = np.logspace(3.95, 4, N_coup)  # Spin-photon couplings expressed in Hz
couplings = np.logspace(5, 6, N_coup)  # Spin-photon couplings expressed in Hz
dist_couplings = prepare_couplingdist(
    couplings, g_min=1e-2, g_max=1.6e5
)  # Coupling distribution
w_coup = dist_couplings[1]  # Weight of each coupling
collective_coupling = 8.0e6  # Fixed collective coupling
N_spins = collective_coupling**2 / np.sum(
    couplings * couplings * w_coup
)  # Total number of spins in the sample to account for the collective coupling
print(f"Number of spins: {N_spins:.2E}")

### Parameters describing the interaction through the excitation line ###

pot = -40.0  # Power of the input pulses
omega_driving = omega_s  # Driving frequency for the input pulse
tp_list = np.arange(0, 1001, 10)  # Pulse lengths

### Numerical details ###
delta_t = 1e-1  # Maximum integration step
tol = 1e-6  # Error tolerance in the integration
N_total = int(N_disc * N_coup)  # Total number of cluster spins to simulate
spin_freqs = np.repeat(dist_freqs[0], N_coup)
weights = np.repeat(dist_freqs[1], N_coup)
dist_freqs = [spin_freqs, weights]
dist_rescoup = np.tile(dist_couplings[0] * 2 * np.pi * 1e-9, N_disc)
w_coup = np.tile(w_coup, N_disc)

ref_shift = compute_refshift(
    dist_freqs, [dist_rescoup, w_coup, N_spins], omega_c, Gamma
)
xi = ref_shift
print(f"Reference shift: {ref_shift/2/np.pi*1e6:.3f} kHz")
