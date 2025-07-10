#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from eom import solve_dynamics
from utils import prepare_results_file, plot_results
from test1 import *

###################### Dictionary with all relevant hyperparams ######################

filename = prepare_results_file(
    kappa,
    kappa_ext,
    omega_s,
    Delta,
    omega_c,
    N_disc,
    g_s,
    N_coup,
    couplings,
    N_spins,
    T1,
    T2,
    pot,
    delta_t,
    tp_list,
    omega_driving,
    folder="Results/",
)

########################################################################################
# %%

print("Solving system...", flush=True)

solve_dynamics(
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
)

# %%
plot_results(filename)
