import os

import numpy as np
import matplotlib.pyplot as plt
import h5py


def prepare_results_file(
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
    folder="./",
):
    hiperparametros = {
        "kappa": kappa,
        "kappa_ext": kappa_ext,
        "omega_s": omega_s,
        "detuning": Delta,
        "omega_c": omega_c,
        "Ndisc": N_disc,
        "gamma_s": g_s,
        "N_coup": N_coup,
        "couplings": couplings,
        "Nspins": N_spins,
        "T1": T1,
        "T2": T2,
        "power": pot,
        "integration_delta_t": delta_t,
        "pulsetime_list": tp_list,
        "driving_frequency": omega_driving,
    }

    idx = 0
    filename = folder + f"SDNdisc{N_disc}Ncoup{N_coup}_{idx}.h5"

    while os.path.exists(filename) == True:
        idx += 1
        filename = folder + f"SDNdisc{N_disc}Ncoup{N_coup}_{idx}.h5"

    with h5py.File(filename, "w") as f:
        simulations = f.create_group(f"Experiment_omega_{omega_driving:.5f}")
        hyper = f.create_group("Hyperparameters")

        for key, value in hiperparametros.items():
            if isinstance(value, (int, float)):
                hyper.attrs[key] = value
            elif isinstance(value, (list, np.ndarray)):
                hyper.create_dataset(key, data=value)
            elif isinstance(value, str):
                hyper.attrs[key] = np.string_(value)

    with h5py.File(filename, "r+") as f:
        sim = f[f"Experiment_omega_{omega_driving:.5f}"]
        dset = sim.create_dataset("Results", shape=(len(tp_list), 6), dtype="float64")
    return filename


def update_file(
    filename, omega_driving, mv_x, mv_p, mv_sx, mv_sy, mv_sz, shift, ind_tp
):
    with h5py.File(filename, "r+") as f:
        sim = f[f"Experiment_omega_{omega_driving:.5f}"]
        dset = sim["Results"]
        dset[ind_tp, :] = np.array((mv_x, mv_p, mv_sx, mv_sy, mv_sz, shift))


def plot_results(filename):
    with h5py.File(filename, "r") as f:
        hyper = f["Hyperparameters"]
        print(f"Hiperparámetros de la simulación:")

        for attr_name, attr_value in hyper.attrs.items():
            print(f"  {attr_name}: {attr_value}")

        for dset_name in hyper:
            dset_value = hyper[dset_name]
            print(f"  {dset_name}: {dset_value}")

    with h5py.File(filename, "r") as f:
        hyper = f["Hyperparameters"]
        driving_freq = hyper.attrs["driving_frequency"]
        tp_list = hyper["pulsetime_list"][:]

    with h5py.File(filename, "r") as f:
        sim = f[f"Experiment_omega_{driving_freq:.5f}"]
        results = sim["Results"][:]

    shifts = np.abs(results[..., -1]) / 2 / np.pi * 1e6

    fig, ax = plt.subplots(1, 1, figsize=(13, 6))

    ax.plot(tp_list, shifts, "o", ms=12, mew=1.5, mfc="white", mec="firebrick")
    ax.set_ylabel(r"$\delta\omega_r/2\pi$ (kHz)")
    ax.set_xlabel(r"$t_{\mathrm{pump}}$ (ns)")

    # * We save the PINN loss history
    ruta_loss = f"RK"

    with open(ruta_loss, "w") as archivo:
        for (
            valor1,
            valor2,
        ) in zip(
            tp_list,
            shifts,
        ):
            archivo.write(f"{valor1}\t{valor2}\n")

    plt.show()
