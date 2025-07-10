import time

# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras.optimizers import Adam

from functions import *
from red_neuronal import *
from eom import *
from test1 import *
from pulse_construction import *


# *We set the inicial conditions
x0 = [0, 0, 0, 0, -1]

# * We generate the values of z within the domain.
N_train_max = 400
tmax = 800

# * It gives the number of "chichetas" used
N_intervalos = 4

salto = int(N_train_max / N_intervalos)
salto_t = tmax / N_intervalos
norm = N_train_max

N_train = 0
tmed = 0
t0 = 0
pos_DBC = []
DBC = []
LossODE = []

# * Definición pulsos en array
bin_resonator = []
line_couplings = []
pot = 10 ** (pot / 10) / 1000
p_square_res = power_term(pot, kappa_ext, omega_c)
t_pulse = 0
factor = (N_train_max - 1) / tmax
for i in range(N_train_max):
    t_pulse = t_pulse + float(tmax / N_train_max)
    bin_resonator.append(square_amplitude(t_pulse, 0, tmax, p_square_res, rise=4.0))
    p_square_line = square_amplitude(t_pulse, 0, tmax, pot, rise=4.0)
    dist_linecoup = prepare_linecoupling_dist(r_line, p_square_line)
    line_couplings.append(dist_linecoup * 2 * np.pi * 28.0)

# * Input and output neurons (from the data)
input_neurons = 1
output_neurons = 3 * N_total + 2

# * Hiperparameters
batch_size = 1
epochs = 20000
lr = 0.000001

# * Stops after certain epochs without improving and safe the best weight
#! If the simulation ends normally instead of by this callback, the program will take last weights not best
callbacks = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=700, restore_best_weights=True
)
# * We create an auxiliary y_train.
start_time = time.time()
for i in range(N_intervalos):

    # * Define the model
    initializer = tf.keras.initializers.GlorotUniform(seed=5)
    activation = "tanh"
    input = Input(shape=(input_neurons,))
    x = Dense(500, activation=activation, kernel_initializer=initializer)(input)
    x = Dense(500, activation=activation, kernel_initializer=initializer)(x)
    x = Dense(500, activation=activation, kernel_initializer=initializer)(x)
    x = Dense(500, activation=activation, kernel_initializer=initializer)(x)
    x = Dense(500, activation=activation, kernel_initializer=initializer)(x)
    output = Dense(output_neurons, kernel_initializer=initializer, activation=None)(x)

    model = ODE_2nd(input, output)

    callback = SaveBestAtEnd(
        f"pesos/inter={N_intervalos}_t={tmax}_lr={lr}.h5",
        monitor="loss",
        mode="min",
    )

    # *Define the metrics, optimizer and loss
    loss = tf.keras.losses.MeanSquaredError()
    metrics = tf.keras.metrics.MeanSquaredError()
    optimizer = Adam(learning_rate=lr)  # standart 0.00001

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics, run_eagerly=False)
    model.summary()

    N_train = N_train + salto
    tmed = tmed + salto_t

    t_train = np.linspace(t0, tmed, N_train)
    t_train = np.reshape(t_train, (N_train, 1))

    y_train = np.zeros((t_train.shape[0], 1))
    omega = omega_driving

    # * Set ODE parameters and initial conditions
    model.set_ODE_param(
        t0=[t0],
        x0=x0,
        omega=omega,
        omega_c=omega_c,
        omega_j=dist_freqs[0],
        kappa=kappa,
        N_spin=N_spins,
        pesos=dist_freqs[1],
        g_j=dist_rescoup,
        ro_j=w_coup,
        gamma2=Gamma,
        gamma=gamma,
        N_tot=N_total,
        DBC=DBC,
        pos_DBC=pos_DBC,
        bin_resonator=bin_resonator,
        line_couplings=line_couplings,
        tmax=tmax,
        factor=factor,
    )
    # * Saves the best weights

    history = model.fit(
        t_train,
        y_train,
        batch_size,
        epochs=epochs,
        verbose=0,
        callbacks=[callback, LossODETracker(N_train=N_train, LossODE=LossODE)],
    )  # ,shuffle=False)

    x_pred = model.predict(t_train)
    DBC.append(x_pred[N_train - 1])
    pos_DBC.append(t_train[N_train - 1])

x0_NN = model(model.t0, training=False)

chincheta_DBC = []
chinchetas = np.array(DBC)
for l in pos_DBC:
    chincheta_DBC.append(l[0])


# * We save the PINN data trajectories
ruta_chinchetas = f"DBC/Ninter={N_intervalos}_t={tmax}_lr={lr}.txt"

with open(ruta_chinchetas, "w") as archivo:
    for valor1, valor2, valor3, valor4, valor5, valor6 in zip(
        chincheta_DBC,
        chinchetas[:, 0],
        chinchetas[:, 1],
        chinchetas[:, 2],
        chinchetas[:, 3],
        chinchetas[:, 4],
        # chinchetas[:, 5],
        # chinchetas[:, 6],
        # chinchetas[:, 7],
    ):
        archivo.write(f"{valor1}\t{valor2}\t{valor3}\t{valor4}\t{valor5}\t{valor6}\n")


# fig, ax = plt.subplots(dpi=100)

# # * PINN
# ax.plot(
#     t_train,
#     x_pred,
#     marker="o",
#     markersize=3.0,
#     linestyle="solid",
#     linewidth=0,
#     label=r"$\theta(\xi)$ PINN",
# )

# t_RK = np.linspace(0, 10, 10 * N_train_max)
# cos_sol = np.cos(t_RK)

# # * "Solución exacta"
# ax.plot(
#     t_RK,
#     cos_sol,
#     marker="o",
#     markersize=0,
#     linestyle="-",
#     linewidth=1,
#     label=r"$\theta(\xi)$ Analytical",
# )


# # *"Chinchetas"
# ax.plot(aux2, chinchetas, "o", markersize=6, label="chinchetas")

# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

# ax.set_title(rf"A={A}", fontsize=14)
# ax.set_xlabel(r"$t(s)$", fontsize=16)
# ax.set_ylabel(r"$x(m)$", fontsize=16)

# # * Adds a legend
# ax.legend()


# plt.grid(False)  # Optional
# # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
# #                mode="expand", borderaxespad=0, ncol=2,fontsize=12)
# ax.legend(fontsize=14, frameon=False)

# fig.set_size_inches(6, 6)

# plt.show()

# # * Summarize history for loss
# plt.plot(
#     np.log10(history.history["lossreal"]),
#     marker="o",
#     markersize=0.0,
#     linewidth=1,
#     label="loss",
# )
# plt.plot(
#     np.log10(LossODE),
#     marker="o",
#     markersize=0,
#     linewidth=1,
#     label="lossODE",
# )
# plt.plot(
#     np.log10(history.history["lossBC"]),
#     marker="o",
#     markersize=0,
#     linewidth=1,
#     label="lossBC",
# )
# ax.set_xlabel("época", fontsize=16)

# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)


# plt.legend()

# plt.legend(fontsize=14, frameon=False)
# plt.show()


# * We save the PINN loss history
ruta_loss = f"loss/Ninter={N_intervalos}_t={tmax}_lr={lr}.txt"

with open(ruta_loss, "w") as archivo:
    for valor1, valor2, valor3 in zip(
        history.history["lossreal"],
        LossODE,
        history.history["lossBC"],
    ):
        archivo.write(f"{valor1}\t{valor2}\t{valor3}\n")

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")
