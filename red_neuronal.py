import tensorflow as tf
import numpy as np


class ODE_2nd(tf.keras.Model):
    def set_ODE_param(
        self,
        t0,
        x0,
        omega,
        omega_c,
        omega_j,
        kappa,
        N_spin,
        pesos,
        g_j,
        ro_j,
        gamma2,
        gamma,
        N_tot,
        DBC,
        pos_DBC,
        bin_resonator,
        line_couplings,
        tmax,
        factor,
    ):
        """
        Set parameters and initial conditions for the ODE
        """
        self.t0 = tf.constant([t0], dtype=tf.float32)
        self.x0_true = tf.constant(x0, dtype=tf.float32)
        self.omega = tf.constant(omega, dtype=tf.float32)
        self.omega_c = tf.constant(omega_c, dtype=tf.float32)
        self.omega_j = tf.constant(omega_j, dtype=tf.float32)
        self.kappa = tf.constant(kappa, dtype=tf.float32)
        self.N_spin = tf.constant(N_spin, dtype=tf.float32)
        self.pesos = tf.constant(pesos, dtype=tf.float32)
        self.g_j = tf.constant(g_j, dtype=tf.float32)
        self.ro_j = tf.constant(ro_j, dtype=tf.float32)
        self.gamma2 = tf.constant(gamma2, dtype=tf.float32)
        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.N_tot = tf.constant(N_tot, dtype=tf.int32)
        self.DBC = tf.constant(DBC, dtype=tf.float32)
        self.pos_DBC = tf.constant(pos_DBC, dtype=tf.float32)
        self.bin_resonator = tf.constant(bin_resonator, dtype=tf.float32)
        self.line_couplings = tf.constant(line_couplings, dtype=tf.float32)
        self.tmax = tf.constant(tmax, dtype=tf.int32)
        self.factor = tf.constant(factor, dtype=tf.float32)

    def train_step(self, data):
        """
        Training ocurrs here
        """
        t, x_true = data
        with tf.GradientTape() as tape:
            # * Initial conditions
            tape.watch(self.t0)
            tape.watch(self.x0_true)
            tape.watch(t)

            with tf.GradientTape() as tape0:
                tape0.watch(self.t0)
                x0_pred = self(self.t0, training=False)
                tape0.watch(x0_pred)

            with tf.GradientTape() as tape1:
                tape1.watch(t)
                x = self(t, training=False)
                tape1.watch(x)
            dx_dt = tape1.jacobian(x, t)
            # dx_dt = tf.linalg.diag_part(dx_dt)  # shape (100, 8, 1)
            dx_dt = tf.squeeze(dx_dt, axis=-1)
            dx_dt = tf.reshape(dx_dt, shape=x.shape)
            tape.watch(t)
            tape.watch(x)
            tape.watch(dx_dt)

            idx = tf.cast(tf.round(self.factor * t[:, 0]), tf.int32)  # (batch_size,)
            # idx = tf.clip_by_value(
            #     idx, 0, (self.tmax - 1)
            # )  # garantiza que idx ∈ [0, 1000]
            bin_values = tf.gather(self.bin_resonator, idx)  # (batch_size,)
            bin_values = tf.reshape(bin_values, (-1, 1))  # (batch_size, 1)
            line_values = tf.gather(self.line_couplings, idx)  # (batch_size,)
            line_values = tf.reshape(line_values, (-1, 1))  # (batch_size, 1)

            # ? Alternative ODE's order (2)
            lossODE = (
                self.compiled_loss(
                    dx_dt[:, 0],
                    (self.omega_c - self.omega) * x[:, 1]
                    - self.kappa * x[:, 0]
                    - tf.reduce_sum(
                        (
                            self.g_j
                            * tf.sqrt(self.N_spin)
                            * self.pesos
                            * self.ro_j
                            / tf.sqrt(2.0)
                            * x[:, self.N_tot + 2 : 2 * self.N_tot + 2]
                        )
                        - bin_values / tf.sqrt(2 * self.N_spin)
                    ),
                )
                + self.compiled_loss(
                    dx_dt[:, 1],
                    -(self.omega_c - self.omega) * x[:, 0]
                    - self.kappa * x[:, 1]
                    - tf.reduce_sum(
                        (
                            self.g_j
                            * tf.sqrt(self.N_spin)
                            * self.pesos
                            * self.ro_j
                            / tf.sqrt(2.0)
                        )
                        * x[:, 2 : self.N_tot + 2]
                    ),
                )
                + self.compiled_loss(
                    dx_dt[:, 2 : self.N_tot + 2],
                    -(self.omega_j - self.omega)
                    * x[:, self.N_tot + 2 : 2 * self.N_tot + 2]
                    - self.gamma2 * x[:, 2 : 2 + self.N_tot]
                    - tf.sqrt(2.0 * self.N_spin)
                    * self.g_j
                    * x[:, 2 * self.N_tot + 2 :]
                    * x[:, 1]
                    + line_values
                    * (
                        x[:, 2 + 2 * self.N_tot :]
                        - x[:, 2 + self.N_tot : 2 + 2 * self.N_tot]
                    ),
                )
                + self.compiled_loss(
                    dx_dt[:, self.N_tot + 2 : 2 * self.N_tot + 2],
                    (self.omega_j - self.omega) * x[:, 2 : self.N_tot + 2]
                    - self.gamma2 * x[:, self.N_tot + 2 : 2 * self.N_tot + 2]
                    - tf.sqrt(2.0 * self.N_spin)
                    * self.g_j
                    * x[:, 2 * self.N_tot + 2 :]
                    * x[:, 0]
                    + line_values
                    * (-x[:, 2 + 2 * self.N_tot :] + x[:, 2 : 2 + self.N_tot]),
                )
                + self.compiled_loss(
                    dx_dt[:, 2 * self.N_tot + 2 :],
                    tf.sqrt(2.0 * self.N_spin)
                    * self.g_j
                    * (
                        x[:, 2 : 2 + self.N_tot] * x[:, 1]
                        + x[:, 2 + self.N_tot : 2 + 2 * self.N_tot] * x[:, 0]
                    )
                    - self.gamma
                    * (
                        x[:, 2 + 2 * self.N_tot :]
                        + 1 / (self.N_spin * self.pesos * self.ro_j)
                    )
                    + line_values
                    * (
                        x[:, 2 + self.N_tot : 2 + 2 * self.N_tot]
                        - x[:, 2 : 2 + self.N_tot]
                    ),
                )
            )

            # * initial condition loss
            lossBC = self.compiled_loss(x0_pred, self.x0_true)

            # * "Chinchetas" loss
            DBC_NN = self(self.pos_DBC, training=False)
            loss = lossODE + lossBC + self.compiled_loss(self.DBC, DBC_NN) * 5

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(x_true, x)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.pop("mean_squared_error")
        metrics["lossreal"] = loss
        metrics["lossODE"] = lossODE
        metrics["lossBC"] = lossBC
        metrics["t"] = t
        metrics["x0"] = x0_pred[:, 0]
        metrics["y0"] = x0_pred[:, 1]
        metrics["vx0"] = x0_pred[:, 2]
        metrics["vy0"] = x0_pred[:, 3]
        return metrics


class FourierLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, scale=10.0):
        super(FourierLayer, self).__init__()
        self.output_dim = output_dim
        self.scale = scale

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.B = self.add_weight(
            shape=(input_dim, self.output_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            trainable=False,
        )

    def call(self, inputs):
        projection = tf.matmul(inputs, self.B) * self.scale
        return tf.concat([tf.sin(projection), tf.cos(projection)], axis=-1)


class LossODETracker(tf.keras.callbacks.Callback):
    def __init__(self, N_train, LossODE):
        super().__init__()
        self.N_train = N_train  # Guardas la variable que quieras pasar
        self.LossODE = LossODE

    def on_train_begin(self, logs=None):
        self.epoch_loss_ode = []

    def on_epoch_begin(self, epoch, logs=None):
        # Inicializar acumulador para esta época
        self.loss_ode_accum = 0.0

    def on_train_batch_end(self, batch, logs=None):
        # Sumar la lossODE de este batch al acumulador
        self.loss_ode_accum += float(logs["lossODE"])

    def on_epoch_end(self, epoch, logs=None):
        # Guardar la suma total de esta época
        self.LossODE.append(self.loss_ode_accum / self.N_train)
        print(
            f"\n Epoch {epoch + 1} - Total lossODE: {self.loss_ode_accum/self.N_train}"
        )
