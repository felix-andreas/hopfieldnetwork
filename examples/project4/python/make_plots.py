from __future__ import division, print_function
import numpy as np
import matplotlib as mpl

mpl.use("Agg")  # wegen HU pool / screen session
print("Matplotlib: {}".format(mpl.__version__))
import matplotlib.pyplot as plt

# Aufgabe 4.1
# Plot Fehlerrate in Abhaengigkeit von p / N
N_statistic = 1000
npzfile = np.load("../latex/images/4_1/fehlerrate_{}_data.npz".format(N_statistic))
p_N_vec = npzfile["arr_0"]
error_array = npzfile["arr_1"]
t_steps_vec_max_iter = npzfile["arr_2"]

fig = plt.figure(1, figsize=(9, 5))
plt.plot(p_N_vec, error_array[:, 0], label="Async: Fehlerrate nach einer Iteration")
plt.plot(
    p_N_vec, error_array[:, 1], label="Async: Fehlerrate nach Erreichen des Fixpunktes"
)
plt.plot(p_N_vec, error_array[:, 2], label="Sync: Fehlerrate am Fixpunktes")
plt.plot(
    p_N_vec, error_array[:, 3], label="Sync: Fehlerrate am Fixpunktes oder Oscillation"
)
plt.xlabel("$p / N$")
plt.ylabel("Fehlerrate")
plt.legend()
plt.tight_layout()
plt.savefig("../latex/images/4_1/fehlerrate_{}.pdf".format(N_statistic))

fig2 = plt.figure(2, figsize=(9, 5))
plt.plot(p_N_vec, t_steps_vec_max_iter[:, 0], label="Async")
plt.plot(p_N_vec, t_steps_vec_max_iter[:, 1], label="Sync")
plt.xlabel("$p / N$")
plt.ylabel("Durchschnittliche Iterationen bis zum Erreichen des Fixpunktes")
plt.legend()
plt.tight_layout()
plt.savefig("../latex/images/4_1/iterationen_fixpunkt_{}.pdf".format(N_statistic))

# Aufgabe 4.2.1
N_statistic1, N_statistic2 = 1000, 1000
npzfile = np.load(
    "../latex/images/4_2_1/spurious_states_{}_{}_data.npz".format(
        N_statistic1, N_statistic2
    )
)
p_N_vec = npzfile["arr_0"]
P_spurious_states = npzfile["arr_1"]

plt.figure("4.2.1", figsize=(9, 5))
plt.plot(p_N_vec, P_spurious_states)
plt.xlabel("$p / N$")
plt.ylabel("$P(\\mathrm{spurious \; states})$")
plt.tight_layout()
plt.savefig(
    "../latex/images/4_2_1/spurious_states_{}_{}.pdf".format(N_statistic1, N_statistic2)
)

# Aufgabe 4.2.2
N_statistic1, N_statistic2 = 100, 100
npzfile = np.load(
    "../latex/images/4_2_2/spurious_states_finite_temperature_{}_{}_data.npz".format(
        N_statistic1, N_statistic2
    )
)
p_N_vec = npzfile["arr_0"]
P_spurious_states = npzfile["arr_1"]
beta_vec = npzfile["arr_2"]

plt.figure("4.2.2", figsize=(9, 5))
plt.plot(p_N_vec, P_spurious_states[:, 0], label="No finite temperature")
for k, beta in enumerate(beta_vec):
    plt.plot(p_N_vec, P_spurious_states[:, 1 + k], label="$\\beta$ = {}".format(beta))

plt.xlabel("$p / N$")
plt.ylabel("$P(\\mathrm{spurious \; states})$")
plt.legend()
plt.tight_layout()
plt.savefig(
    "../latex/images/4_2_2/spurious_states_finite_temperature_{}_{}.pdf".format(
        N_statistic1, N_statistic2
    )
)

# Plot sigmoid function
plt.figure("sigmoid", figsize=(9, 5))
h = np.linspace(-1, 1, 1000)
# beta_vec = np.array([0.1, 1, 2, 5, 10, 100])
sig = 1 / (1 + np.exp(-2 * np.outer(h, beta_vec)))
sig0 = 1 / (1 + np.exp(-2 * h * 10000))
plt.plot(h, sig0, label="No finite temperature")
for k, beta in enumerate(beta_vec):
    plt.plot(h, sig[:, k], label="$\\beta$ = {}".format(beta))

plt.legend()
plt.xlabel("$h$")
plt.ylabel("$\\frac{1}{1 + exp(-2 \\beta h)}$")
plt.tight_layout()
plt.savefig(
    "../latex/images/sigmoid_finite_temperatures.pdf".format(N_statistic1, N_statistic2)
)
