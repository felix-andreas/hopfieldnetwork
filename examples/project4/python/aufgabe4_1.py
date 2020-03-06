from __future__ import division, print_function
import sys
from time import process_time
import numpy as np

sys.path.append("../../..")
from hopfieldnetwork import HopfieldNetwork

timer = process_time()

### 4.1 Berechnung der Fehlerrate fuer 100 Neuronen
print("4.1 Berechnung der Fehlerrate fuer 100 Neuronen fuer p/N:")
N = 100
N_p = 46
p_N_vec = np.linspace(0.1, 1.0, N_p)
p_vec = np.ceil(p_N_vec * N).astype("int")
error_array = np.empty((N_p, 4))  # 0,1 async; 2,3 sync
t_steps_vec_max_iter = np.empty((N_p, 2))
N_statistic = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # comandline argument

for i, p in enumerate(p_vec):
    print(p / N, end=" ", flush=True)
    correct_bits = np.zeros(4)
    t_steps = np.zeros(2)
    for j in range(N_statistic):
        hopfield_network = HopfieldNetwork(N=N)
        hopfield_network.train_pattern(2 * np.random.randint(2, size=(N, p)) - 1)
        for k in range(p):
            # Async: Fehlerrate nach einer Iteration
            hopfield_network.set_initial_neurons_state(
                np.copy(hopfield_network.xi[:, k])
            )
            hopfield_network.update_neurons(1, "async")
            correct_bits[0] += np.sum(hopfield_network.S == hopfield_network.xi[:, k])
            # Async: Fehlerrate nach Erreichen des Fixpunktes
            hopfield_network.update_neurons(0, "async", run_max=True)
            correct_bits[1] += np.sum(hopfield_network.S == hopfield_network.xi[:, k])
            t_steps[0] += hopfield_network.t
            # Sync: Fehlerrate nach einer Iteration
            hopfield_network.set_initial_neurons_state(
                np.copy(hopfield_network.xi[:, k])
            )
            hopfield_network.update_neurons(1, "sync")
            correct_bits[2] += np.sum(hopfield_network.S == hopfield_network.xi[:, k])
            # Sync: Fehlerrate nach Erreichen des Fixpunktes
            hopfield_network.update_neurons(0, "sync", run_max=True)
            correct_bits[3] += np.sum(hopfield_network.S == hopfield_network.xi[:, k])
            t_steps[1] += hopfield_network.t
    error_array[i, :] = 1 - correct_bits / p / hopfield_network.N / N_statistic
    t_steps_vec_max_iter[i, :] = t_steps / p / N_statistic

np.savez(
    "../latex/images/4_1/fehlerrate_{}_data".format(N_statistic),
    p_N_vec,
    error_array,
    t_steps_vec_max_iter,
)

print("\n\nProcess time: {:.3f} s".format(process_time() - timer))
