from __future__ import division, print_function
import sys
from time import process_time
import numpy as np

sys.path.append("../../..")
from hopfieldnetwork import HopfieldNetwork

timer = process_time()

### 4.2 Endliche Temperaturen
print("4.2.2 Endliche Temperaturen")
iterations = 20

N = 100
N_p = 30
p_N_vec = np.linspace(0.01, 0.3, N_p)
p_vec = np.ceil(p_N_vec * N).astype("int")
beta_vec = np.array((2, 3, 5, 10, 20, 50, 100))
P_spurious_states = np.zeros((N_p, 1 + beta_vec.size))  # 0,1 async; 2,3 sync
N_statistic1 = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # comandline argument
N_statistic2 = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # comandline argument

for i, p in enumerate(p_vec):
    print(p / N, end=" ", flush=True)
    for j in range(N_statistic1):
        hopfield_network = HopfieldNetwork(N=N)
        hopfield_network.train_pattern(2 * np.random.randint(2, size=(N, p)) - 1)
        initial_pattern = 2 * np.random.randint(2, size=N) - 1
        for j in range(N_statistic2):
            # update without finite temperatures
            hopfield_network.set_initial_neurons_state(np.copy(initial_pattern))
            hopfield_network.update_neurons(iterations, "async")
            d = np.sum(
                hopfield_network.S.reshape(hopfield_network.N, 1)
                != hopfield_network.xi,
                axis=0,
            )
            P_spurious_states[i, 0] += np.min((d, N - d)) / N > 0.05

            # update with finite temperatures
            for k, beta in enumerate(beta_vec):
                hopfield_network.set_initial_neurons_state(np.copy(initial_pattern))
                hopfield_network.update_neurons_with_finite_temp(
                    iterations, "async", beta=beta
                )
                d = np.sum(
                    hopfield_network.S.reshape(hopfield_network.N, 1)
                    != hopfield_network.xi,
                    axis=0,
                )
                P_spurious_states[i, 1 + k] += np.min((d, N - d)) / N > 0.05

P_spurious_states /= N_statistic1 * N_statistic2

np.savez(
    "../latex/images/4_2_2/spurious_states_finite_temperature_{}_{}_data".format(
        N_statistic1, N_statistic2
    ),
    p_N_vec,
    P_spurious_states,
    beta_vec,
)

print("\n\nProcess time: {:.3f} s".format(process_time() - timer))
