from __future__ import division, print_function
import sys
from time import process_time
import numpy as np

sys.path.append("../../..")
from hopfieldnetwork import HopfieldNetwork


timer = process_time()

### 4.2 Spurious states
print("4.2.1 Spurious states")

N = 100
N_p = 30
p_N_vec = np.linspace(0.01, 0.3, N_p)
p_vec = np.ceil(p_N_vec * N).astype("int")
print(p_vec)
P_spurious_states = np.zeros(N_p)  # 0,1 async; 2,3 sync
N_statistic1 = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # comandline argument 1
N_statistic2 = int(sys.argv[2]) if len(sys.argv) > 2 else 1  # comandline argument 2

for i, p in enumerate(p_vec):
    for j in range(N_statistic1):
        hopfield_network = HopfieldNetwork(N=N)
        hopfield_network.train_pattern(2 * np.random.randint(2, size=(N, p)) - 1)
        for j2 in range(N_statistic2):
            hopfield_network.set_initial_neurons_state(
                2 * np.random.randint(2, size=N) - 1
            )
            hopfield_network.update_neurons(0, "async", run_max=True)
            d = np.sum(
                hopfield_network.S.reshape(hopfield_network.N, 1)
                != hopfield_network.xi,
                axis=0,
            )
            P_spurious_states[i] += np.min((d, N - d)) / N > 0.05
P_spurious_states /= N_statistic1 * N_statistic2

np.savez(
    "../latex/images/4_2_1/spurious_states_{}_{}_data".format(
        N_statistic1, N_statistic2
    ),
    p_N_vec,
    P_spurious_states,
)

print("\n\nProcess time: {:.3f} s".format(process_time() - timer))
