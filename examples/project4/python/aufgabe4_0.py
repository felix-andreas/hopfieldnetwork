from __future__ import division, print_function
import numpy as np
import os, sys
from time import process_time

sys.path.append("../../..")
from hopfieldnetwork import HopfieldNetwork
from hopfieldnetwork import images2xi, plot_network_development, DATA_DIR

timer = process_time()

### Hopfield Modells mit 10000 Neuronen: Bilder bekannter Physiker
print("4.0 Test des Hopfield Modells")
print("Hopfield Modells mit 10000 Neuronen: Bilder bekannter Physiker")
# Lade Bilder bekannter Physiker als NumPy array
N = 100 ** 2
path_list = [
    os.path.join(DATA_DIR, "images/famous_physicists", f)
    for f in [
        "einstein.jpg",
        "hilbert.jpg",
        "curie.jpg",
        "de_broglie.jpg",
        "dirac.jpg",
        "feynman.jpg",
        "heisenberg.jpg",
        "humboldt.jpg",
        "newton.jpg",
        "schroedinger.jpg",
    ]
]
xi = images2xi(path_list, N)
# Speichere/Trainiere Bilder bekannter Physiker im Netzwerk
hopfield_network = HopfieldNetwork(N=N)
hopfield_network.train_pattern(xi)

## Test: Bildrekonstruktion aus Teilbild
# Setze 'halben' Einstein als Startkonfiguartion
einstein = np.copy(xi[:, 0])
half_einstein = np.copy(xi[:, 0])
half_einstein[: int(N / 2)] = -1
hopfield_network.set_initial_neurons_state(np.copy(half_einstein))
# Plotte Neuronenkonfiguartion fuer 3 Zeitschritte
plot_network_development(
    hopfield_network, 3, "async", einstein, "../latex/images/reconstruct_einstein.pdf"
)

## Test: Bildrekonstruktion aus verrauschtem Bild
# Setze 'verrauschten' Hilbert als Startkonfiguartion
hilbert = np.copy(xi[:, 1])
faded_hilbert = np.copy(xi[:, 1])
faded_hilbert[np.random.choice(N, int(N / 4))] *= -1
hopfield_network.set_initial_neurons_state(np.copy(faded_hilbert))
# Plotte Neuronenkonfiguartion fuer 3 Zeitschritte
plot_network_development(
    hopfield_network, 3, "async", hilbert, "../latex/images/reconstruct_hilbert.pdf"
)

### Oszillationen im synchronen Modus
print("\n\n4.0 Oszillationen im synchronen Modus")
hopfield_network = HopfieldNetwork(N=4)
pattern1 = np.array([1, 1, -1, -1])
pattern2 = np.array([1, -1, 1, -1])
patterns = np.column_stack((pattern1, pattern2))
hopfield_network.train_pattern(patterns)
hopfield_network.set_initial_neurons_state(np.array([1, -1, -1, -1]))

plot_network_development(
    hopfield_network,
    6,
    "sync",
    pattern1,
    "../latex/images/sync_oscillation.pdf",
    anno_hamming=False,
)

print("\nProcess time: {:.3f} s".format(process_time() - timer))
