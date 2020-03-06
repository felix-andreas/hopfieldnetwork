from __future__ import division, print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .libary import HopfieldNetwork, hamming_distance


# Class for attributes
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# Image to network file utilities
def image2numpy_array(path, size):
    img_pil = Image.open(path)
    img_pil = img_pil.resize(size)
    img_pil = img_pil.convert("1")  # convert image to black and white
    img_np = np.zeros(img_pil.size, dtype="uint8")
    for i in range(img_pil.size[0]):
        for j in range(img_pil.size[1]):
            img_np[i, j] = img_pil.getpixel((i, j))
    img_np = np.where(img_np <= 0, 1, -1)
    return img_np.transpose()


def images2xi(path_vec, N):
    p = len(path_vec)
    xi = np.empty((N, p), dtype="int8")  # array with saved patterns
    N_sqrt = int(np.sqrt(N))
    for i, filename in enumerate(path_vec):
        xi[:, i] = image2numpy_array(filename, (N_sqrt, N_sqrt)).flatten()
    return xi


def images2network_file(N, input_path_vec, output_path):
    hopfield_network = HopfieldNetwork(N=N)
    xi = images2xi(input_path_vec, N)
    hopfield_network.train_pattern(xi)
    hopfield_network.save_network(output_path)


# plot network
def plot_network_development(
    network, timesteps, mode, exact_state, outputpath, anno_hamming=True
):
    fig, axarr = plt.subplots(1, timesteps)
    fig.set_size_inches(4 * timesteps, 4)
    N_sqrt = int(np.sqrt(network.N))
    for t in range(timesteps):
        axarr[t].imshow(
            np.copy(network.S.reshape((N_sqrt, N_sqrt))),
            cmap="Blues",
            vmin=-1,
            vmax=+1,
            interpolation="none",
        )
        axarr[t].get_xaxis().set_visible(False)
        axarr[t].get_yaxis().set_visible(False)
        hd = hamming_distance(exact_state, network.S)
        if anno_hamming:
            axarr[t].set_title("t = {}, hamming distance = {}".format(t, hd))
            fig.tight_layout()
        else:
            axarr[t].set_title("t = {}".format(t), fontsize=32)
        network.update_neurons(1, mode)
    fig.savefig(outputpath)


if __name__ == "__main__":

    def listdir_fullpath(d):
        return [os.path.join(d, f) for f in os.listdir(d)]

    dir = "images"
    pathvec = listdir_fullpath("images/test")
    N = 100
    images2network_file(N, pathvec, "hopfield_network_examples/test")
