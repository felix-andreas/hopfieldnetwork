from __future__ import print_function, division
import numpy as np


class HopfieldNetwork:
    def __init__(self, N=100, filepath=None):
        if not filepath:  # create new hopfield network with N neurons
            self.initialize_new_network(N)
        else:  # load hopfield network from file
            self.load_network(filepath)

    def initialize_new_network(self, N):
        self.N = N  # number of neurons
        self.w = np.zeros((N, N))  # weight matrix
        self.xi = np.empty((N, 0), dtype="int8")  # array with saved patterns
        self.S = -1 * np.ones(N, dtype="int8")  # state of the neurons
        self.p = 0  # number of saved patterns
        self.t = 0  # time steps

    def load_network(self, filepath):
        npzfile = np.load(filepath)
        self.w = npzfile["arr_0"]
        self.xi = npzfile["arr_1"]
        self.N = self.w.shape[0]
        self.S = -1 * np.ones(self.N, dtype="int8")
        self.p = self.xi.shape[1]
        self.t = 0

    def save_network(self, filepath):
        np.savez(filepath, self.w, self.xi)

    def train_pattern(self, input_pattern):
        self.w += construct_hebb_matrix(input_pattern)
        self.xi = np.column_stack((self.xi, input_pattern))
        self.p = self.xi.shape[1]

    def remove_pattern(self, i):
        if i < self.p:
            self.w -= construct_hebb_matrix(self.xi[:, i])
            self.xi = np.delete(self.xi, i, axis=1)
            self.p = self.xi.shape[1]
        else:
            print("There is no pattern to remove!")

    def set_initial_neurons_state(self, S_initial):  # uses S_initial in place
        if len(S_initial.shape) != 1 or S_initial.shape[0] != self.N:
            raise ValueError(
                "Unexpected shape/size of initial neuron state: {}".format(
                    S_initial.shape
                )
            )
        self.t = 0  # reset timer1 for new initial state vector
        self.S = S_initial  # set new initial neuron state

    def update_neurons(self, iterations, mode, run_max=False):
        self.t += iterations
        if mode == "async":
            for _ in range(iterations):
                for i in np.random.permutation(self.N):  # semi-random
                    self.S[i] = sign_0(np.dot(self.w[i, :], self.S))
            if run_max:
                while True:
                    last_S = np.copy(self.S)
                    for i in np.random.permutation(self.N):  # semi-random
                        self.S[i] = sign_0(np.dot(self.w[i, :], self.S))
                    if np.array_equal(last_S, self.S):
                        return
                    self.t += 1

        elif mode == "sync":
            for _ in range(iterations):
                self.S = sign_0(np.dot(self.w, self.S))
            if run_max:
                while True:
                    second_last_S = np.copy(self.S)
                    for i in range(2):
                        last_S = np.copy(self.S)
                        self.S = sign_0(np.dot(self.w, self.S))
                        if np.array_equal(last_S, self.S):
                            return
                        self.t += 1
                    if np.array_equal(second_last_S, self.S):
                        # print('Reached oscillating neuron state.')
                        return  # break if oscillating

    def update_neurons_with_finite_temp(self, iterations, mode, beta):
        self.t += iterations
        if mode == "async":
            for _ in range(iterations):
                for i in np.random.permutation(self.N):  # semi-random
                    self.S[i] = (
                        2
                        * (
                            1 / (1 + np.exp(-2 * beta * np.dot(self.w[i, :], self.S)))
                            >= np.random.rand(1)
                        )
                        - 1
                    )
        elif mode == "sync":
            for _ in range(iterations):
                self.S = (
                    2
                    * (
                        1 / (1 + np.exp(-2 * beta * np.dot(self.w, self.S)))
                        >= np.random.rand(self.N)
                    )
                    - 1
                )
        else:
            raise ValueError("Unkown mode: {}".format(mode))

    def compute_energy(self, S):
        return -0.5 * np.einsum("i,ij,j", S, self.w, S)

    def check_stability(self, S):  # stability condition
        return np.array_equal(S, sign_0(np.dot(self.w, S)))


def construct_hebb_matrix(xi):
    n = xi.shape[0]
    if len(xi.shape) == 1:
        w = np.outer(xi, xi) / n  # p = 1
    elif len(xi.shape) == 2:
        w = np.einsum("ik,jk", xi, xi) / n  # p > 1
    else:
        raise ValueError("Unexpected shape of input pattern xi: {}".format(xi.shape))
    np.fill_diagonal(w, 0)  # set diagonal elements to zero
    return w


def hamming_distance(x, y):
    return np.sum(x != y)


def sign_0(array):  # x=0 -> sign_0(x) = 1
    return np.where(array >= -1e-15, 1, -1)  # machine precision: null festhalten
