<img src="hopfieldnetwork/data/icon/icon.svg" width="64" height="64" align="left"/>

# hopfieldnetwork

> A Hopfield network is a special kind of an artifical neural network. It implements a
> so called associative or content addressable memory. This means that memory contents
> are not reached via a memory address, but that the network responses to an input
> pattern with that stored pattern which has the highest similarity.

**hopfieldnetwork** is a Python package which provides an implementation of a Hopfield
network. The package also includes a graphical user interface.

## Installing

Install and update using pip:

``` sh
pip install -U hopfieldnetwork
```

## Requirements

* Python 2.7 or higher (CPython or PyPy)
* NumPy
* Matplotlib

### Usage

Import the `HopfieldNetwork` class:

``` python
from hopfieldnetwork import HopfieldNetwork
```

Create a new Hopfield network of size _N_ = 100:

``` python
hopfield_network1 = HopfieldNetwork(N=100)
```

Save / Train Images into the Hopfield network:

``` python
hopfield_network1.train_pattern(input_pattern)
```

Start an asynchronous update with 5 iterations:

``` python
hopfield_network1.update_neurons(iterations=5, mode="async")
```

Compute the energy function of a pattern:

``` python
hopfield_network1.compute_energy(input_pattern)
```

Save a network as a file:

``` python
hopfield_network1.save_network("path/to/file")
```

Open an already trained Hopfield network:

``` python
hopfield_network2 = HopfieldNetwork(filepath="network2.npz")
```

### Graphical user interface

![Hopfield network GUI](examples/project4/latex/images/gui_screenshot.png?raw=true)

In the Hopfield network GUI, the one-dimensional vectors of the neuron states are
visualized as a two-dimensional binary image. The user has the option to load different
 pictures/patterns into network and then start an asynchronous or synchronous update
 with or without finite temperatures. There are also prestored different networks in the
  examples tab.

**Start the UI:**

If you installed the `hopfieldnetwork` package via pip, you can start the UI with:

    hopfieldnetwork-ui

Otherwise you can start UI by running `gui.py` as module:

    python -m hopfieldnetwork.gui

### GUI Layout

The Hopfield network GUI is divided into three frames:

**Input frame**\
The input frame (left) is the main point of interaction with the network. The user can
change the state of an input neuron by a left click to +1, accordingly by to right-click
 to -1. This will only change the state of the input pattern not the state of the actual
  network. The input pattern can be transfered to the network with the buttons below:

* **Set intial** sets the current input pattern as the start configuration of the neurons.
* **Save / Train** stores / trains the current input pattern into the Hopfield network.
* **Rand** sets a random input pattern.
* **Clear** sets all points of the input pattern to -1.

**Output frame**\
The output frame (center) shows the current neuron configuration.

* **Sync update** starts a synchronous update.
* **Async update** starts an asynchronous update.
* **Randomize** randomly flips the state of one tenth of the neurons.
* **Set partial** sets the first half of the neurons to -1.
* **Set random** sets a random neuron state.

**Saved pattern frame**\
The Saved pattern frame (right) shows the pattern currently saved in the network.

* **Set initial** sets the currently displayed image as new neuron state.
* **Set input** sets the currently displayed image as input pattern.
* **Remove** removes the currently displayed image from the Hopfield network.

**Menu bar**

* In the **Network** tab, a new Hopfield network of any size can be initialized.

In addition, it is possible to save the current network and load stored networks.
Also, a raster graphic (JPG, PNG, GIF, TIF) can be added to the network or an entirly
new network can be created out of multiple images.

* In the **Options** tab, the update with finite temperatures can be (de)activated.
* **View** offers options for visually changing the GUI.
* In the **Examples** tab, different example networks can be loaded.

## License

[GNU General Public License v3.0](https://github.com/andreasfelix/hopfieldnetwork/blob/master/LICENSE)

