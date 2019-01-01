<img src="data/icon/icon.svg" width="128" height="128" align="right"/>

# Hopfield network: a form of recurrent artificial neural network



An implementiation of a Hopfield network in Python. It can be used as Python package or with the included GUI.

### Requirements
    matplotlib
    numpy


## Package
Add the Package to your Pyton path and import the Hopfield network class:\
`from hopfieldnetwork import HopfieldNetwork`

### Usage

Create a new Hopfield network of size _N_ = 100:\
`hopfield_network1 = HopfieldNetwork(N=100)`

Open an already trained Hopfield network:\
`hopfield_network2 = HopfieldNetwork(filepath=’network2.npz’)`

Save a network as a file:\
`hopfield_network3.save_network(’path/to/file’)`

Save / Train Images into the Hopfield network:\
`hopfield_network1.train_pattern(input_pattern)`

Start an asynchronous update with 5 iterations:\
`hopfield_network1.update_neurons(iterations=5, mode=’async’)`

Compute the energy function of a pattern:\
`hopfield_network1.compute_energy(input_pattern)`


## GUI

![Hopfield network GUI](examples/project4/latex/images/gui_screenshot.png?raw=true)

In the Hopfield network GUI, the one-dimensional vectors of the neuron states are visualized as a two-dimensional binary image. The user has the option to load different pictures/patterns into network and then start an asynchronous or synchronous update with or without finite temperatures. There are also prestored different networks in the examples tab.


**Run the GUI with:**\
`python2/python3 start_gui.py`

### GUI Layout
The Hopfield network GUI is divided into three frames:

### Input frame
The input frame (left) is the main point of interaction with the network. The user can change the state of an input neuron by a left click to +1, accordingly by to right-click to -1. This will only change the state of the input pattern not the state of the actual network. The input pattern can be transfered to the network with the buttons below:
- **Set intial** sets the current input pattern as the start configuration of the neurons.
- **Save / Train** stores / trains the current input pattern into the Hopfield network.
- **Rand** sets a random input pattern.
- **Clear** sets all points of the input pattern to -1.

### Output frame
The output frame (center) shows the current neuron configuration.
- **Sync update** starts a synchronous update.
- **Async update** starts an asynchronous update.
- **Randomize** randomly flips the state of one tenth of the neurons.
- **Set partial** sets the first half of the neurons to -1.
- **Set random** sets a random neuron state.

### Saved pattern frame
The Saved pattern frame (right) shows the pattern currently saved in the network.
- **Set initial** sets the currently displayed image as new neuron state.
- **Set input** sets the currently displayed image as input pattern.
- **Remove** Removes the currently displayed image from the Hopfield network.

### Menu bar
- In the **Network** tab, a new Hopfield network of any size can be initialized.
In addition, it is possible to save the current network and load stored networks. Also, a raster graphic (JPG, PNG, GIF, TIF) can be added to the network or an entirly new network can be created out of multiple images.
- In the **Options** tab, the update with finite temperatures can be (de)activated.
- **View** offers options for visually changing the GUI.
- In the **Examples** tab, different example networks can be loaded.
