# Hopfield network: a form of recurrent artificial neural network
An implementiation of a Hopfield network in Python. It can be used as Python package or with the included GUI.
****
## Package
Add the Package to your Pyton path and import the Hopfield network class:
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

In the Hopfield network GUI, the one-dimensional vectors of the neuron states are visualized as two-dimensional binary image. The user has the option to load different pictures/patterns into network and then start an asynchronously or synchronously update with or without fine temperatures. There are also prestored different networks in the examples tab.


**Run the GUI with:**\
`python2/python3 start_gui.py`


The Hopfield network GUI is divided into three frames:

### Input frame
The input frame (left) is the main point of interaction with the network. The user has the possibility here an input pattern by a left click on +1, accordingly by to right-click on -1. This does not affect the network for now. First. The network can be changed with the buttons below:
- **Set intial** sets the current input pattern as the start configuration of the neurons.
- **Save / Train** stores / trains the current input pattern into the Hopfield network.
- **Rand** sets a random input pattern.
- **Clear** sets all points of the input pattern to -1.

### Output frame
The output frame (center) indicates the current neuron configuration. With the
The network can be updated asynchronously or synchronously.
- **Sync update** starts a synchronous update.
- **Async update** starts an asynchronous update.
- **Randomize** randomly flips the state of one tenth of the neurons.
- **Set partial** sets the first half of the neurons to -1.
- **Set random** sets a random neuron state.

### Saved pattern frame
The Saved pattern frame (right) offers the possibility to save the already saved data in the network.
to view or remove images from the network.
- **Set initial** sets the currently displayed image as new neuron state.
- **Set input** sets the currently displayed image as input pattern.

### Menu bar
- In the **Network** tab, a new Hopfield network of any size can be initialized.
In addition, it is possible to save the current network as well as a stored network.
loaded network. Also, a raster graphic (JPG, PNG, GIF, TIF) can be added to the network
be saved or directly create a new network of multiple images.
- In the **Options** tab, the update with finite temperatures can be (de) activated.
- **View** offers options for visually changing the GUI.
- In the **Examples** tab you can load some different example networks.
