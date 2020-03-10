from __future__ import division, print_function
import sys, os

try:  # Python2
    import Tkinter as tk
    import tkFileDialog
except ImportError:  # Python3
    import tkinter as tk
    import tkinter.filedialog as tkFileDialog

import numpy as np

np.set_printoptions(threshold=np.inf)
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# define matplotlib fonts
SIZE_1 = 14
SIZE_2 = 16
SIZE_3 = 12
plt.rc("font", size=SIZE_1)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_1)  # fontsize of the axes title
plt.rc("xtick", labelsize=SIZE_1)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_1)  # fontsize of the tick labels
plt.rc("axes", labelsize=SIZE_2)  # fontsize of the x and y labels
plt.rc("legend", fontsize=SIZE_3)  # legend fontsize

from .libary import HopfieldNetwork
from .utils import AttrDict, images2xi
from .tk_utils import CreateToolTip, checkOS, ScrollSpinbox

BASE_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data")
EXAMPLES_PATH = os.path.join(DATA_PATH, "hopfield_network_examples")
PHYSICISTS_PATH = os.path.join(DATA_PATH, "famous_physicists")

# print system informations
print("Python:     {}.{}.{}".format(*sys.version_info[:3]))
print("Matplotlib: {}".format(mpl.__version__))
print("OS:         {}\n".format(checkOS()))


# main application
class GUI:
    def __init__(self):
        self.master = tk.Tk()
        self.master.withdraw()

        # load default settings
        self.settings = AttrDict()
        self.settings.default_N = 25
        self.settings.cmap = "Blues"
        self.settings.show_ticks = tk.BooleanVar()
        self.settings.show_ticks.set(False)
        self.settings.finite_temperature = tk.BooleanVar()
        self.settings.finite_temperature.set(False)
        self.settings.beta = tk.DoubleVar()
        self.settings.beta.set(10000)
        self.settings.default_canvas_width = 300
        self.settings.default_canvas_height = 300
        self.settings.stable_color = "green"
        self.settings.not_stable_color = "red"

        # define layout
        self.main_font = "Helvetica"
        self.button_font = self.main_font + " 10 bold"
        self.label_font = self.main_font + " 12 bold"
        self.energy_label_text = "H = {:0.3f}"
        self.time_label_text = "t = {}"
        self.highlightthickness = 1
        self.n_neurons_vec = (4, 9, 25, 36, 100, 400, 900, 1600, 2500, 3600, 10000)

        # initialize Hopfield network
        self.hopfield_network = HopfieldNetwork(N=self.settings.default_N)
        self.initialize_hopfield_network_variables()

        # build GUI
        self.configure_gui()
        self.create_widgets()
        self.master.deiconify()

    def initialize_hopfield_network_variables(self):
        self.N_sqrt = int(np.sqrt(self.hopfield_network.N))
        self.matrix_size = (self.N_sqrt, self.N_sqrt)
        self.input_matrix = -1 * np.ones(self.matrix_size, dtype="int8")
        self.input_matrix[0, 0] *= -1
        self.id_current_viewer_pattern = 0

    def configure_gui(self):
        self.master.title("Hopfield network")
        self.master.geometry("1280x600")
        self.master.minsize(1000, 500)
        self.master.option_add("*Font", self.main_font + " 10")
        self.master.tk_setPalette(
            background="white",
            activeBackground="#%02x%02x%02x" % (99, 158, 228),
            activeForeground="white",
        )
        self.master.protocol("WM_DELETE_WINDOW", self.master.quit)
        imgicon = tk.PhotoImage(file=os.path.join(DATA_PATH, "icon", "icon.gif"))
        self.master.tk.call("wm", "iconphoto", self.master._w, imgicon)

    def create_widgets(self):
        self.grid_configure(self.master, 2, 3)
        self.create_input_frame()
        self.create_output_frame()
        self.create_viewer_frame()
        self.create_menubar()

    def grid_configure(self, widget, N, M):
        for i in range(N):
            widget.grid_rowconfigure(i, weight=1)
        for j in range(M):
            widget.grid_columnconfigure(j, weight=1)

    def create_menubar(self):
        self.menubar = tk.Menu(self.master)
        self.master.config(menu=self.menubar)

        menu_network = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Network", menu=menu_network)
        menu_network.add_command(
            label="New network",
            command=lambda: self.initialize_new_network(self.hopfield_network.N),
        )
        menu_new_network_neuron_number = tk.Menu(menu_network, tearoff=0)
        menu_network.add_cascade(
            label="New network with", menu=menu_new_network_neuron_number
        )
        for i in self.n_neurons_vec:
            menu_new_network_neuron_number.add_command(
                label="{} neurons".format(i),
                command=lambda i=i: self.initialize_new_network(N=i),
            )
        menu_network.add_separator()
        menu_network.add_command(
            label="Open network from file", command=self.load_hopfield_network
        )
        menu_network.add_command(
            label="Save network to file", command=self.save_hopfield_network
        )
        menu_network.add_separator()
        menu_network.add_command(
            label="Add images to network", command=self.add_images_to_network
        )
        menu_build_network_neuron_number = tk.Menu(menu_network, tearoff=0)
        menu_network.add_cascade(
            label="Build network from images", menu=menu_build_network_neuron_number
        )
        for i in self.n_neurons_vec:
            menu_build_network_neuron_number.add_command(
                label="{} neurons".format(i),
                command=lambda i=i: self.build_network_from_images(N=i),
            )
        menu_network.add_separator()
        menu_network.add_command(label="Exit", command=self.master.quit)

        menu_options = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Options", menu=menu_options)
        menu_options.add_checkbutton(
            label="Finite temperature",
            variable=self.settings.finite_temperature,
            command=self.toggle_finite_temperature,
        )

        menu_view = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="View", menu=menu_view)
        menu_canvas_size = tk.Menu(self.menubar, tearoff=0)
        menu_view.add_cascade(label="Canvas size", menu=menu_canvas_size)
        menu_canvas_size.add_command(
            label="tiny", command=lambda: self.change_canvas_size(100, 100)
        )
        menu_canvas_size.add_command(
            label="small", command=lambda: self.change_canvas_size(200, 200)
        )
        menu_canvas_size.add_command(
            label="normal", command=lambda: self.change_canvas_size(300, 300)
        )
        menu_canvas_size.add_command(
            label="large", command=lambda: self.change_canvas_size(400, 400)
        )
        menu_colors = tk.Menu(self.menubar, tearoff=0)
        menu_view.add_cascade(label="Pattern colors", menu=menu_colors)
        menu_colors.add_command(
            label="Black/White", command=lambda: self.change_cmap_color("binary")
        )
        menu_colors.add_command(
            label="Reds", command=lambda: self.change_cmap_color("Reds")
        )
        menu_colors.add_command(
            label="Greens", command=lambda: self.change_cmap_color("Greens")
        )
        menu_colors.add_command(
            label="Blues", command=lambda: self.change_cmap_color("Blues")
        )
        menu_colors.add_command(
            label="Blue/Green", command=lambda: self.change_cmap_color("GnBu")
        )
        menu_view.add_checkbutton(
            label="Show ticks",
            variable=self.settings.show_ticks,
            command=self.toggle_ticks,
        )

        menu_examples = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Examples", menu=menu_examples)
        menu_examples.add_command(
            label="1 Pictures of famous physicists (10000 neurons)",
            command=lambda: self.build_network_from_images(
                input_path_vec=[
                    os.path.join(PHYSICISTS_PATH, f)
                    for f in os.listdir(PHYSICISTS_PATH)
                ],
                N=10000,
            ),
        )
        menu_examples.add_command(
            label="2 Random patterns (100 neurons, 20 patterns)",
            command=lambda: self.build_network_from_random_patterns(100, 20),
        )
        menu_examples.add_command(
            label="3 ABC (25 neurons, 3 patterns)",
            command=lambda: self.load_hopfield_network(
                path=os.path.join(EXAMPLES_PATH, "abc_25neu_3pat.npz")
            ),
        )
        menu_examples.add_command(
            label="4 Oscillating when sync update  (9 neurons, 3 patterns)",
            command=lambda: self.load_hopfield_network(
                path=os.path.join(
                    EXAMPLES_PATH, "oscillating_when_syn_update_9neu_5pat.npz"
                )
            ),
        )

        menu_debug = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Debug", menu=menu_debug)
        menu_debug.add_command(
            label="Update all frames", command=self.update_all_frames
        )
        menu_debug.add_separator()
        menu_debug.add_command(
            label="Print number of neurons",
            command=lambda: print(
                "Number of neurons: {}\n".format(self.hopfield_network.N)
            ),
        )
        menu_debug.add_command(
            label="Print number of saved pattern",
            command=lambda: print(
                "Number of saved pattern: {}\n".format(self.hopfield_network.p)
            ),
        )
        menu_debug.add_command(
            label="Print time step",
            command=lambda: print(
                "Current time step: {}\n".format(self.hopfield_network.t)
            ),
        )
        menu_debug.add_separator()
        menu_debug.add_command(
            label="Print neuron state S",
            command=lambda: print(
                "Shape of S {}\n".format(self.hopfield_network.S.shape),
                repr(self.hopfield_network.S.reshape(self.N_sqrt, self.N_sqrt)),
            ),
        )
        menu_debug.add_command(
            label="Print weight matrix",
            command=lambda: print(
                "Shape of w {}\n".format(self.hopfield_network.w.shape),
                repr(self.hopfield_network.w),
            ),
        )
        menu_debug.add_command(
            label="Print weight matrix diagonal",
            command=lambda: print(
                "Diagonal of w {}\n", np.diagonal(self.hopfield_network.w)
            ),
        )
        menu_debug.add_command(
            label="Print input pattern",
            command=lambda: print(
                "Shape of input pattern {}\n".format(self.input_matrix.shape),
                repr(self.input_matrix),
            ),
        )
        menu_debug.add_command(
            label="Print saved pattern",
            command=lambda: print(
                "Shape of saved pattern {}\n".format(self.hopfield_network.xi.shape),
                repr(
                    self.hopfield_network.xi[:, self.id_current_viewer_pattern].reshape(
                        self.N_sqrt, self.N_sqrt
                    )
                ),
            ),
        )

    # Input frame
    def create_input_frame(self):
        self.input_frame = tk.Frame(self.master)
        self.input_frame.grid(row=0, column=0, sticky="wens")
        # label energy input pattern
        self.label_energy_input_pattern = tk.Label(
            self.input_frame,
            text=self.energy_label_text.format(0),
            font=self.label_font,
        )
        self.label_energy_input_pattern.pack(expand=1)
        # create input canvas/image
        self.input_fig = plt.figure("Input pattern", facecolor="white")
        self.input_canvas = FigureCanvasTkAgg(self.input_fig, master=self.input_frame)
        self.input_canvas.get_tk_widget().configure(
            width=self.settings.default_canvas_width,
            height=self.settings.default_canvas_height,
            highlightbackground="black",
            highlightthickness=self.highlightthickness,
        )
        self.input_canvas.get_tk_widget().pack(side="top", expand=1)
        self.input_fig.canvas.mpl_connect(
            "button_press_event", self.change_input_pattern
        )
        self.input_fig.canvas.mpl_connect(
            "motion_notify_event", self.change_input_pattern
        )
        self.input_fig.canvas.mpl_connect(
            "button_release_event", lambda event: self.update_input_frame()
        )
        self.input_ax = self.input_fig.add_subplot(111)
        self.im_input_frame = self.input_ax.imshow(
            self.input_matrix,
            cmap=self.settings.cmap,
            vmin=-1,
            vmax=+1,
            interpolation="none",
        )
        self.set_axes_layout(self.input_fig, self.input_ax)
        self.input_canvas.draw()
        # label stability input pattern
        self.label_stability_input_pattern = tk.Label(
            self.input_frame, font=self.label_font
        )
        self.update_stability_label(
            self.input_matrix.flatten(), self.label_stability_input_pattern
        )
        self.label_stability_input_pattern.pack()
        # create input controls
        self.input_controls_frame = tk.LabelFrame(
            self.master, text="Input pattern", font=self.label_font
        )
        self.input_controls_frame.grid(row=1, column=0)
        self.button_set_initial = tk.Button(
            self.input_controls_frame,
            text="Set intial",
            font=self.button_font,
            command=self.set_input_pattern_as_initial_state,
        )
        CreateToolTip(
            self.button_set_initial,
            "Sets current input pattern as initial neuron state.",
        )
        self.button_set_initial.pack(side="left")
        self.button_save = tk.Button(
            self.input_controls_frame,
            text="Save/Train",
            font=self.button_font,
            command=self.train_input_pattern_to_hopfield_network,
        )
        CreateToolTip(self.button_save, "Save/Train input pattern to Hopfield network.")
        self.button_save.pack(side="left")
        self.button_set_random_input_pattern = tk.Button(
            self.input_controls_frame,
            text="Rand",
            font=self.button_font,
            command=self.set_random_input_pattern,
        )
        CreateToolTip(self.button_set_random_input_pattern, "Set random input pattern.")
        self.button_set_random_input_pattern.pack(side="left")
        self.button_clear = tk.Button(
            self.input_controls_frame,
            text="Clear",
            font=self.button_font,
            command=self.clear_input_pattern,
        )
        CreateToolTip(self.button_clear, "Clear input pattern.")
        self.button_clear.pack(side="left")

    def update_input_frame(self):
        self.im_input_frame.set_data(self.input_matrix)
        self.input_canvas.draw()
        self.label_energy_input_pattern.configure(
            text=self.energy_label_text.format(
                self.hopfield_network.compute_energy(self.input_matrix.flatten())
            )
        )
        self.update_stability_label(
            self.input_matrix.flatten(), self.label_stability_input_pattern
        )

    def change_input_pattern(self, event):
        if event.inaxes is None:
            return
        elif event.button == 1:
            self.input_matrix[
                int(round(event.ydata)), int(round(event.xdata))
            ] = 1  # x und y vertauscht wegen imshow
        elif event.button == 3:
            self.input_matrix[
                int(round(event.ydata)), int(round(event.xdata))
            ] = -1  # x und y vertauscht wegen imshow
        self.im_input_frame.set_data(self.input_matrix)
        self.input_canvas.draw()

    def set_input_pattern_as_initial_state(self):
        self.hopfield_network.set_initial_neurons_state(
            np.copy(self.input_matrix.flatten())
        )
        self.update_output_frame()

    def train_input_pattern_to_hopfield_network(self):
        self.hopfield_network.train_pattern(self.input_matrix.flatten())
        self.update_viewer_frame()
        self.update_output_frame()
        self.update_input_frame()

    def set_random_input_pattern(self):
        self.input_matrix = 2 * np.random.randint(2, size=self.matrix_size) - 1
        self.update_input_frame()

    def clear_input_pattern(self):
        self.input_matrix = -1 * np.ones(self.matrix_size)
        self.update_input_frame()

    # Output frame
    def create_output_frame(self):
        self.output_frame = tk.Frame(self.master)
        self.output_frame.grid(row=0, column=1, sticky="wens")
        # label output pattern energy/time
        self.label_output_pattern = tk.Label(
            self.output_frame,
            text=", ".join((self.energy_label_text, self.time_label_text)).format(0, 0),
            font=self.label_font,
        )
        self.label_output_pattern.pack(expand=1)
        # create output canvas/image
        self.output_fig = plt.figure("Output pattern", facecolor="white")
        self.output_canvas = FigureCanvasTkAgg(
            self.output_fig, master=self.output_frame
        )
        self.output_canvas.get_tk_widget().configure(
            width=self.settings.default_canvas_width,
            height=self.settings.default_canvas_height,
            highlightbackground="black",
            highlightthickness=self.highlightthickness,
        )
        self.output_canvas.get_tk_widget().pack(side="top", expand=1)
        self.output_ax = self.output_fig.add_subplot(111)
        self.im_output_frame = self.output_ax.imshow(
            self.hopfield_network.S.reshape(self.matrix_size),
            cmap=self.settings.cmap,
            vmin=-1,
            vmax=+1,
            interpolation="none",
        )
        self.set_axes_layout(self.output_fig, self.output_ax)
        self.output_canvas.draw()
        # label stability output pattern
        self.label_stability_output_pattern = tk.Label(
            self.output_frame, font=self.label_font
        )
        self.update_stability_label(
            self.hopfield_network.S, self.label_stability_output_pattern
        )
        self.label_stability_output_pattern.pack()
        # create output controls
        self.output_controls_frame = tk.LabelFrame(
            self.master, text="Output pattern", font=self.label_font
        )
        self.output_controls_frame.grid(row=1, column=1)
        self.grid_configure(self.output_controls_frame, 3, 3)
        self.button_sync_update = tk.Button(
            self.output_controls_frame,
            text="1 sync update",
            font=self.button_font,
            command=lambda: self.run_update(1, "sync"),
        )
        CreateToolTip(
            self.button_sync_update,
            "Run single synchronous update of the Hopfield network.",
        )
        self.button_sync_update.grid(row=0, column=0, sticky="wens")
        self.button_async_update = tk.Button(
            self.output_controls_frame,
            text="1 async update",
            font=self.button_font,
            command=lambda: self.run_update(1, "async"),
        )
        CreateToolTip(
            self.button_async_update,
            "Run single asynchronous update of the Hopfield network.",
        )
        self.button_async_update.grid(row=0, column=1, sticky="wens")
        self.button_randomize_initial_state = tk.Button(
            self.output_controls_frame,
            text="Randomize",
            font=self.button_font,
            command=self.set_randomize_state,
        )
        CreateToolTip(
            self.button_randomize_initial_state, "Randomize current neuron state."
        )
        self.button_randomize_initial_state.grid(row=0, column=2, sticky="wens")
        self.button_sync_update_10 = tk.Button(
            self.output_controls_frame,
            text="5 sync update",
            font=self.button_font,
            command=lambda: self.run_update(5, "sync"),
        )
        CreateToolTip(
            self.button_sync_update_10,
            "Run 10 synchronous updates of the Hopfield network.",
        )
        self.button_sync_update_10.grid(row=1, column=0, sticky="wens")
        self.button_async_update_10 = tk.Button(
            self.output_controls_frame,
            text="5 async update",
            font=self.button_font,
            command=lambda: self.run_update(5, "async"),
        )
        CreateToolTip(
            self.button_async_update_10,
            "Run 10 asynchronous updates of the Hopfield network.",
        )
        self.button_async_update_10.grid(row=1, column=1, sticky="wens")
        self.button_set_partial_initial = tk.Button(
            self.output_controls_frame,
            text="Set partial",
            font=self.button_font,
            command=self.set_partial_initial_state,
        )
        CreateToolTip(
            self.button_set_partial_initial,
            "Set partial pattern as intial neuron state.",
        )
        self.button_set_partial_initial.grid(row=1, column=2, sticky="wens")
        self.button_sync_update_max = tk.Button(
            self.output_controls_frame,
            text="Max sync update",
            font=self.button_font,
            command=lambda: self.run_update(0, "sync", run_max=True),
        )
        CreateToolTip(
            self.button_sync_update_max,
            "Run maximum iterations synchronous updates of the Hopfield network.",
        )
        self.button_sync_update_max.grid(row=2, column=0, sticky="wens")
        self.button_async_update_max = tk.Button(
            self.output_controls_frame,
            text="Max async update",
            font=self.button_font,
            command=lambda: self.run_update(0, "async", run_max=True),
        )
        CreateToolTip(
            self.button_async_update_max,
            "Run maximum iterations asynchronous updates of the Hopfield network.",
        )
        self.button_async_update_max.grid(row=2, column=1, sticky="wens")
        self.button_set_random_initial = tk.Button(
            self.output_controls_frame,
            text="Set random",
            font=self.button_font,
            command=self.set_random_initial_state,
        )
        CreateToolTip(
            self.button_set_random_initial, "Set random pattern as intial neuron state."
        )
        self.button_set_random_initial.grid(row=2, column=2, sticky="wens")
        # finite temperatures
        self.label_finite_temperature = tk.Label(
            self.output_controls_frame,
            text="Inverse temperature",
            font=self.button_font,
        )
        self.label_finite_temperature.grid(row=2, column=0, sticky="wens")
        self.label_finite_temperature.grid_remove()
        self.scrollspinbox_finite_temperature = ScrollSpinbox(
            self.output_controls_frame,
            textvariable=self.settings.beta,
            from_=0,
            to=1e10,
        )
        self.scrollspinbox_finite_temperature.grid(row=2, column=1, sticky="wens")
        self.scrollspinbox_finite_temperature.grid_remove()

    def update_output_frame(self):
        self.im_output_frame.set_data(self.hopfield_network.S.reshape(self.matrix_size))
        self.output_canvas.draw()
        self.label_output_pattern.configure(
            text=", ".join((self.energy_label_text, self.time_label_text)).format(
                self.hopfield_network.compute_energy(self.hopfield_network.S),
                self.hopfield_network.t,
            )
        )
        self.update_stability_label(
            self.hopfield_network.S, self.label_stability_output_pattern
        )

    def run_update(self, iterations, mode, run_max=False):
        if not self.settings.finite_temperature.get():
            self.hopfield_network.update_neurons(
                iterations=iterations, mode=mode, run_max=run_max
            )
        else:
            self.hopfield_network.update_neurons_with_finite_temp(
                iterations=iterations, mode=mode, beta=self.settings.beta.get()
            )
        self.update_output_frame()

    def set_partial_initial_state(self):
        self.hopfield_network.S[: int(self.hopfield_network.N / 2)] = -1
        self.hopfield_network.set_initial_neurons_state(self.hopfield_network.S)
        self.update_output_frame()

    def set_randomize_state(self):
        self.hopfield_network.S[
            np.random.choice(self.hopfield_network.N, int(self.hopfield_network.N / 10))
        ] *= -1
        self.hopfield_network.set_initial_neurons_state(self.hopfield_network.S)
        self.update_output_frame()

    def set_random_initial_state(self):
        self.hopfield_network.set_initial_neurons_state(
            2 * np.random.randint(2, size=self.hopfield_network.N) - 1
        )
        self.update_output_frame()

    # Viewer frame
    def create_viewer_frame(self):
        self.viewer_frame = tk.Frame(self.master)
        self.viewer_frame.grid(row=0, column=2, sticky="wens")
        # label viewer pattern energy
        self.label_energy_viewer_pattern = tk.Label(
            self.viewer_frame,
            text=self.energy_label_text.format(0),
            font=self.label_font,
        )
        self.label_energy_viewer_pattern.pack(expand=1)
        # create viewer canvas/image
        self.viewer_fig = plt.figure("Viewer pattern", facecolor="white")
        self.viewer_canvas = FigureCanvasTkAgg(
            self.viewer_fig, master=self.viewer_frame
        )
        self.viewer_canvas.get_tk_widget().configure(
            width=self.settings.default_canvas_width,
            height=self.settings.default_canvas_height,
            highlightbackground="black",
            highlightthickness=self.highlightthickness,
        )
        self.viewer_canvas.get_tk_widget().pack(side="top", expand=1)
        self.viewer_ax = self.viewer_fig.add_subplot(111)
        self.viewer_anno = self.viewer_ax.annotate(
            "No saved pattern.",
            xy=(0.5, 0.5),
            xycoords="figure fraction",
            va="center",
            ha="center",
            fontsize=20,
        )
        self.viewer_anno.set_text("No saved pattern.")
        self.im_viewer_frame = self.viewer_ax.imshow(
            -np.ones(self.matrix_size),
            cmap=self.settings.cmap,
            vmin=-1,
            vmax=+1,
            interpolation="none",
        )
        self.set_axes_layout(self.viewer_fig, self.viewer_ax)
        self.viewer_canvas.draw()
        # label stability viewer pattern
        self.label_stability_viewer_pattern = tk.Label(
            self.viewer_frame,
            font=self.label_font,
            text="not stable",
            fg=self.settings.not_stable_color,
        )
        self.label_stability_viewer_pattern.pack()
        # create viewer controls
        self.viewer_controls_frame = tk.LabelFrame(
            self.master, text="Saved pattern", font=self.label_font
        )
        self.viewer_controls_frame.grid(row=1, column=2)
        self.viewer_controls_subframe1 = tk.Frame(self.viewer_controls_frame)
        self.viewer_controls_subframe1.pack()
        self.viewer_controls_subframe2 = tk.Frame(self.viewer_controls_frame)
        self.viewer_controls_subframe2.pack()
        self.button_prev_saved_pattern = tk.Button(
            self.viewer_controls_subframe1,
            text="< prev",
            font=self.button_font,
            command=lambda: self.change_viewer_pattern(
                self.id_current_viewer_pattern - 1
            ),
        )
        CreateToolTip(self.button_prev_saved_pattern, "Show previous saved pattern.")
        self.button_prev_saved_pattern.pack(side="left")
        self.label_id_current_saved_pattern = tk.Label(
            self.viewer_controls_subframe1, text=" 0 / 0 ", font=self.button_font
        )
        self.label_id_current_saved_pattern.pack(side="left")
        self.button_next_saved_pattern = tk.Button(
            self.viewer_controls_subframe1,
            text="> next",
            font=self.button_font,
            command=lambda: self.change_viewer_pattern(
                self.id_current_viewer_pattern + 1
            ),
        )
        CreateToolTip(self.button_next_saved_pattern, "Show next saved pattern.")
        self.button_next_saved_pattern.pack(side="left")
        self.button_set_viewer_pattern_as_input_pattern = tk.Button(
            self.viewer_controls_subframe2,
            text="Set input",
            font=self.button_font,
            command=self.set_viewer_pattern_as_input_pattern,
        )
        CreateToolTip(
            self.button_set_viewer_pattern_as_input_pattern,
            "Set saved pattern as input pattern.",
        )
        self.button_set_viewer_pattern_as_input_pattern.pack(side="left")
        self.button_remove_current_viewer_pattern = tk.Button(
            self.viewer_controls_subframe2,
            text="Remove",
            font=self.button_font,
            command=lambda: self.remove_saved_pattern(self.id_current_viewer_pattern),
        )
        self.button_set_viewer_pattern_as_initial_state = tk.Button(
            self.viewer_controls_subframe2,
            text="Set initial",
            font=self.button_font,
            command=self.set_viewer_pattern_as_initial_state,
        )
        CreateToolTip(
            self.button_set_viewer_pattern_as_initial_state,
            "Set saved pattern as initial sate.",
        )
        self.button_set_viewer_pattern_as_initial_state.pack(side="left")
        CreateToolTip(
            self.button_remove_current_viewer_pattern,
            "Remove saved pattern from network.",
        )
        self.button_remove_current_viewer_pattern.pack(side="left")

    def update_viewer_frame(self):
        if self.hopfield_network.p != 0:
            self.label_id_current_saved_pattern.configure(
                text=" {} / {} ".format(
                    self.id_current_viewer_pattern + 1, self.hopfield_network.p
                )
            )
            self.viewer_anno.set_text("")
            self.im_viewer_frame.set_data(
                self.hopfield_network.xi[:, self.id_current_viewer_pattern].reshape(
                    self.matrix_size
                )
            )
            self.label_energy_viewer_pattern.configure(
                text=self.energy_label_text.format(
                    self.hopfield_network.compute_energy(
                        self.hopfield_network.xi[:, self.id_current_viewer_pattern]
                    )
                )
            )
            self.update_stability_label(
                self.hopfield_network.xi[:, self.id_current_viewer_pattern],
                self.label_stability_viewer_pattern,
            )
        else:
            self.label_id_current_saved_pattern.configure(text=" 0 / 0 ")
            self.label_stability_viewer_pattern.configure(
                text="not stable", fg=self.settings.not_stable_color
            )
            self.viewer_anno.set_text("No saved pattern.")
            self.im_viewer_frame.set_data(-np.ones(self.matrix_size))
            self.label_energy_viewer_pattern.configure(
                text=self.energy_label_text.format(0)
            )
        self.viewer_canvas.draw()

    def change_viewer_pattern(self, id_new_pattern):
        if self.hopfield_network.p == 0:
            pass
        else:
            if id_new_pattern < 0:
                self.id_current_viewer_pattern = self.hopfield_network.p - 1
            elif id_new_pattern >= self.hopfield_network.p:
                self.id_current_viewer_pattern = 0
            else:
                self.id_current_viewer_pattern = id_new_pattern

        self.update_viewer_frame()

    def set_viewer_pattern_as_initial_state(self):
        if self.hopfield_network.p != 0:
            self.hopfield_network.set_initial_neurons_state(
                np.copy(self.hopfield_network.xi[:, self.id_current_viewer_pattern])
            )
            self.update_output_frame()

    def set_viewer_pattern_as_input_pattern(self):
        if self.hopfield_network.p != 0:
            self.input_matrix = np.copy(
                self.hopfield_network.xi[:, self.id_current_viewer_pattern].reshape(
                    self.matrix_size
                )
            )
            self.update_input_frame()

    def remove_saved_pattern(self, i):
        self.hopfield_network.remove_pattern(i)
        if (
            self.id_current_viewer_pattern == self.hopfield_network.p
            and self.id_current_viewer_pattern > 0
        ):
            self.id_current_viewer_pattern -= 1
        self.update_all_frames()
        print(self.id_current_viewer_pattern)

    # Network functions
    def initialize_new_network(self, N):
        self.hopfield_network = HopfieldNetwork(N=N)
        self.initialize_hopfield_network_variables()
        self.update_all_frames()

    def load_hopfield_network(self, path=None):
        print("load hopfield network from file")
        if not path:
            path = tkFileDialog.askopenfilename(
                initialdir=EXAMPLES_PATH,
                filetypes=(("Numpy zipped archive", ("*.npz")), ("All Files", "*.*")),
                title="Choose a Hopfield etwork file.",
            )
        if not path:
            print("Cancel")
        else:
            self.hopfield_network = HopfieldNetwork(filepath=path)
            self.initialize_hopfield_network_variables()
            self.update_all_frames()

    def save_hopfield_network(self):
        path = tkFileDialog.asksaveasfilename(
            initialdir=EXAMPLES_PATH,
            filetypes=(("Numpy zipped archive", ("*.npz")), ("All Files", "*.*")),
            title="Save hopfiled network as Numpy zipped archive (NPZ).",
        )
        if not path:
            print("Cancel")
        else:
            self.hopfield_network.save_network(path)

    def add_images_to_network(self, input_path_vec=None):
        print("add image to hopfield network")
        if not input_path_vec:
            input_path_vec = tkFileDialog.askopenfilenames(
                initialdir=os.path.join(DATA_PATH, "images"),
                filetypes=(
                    ("Image files", ("*.jpg", "*.png", "*.gif", "*.tif")),
                    ("All Files", "*.*"),
                ),
                title="Choose (multiple) images files.",
            )
        if not input_path_vec:
            print("Cancel")
        else:
            xi = images2xi(input_path_vec, self.hopfield_network.N)
            self.hopfield_network.train_pattern(xi)
            self.update_all_frames()

    def build_network_from_images(self, input_path_vec=None, N=10000):
        print("build hopfield network from images")
        if not input_path_vec:
            input_path_vec = tkFileDialog.askopenfilenames(
                initialdir=os.path.join(DATA_PATH, "images"),
                filetypes=(
                    ("Image files", ("*.jpg", "*.png", "*.gif", "*.tif")),
                    ("All Files", "*.*"),
                ),
                title="Choose (multiple) images files.",
            )
        if not input_path_vec:
            print("Cancel")
        else:
            xi = images2xi(input_path_vec, N)
            self.hopfield_network = HopfieldNetwork(N=N)
            self.hopfield_network.train_pattern(xi)
            self.initialize_hopfield_network_variables()
            self.update_all_frames()

    def build_network_from_random_patterns(self, N, p):
        self.hopfield_network = HopfieldNetwork(N=N)
        self.hopfield_network.train_pattern(2 * np.random.randint(2, size=(N, p)) - 1)
        self.initialize_hopfield_network_variables()
        self.update_all_frames()

    # general functions
    def update_all_frames(self):
        self.im_input_frame.set_extent(
            (-0.5, self.N_sqrt - 0.5, self.N_sqrt - 0.5, -0.5)
        )
        self.im_viewer_frame.set_extent(
            (-0.5, self.N_sqrt - 0.5, self.N_sqrt - 0.5, -0.5)
        )
        self.im_output_frame.set_extent(
            (-0.5, self.N_sqrt - 0.5, self.N_sqrt - 0.5, -0.5)
        )
        self.update_input_frame()
        self.update_output_frame()
        self.update_viewer_frame()

    def update_stability_label(self, pattern, label):
        if self.hopfield_network.check_stability(pattern):
            label.configure(text="stable", fg=self.settings.stable_color)
        else:
            label.configure(text="not stable", fg=self.settings.not_stable_color)

    # Options functions
    def toggle_finite_temperature(self):
        if self.settings.finite_temperature.get():
            self.button_async_update_max.grid_remove()
            self.button_sync_update_max.grid_remove()
            self.label_finite_temperature.grid()
            self.scrollspinbox_finite_temperature.grid()
        else:
            self.label_finite_temperature.grid_remove()
            self.scrollspinbox_finite_temperature.grid_remove()
            self.button_async_update_max.grid()
            self.button_sync_update_max.grid()

    # View functions
    def change_cmap_color(self, color):
        self.settings.cmap = color
        self.im_input_frame.set_cmap(color)
        self.im_output_frame.set_cmap(color)
        self.im_viewer_frame.set_cmap(color)
        self.draw_all()

    def set_axes_layout(self, fig, ax):
        if not self.settings.show_ticks.get():
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        else:
            ax.axis("on")
            fig.tight_layout()

    def draw_all(self):
        self.input_canvas.draw()
        self.output_canvas.draw()
        self.viewer_canvas.draw()

    def toggle_ticks(self):
        print("Show axes {}".format(self.settings.show_ticks.get()))
        if self.settings.show_ticks.get():
            self.input_canvas.get_tk_widget().configure(highlightthickness=0)
            self.output_canvas.get_tk_widget().configure(highlightthickness=0)
            self.viewer_canvas.get_tk_widget().configure(highlightthickness=0)
        else:
            self.input_canvas.get_tk_widget().configure(
                highlightthickness=self.highlightthickness
            )
            self.output_canvas.get_tk_widget().configure(
                highlightthickness=self.highlightthickness
            )
            self.viewer_canvas.get_tk_widget().configure(
                highlightthickness=self.highlightthickness
            )
        self.set_axes_layout(self.input_fig, self.input_ax)
        self.set_axes_layout(self.output_fig, self.output_ax)
        self.set_axes_layout(self.viewer_fig, self.viewer_ax)
        self.draw_all()

    def change_canvas_size(self, width, height):
        self.input_canvas.get_tk_widget().configure(width=width, height=height)
        self.output_canvas.get_tk_widget().configure(width=width, height=height)
        self.viewer_canvas.get_tk_widget().configure(width=width, height=height)


def start_gui():
    app = GUI()
    app.master.mainloop()


if __name__ == "__main__":
    start_gui()
