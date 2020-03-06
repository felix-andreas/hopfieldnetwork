# tk utilities
import sys

try:  # Python2
    import Tkinter as tk
    import tkFileDialog
except ImportError:  # Python3
    import tkinter as tk
    import tkinter.filedialog as tkFileDialog


class CreateToolTip(object):
    def __init__(self, widget, text="widget info"):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.close)

    def enter(self, event=None):
        x, y, *_ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 0.1 * self.widget.winfo_width()
        y += self.widget.winfo_rooty() + self.widget.winfo_height()
        # creates a toplevel master
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the gui master
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(
            self.tw,
            text=self.text,
            justify="left",
            background="white",
            relief="solid",
            borderwidth=1,
        )
        label.pack(ipadx=1)

    def close(self, event=None):
        if self.tw:
            self.tw.destroy()


class ScrollSpinbox(tk.Spinbox):
    def __init__(self, *args, **kwargs):
        tk.Spinbox.__init__(self, *args, **kwargs)
        # Windows and macOS
        self.bind("<MouseWheel>", self.mouseWheel)
        # Linux
        self.bind("<Button-4>", self.mouseWheel)
        self.bind("<Button-5>", self.mouseWheel)

    def mouseWheel(self, event):
        # print(event.delta, event.num)
        if event.num == 5 or event.delta < 0:
            self.invoke("buttondown")
        elif event.num == 4 or event.delta > 0:
            self.invoke("buttonup")


# check OS
def checkOS():
    if sys.platform == "linux" or sys.platform == "linux2":
        operatingsystem = "linux"
    elif sys.platform == "darwin":
        operatingsystem = "macOS"
    elif sys.platform == "win32":
        operatingsystem = "windows"
    return operatingsystem
