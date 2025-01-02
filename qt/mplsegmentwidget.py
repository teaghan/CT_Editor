# Imports
from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib

# Ensure using PyQt5 backend
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Matplotlib canvas class to create figure
class MplSegCanvas(Canvas):
    def __init__(self):
        self.fig = Figure(figsize=(6,6))

        self.gs1 = gridspec.GridSpec(4, 4)
        self.gs1.update(left=0., right=1., bottom=0., top=1.)
        
        self.axs = []
        for i in range(4):
            for j in range(4):
                self.axs.append(self.fig.add_subplot(self.gs1[i,j]))
                self.axs[-1].axis('off')
                self.axs[-1].tick_params(axis='both', which='both',
                                         bottom=False, top=False, labelbottom=False,
                                         left=False, right=False, labelleft=False)        
        
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

# Matplotlib widget
class MplSegmentWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget
        self.canvas = MplSegCanvas()                  # Create canvas object
        self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)