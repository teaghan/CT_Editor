# Imports
from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
import matplotlib

# Ensure using PyQt5 backend
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Update toolbar for plot
NavigationToolbar.toolitems = (('Home', 'Reset original view', 'home', 'home'),
                               ('Back', 'Back to  previous view', 'back', 'back'),
                               ('Forward', 'Forward to next view', 'forward', 'forward'),
                               (None, None, None, None),
                               ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
                               (None, None, None, None))
def _update_buttons_checked(self):
    # sync button checkstates to match active mode (patched)
    if 'pan' in self._actions:
        self._actions['pan'].setChecked(self._active == 'PAN')
    if 'zoom' in self._actions:
        self._actions['zoom'].setChecked(self._active == 'ZOOM')
NavigationToolbar._update_buttons_checked = _update_buttons_checked

# Matplotlib canvas class to create figure
class MplSliceCanvas(Canvas):
    def __init__(self):
        self.fig = Figure()

        gs1 = gridspec.GridSpec(1, 1)
        gs1.update(left=0., right=1., bottom=0., top=1.)
        
        self.ax = self.fig.add_subplot(gs1[0])
                
        self.ax.tick_params(axis='both', which='both',
                       bottom=False, top=False, labelbottom=False,
                       left=False, right=False, labelleft=False)
        
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

# Matplotlib widget
class MplSliceWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget
        self.canvas = MplSliceCanvas()                  # Create canvas object
        self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.canvas)
        # Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self.canvas)        
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)