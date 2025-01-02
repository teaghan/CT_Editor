import sys
sys.path.append('../')
import glob
import os
import numpy as np
import pydicom
from distutils.dir_util import copy_tree

from PyQt5 import QtWidgets, QtCore
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
from PyQt5.QtWidgets import QDialog, QMainWindow, QApplication, QFileDialog
from PyQt5.QtCore import QThread, QObject, pyqtSlot, pyqtSignal

from start_window import Ui_MainWindow
from ct_display import Ui_CTDisplay
from segmentation import Ui_Segmentation
from loading import Ui_Loading

import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import configparser
import torch
from create_luna_cubes import load_itk, resize_image
from training_fns import normalize, denormalize
from network import Refiner, EdgeKernel

model_name = 'hlt_to_nod_1'

cur_dir = os.path.dirname(__file__)
config_dir = os.path.join(cur_dir, '../configs/')
model_dir = os.path.join(cur_dir, '../models/')

def load_CTdicom(dicom_dir):
    '''Load CT scan from dicom directory.

    Parameters:
    dicom_dir (str): Path to dicom files.

    Returns:
    ct_scan (numpy array): 3D CT scan.
    spacing (floats): resolution of the CT scan.
    '''
    print('Reading dicom directory: %s'%dicom_dir)
    dicoms = glob.glob(dicom_dir+'/*')
    for dicom in dicoms:
        # Select directory with CT dcm files
        if os.path.isdir(dicom):
            dcmfiles = [pydicom.read_file(name, force=True) for name in np.sort(glob.glob(dicom+'/*'))]

            # Load ct data
            ct_scan = []
            z_locs = []
            for i, cur_slice in enumerate(dcmfiles):
                try:
                    # Pixel data with rescaled intensities
                    ct_scan.append(cur_slice.pixel_array * cur_slice.RescaleSlope + cur_slice.RescaleIntercept)
                    # Slice location
                    z_locs.append(float(cur_slice.ImagePositionPatient[2]))
                except AttributeError:
                    continue
            ct_scan = np.array(ct_scan)
            z_locs = np.array(z_locs)

            # Sort the slices
            ct_scan = ct_scan[np.argsort(z_locs)]
            z_locs = z_locs[np.argsort(z_locs)]
            print('\tCT scan has shape: %s' % str(ct_scan.shape), end=' ')

            # Obtain resolution info
            dx, dy = np.array(dcmfiles[0].PixelSpacing).astype(float)
            dz = float(dcmfiles[0].SliceThickness) 
            print('and a resolution of (%0.2f, %0.2f, %0.2f)mm' % (dz, dx, dy))
    
    return ct_scan, np.array([dz, dy, dx])

def load_RTdicom(dicom_dir, ct_scan, new_spacing):
    print('Reading dicom directory: %s'%dicom_dir)
    dicoms = glob.glob(dicom_dir+'/*')

    for dicom in dicoms:
        if os.path.isdir(dicom):
            old_dcm_dir = dicom
            dcmfiles = [pydicom.read_file(name, force=True) for name in np.sort(glob.glob(dicom+'/*'))]

            z_locs_orig = []
            for i, cur_slice in enumerate(dcmfiles):
                try:
                    # Slice location
                    z_locs_orig.append(float(cur_slice.ImagePositionPatient[2]))
                except AttributeError:
                    continue
            z_locs_orig = np.array(z_locs_orig)

            # Sort
            z_locs_orig = z_locs_orig[np.argsort(z_locs_orig)]

            # Obtain original resolution info
            dx, dy = np.array(dcmfiles[0].PixelSpacing).astype(float)
            dz = float(dcmfiles[0].SliceThickness) 

            # Create arrays of original pixel locations
            xstart, ystart, _ = np.array(dcmfiles[0].ImagePositionPatient).astype(float)
            zstart = np.min(z_locs_orig)
            x_locs_orig = np.arange(xstart, xstart + dx*dcmfiles[0].Columns, dx)
            y_locs_orig = np.arange(ystart, ystart + dy*dcmfiles[0].Rows, dy)

            # Create arrays of new pixel locations
            x_locs_new = np.arange(xstart, xstart + new_spacing[2]*ct_scan.shape[2], new_spacing[2])
            y_locs_new = np.arange(ystart, ystart + new_spacing[1]*ct_scan.shape[1], new_spacing[1])
            z_locs_new = np.arange(zstart, zstart + new_spacing[0]*ct_scan.shape[0], new_spacing[0])
            
            orig_locs = [z_locs_orig, y_locs_orig, x_locs_orig]
            new_locs = [z_locs_new, y_locs_new, x_locs_new]

        else: 
            rt_dcmfile = pydicom.read_file(dicom, force=True)
            ctrs = rt_dcmfile.ROIContourSequence
            # Loop through contours
            ctr_pts = []
            ROI_Number = []
            ROI_Name = []
            ROI_colour = []
            for j, ctr in enumerate(ctrs):
                # Contour points
                xs = []
                ys = []
                zs = []
                try:
                    for i in range(len(ctr.ContourSequence)):
                        pts = ctr.ContourSequence[i].ContourData
                        xs.append(pts[0::3])
                        ys.append(pts[1::3])
                        zs.append(pts[2::3])
                    p = np.vstack((np.concatenate(zs),np.concatenate(ys),np.concatenate(xs))).T
                    ctr_pts.append(p)
                    # Contour info
                    ROI_Number.append(ctr.ReferencedROINumber)
                    ROI_Name.append(rt_dcmfile.StructureSetROISequence[j].ROIName)
                    ROI_colour.append(ctr.ROIDisplayColor)
                except:
                    continue
            ROI_Number = np.array(ROI_Number).astype(int)
            print('\tFound %i contours.'%(len(ctr_pts)))
    # Convert pts to indices for the ct array
    ctr_indices = []
    for pts in ctr_pts:
        p_indices = []
        for pt in pts:
            p_indices.append([find_nearest(pt[0], z_locs_orig),
                             find_nearest(pt[1], y_locs_orig),
                             find_nearest(pt[2], x_locs_orig)])
        p_indices = np.vstack(p_indices)
        ctr_indices.append(p_indices)
    return old_dcm_dir, rt_dcmfile, orig_locs, new_locs, ROI_Name, ROI_colour, ctr_indices


class RescaleScan(QObject):
    def __init__(self, widget):
        super(RescaleScan, self).__init__()
        self.widget = widget

    finished = pyqtSignal()
    @pyqtSlot()
    def run(self):
        # Rescale the CT scan to a common resolution
        print('Rescaling scan...')
        self.widget.ct_scan, self.widget.new_spacing = resize_image(self.widget.ct_scan, 
                                                                    self.widget.orig_spacing)
        print("Finished rescaling.")
        self.finished.emit()
        
class LoadRT(QObject):
    def __init__(self, widget):
        super(LoadRT, self).__init__()
        self.widget = widget

    finished = pyqtSignal()
    @pyqtSlot()
    def run(self):
        # Rescale the CT scan to a common resolution
        print('Loading structures...')
        (self.widget.old_dcm_dir, self.widget.rt_dcmfile, self.widget.orig_locs, 
         self.widget.new_locs, self.widget.ROI_Name, self.widget.ROI_colour, 
         self.widget.ctr_indices) = load_RTdicom(self.widget.dicom_dir, self.widget.ct_scan, self.widget.new_spacing)
        
        print('Rescaling scan...')
        self.widget.new_ct_scan, _ = resize_image(self.widget.ct_scan, self.widget.new_spacing, self.widget.orig_spacing)

        print("Finished loading.")
        self.finished.emit()
        
def find_nearest(pt, array):
    return np.argmin(np.abs(array-pt))

def load_model(model_name):
    # Model configuration
    config = configparser.ConfigParser()
    config.read(config_dir+model_name+'.ini')
    print(config_dir+model_name+'.ini')
    architecture_config = config['ARCHITECTURE']
    print('\nCreating model: %s'%model_name)
    print('\nConfiguration:')
    for key_head in config.keys():
        if key_head=='DEFAULT':
            continue
        print('  %s' % key_head)
        for key in config[key_head].keys():
            print('    %s: %s'%(key, config[key_head][key]))
            
    # ARCHITECTURE PARAMETERS
    refiner_num_blocks = int(config['ARCHITECTURE']['refiner_num_blocks'])
    refiner_num_filters = int(config['ARCHITECTURE']['refiner_num_filters'])
    refiner_filter_len = int(config['ARCHITECTURE']['refiner_filter_len'])
    fade_perc = float(config['ARCHITECTURE']['fade_perc'])

    # Create Edge Kernel to be applied to output of refiner
    # Collect all possible shapes
    #shapes = np.array([[8,24,24],[10,30,30],[12,36,36],[14,42,42],[16,48,48]]).astype(int)
    shapes = np.array([[8,24,24],[10,30,30],[12,36,36],[14,42,42],[16,48,48],
                       [18,54,54],[20,60,60],[22,66,66],[24,72,72],[26,78,78],
                       [28,84,84],[30,90,90]]).astype(int)
    edgekernel = EdgeKernel(shapes, fade_perc=fade_perc, use_cuda=False)

    # BUILD THE NETWORK
    print('\nBuilding networks...')
    refiner = Refiner(num_blocks=refiner_num_blocks, in_features=1, 
                      nb_features=refiner_num_filters, 
                      filter_len=refiner_filter_len, 
                      init=True, edge_kernel=edgekernel, use_cuda=False)
    # Display model architecture
    print('\n\nREFINER ARCHITECTURE:\n')
    print(refiner)
    
    print('\nLoading saved model...')
    # Load model weights
    model_filename =  os.path.join(model_dir,model_name+'.pth.tar')
    checkpoint = torch.load(model_filename, map_location=lambda storage, loc: storage)
    refiner.load_state_dict(checkpoint['refiner'])
    
    return refiner
    
class EditSample(QObject):
    def __init__(self, widget):
        super(EditSample, self).__init__()
        self.widget = widget

    finished = pyqtSignal()
    @pyqtSlot()
    def run(self):
        print('Editing segment...')        
        sample = torch.clone(torch.from_numpy(self.widget.cur_seg.reshape(1, 1, 
                                                                          *self.widget.cur_seg.shape).astype(np.float32)))
        # Normalize between -1 and 1
        sample = normalize(sample)
        # Run network on sample and denormalize back to original pixel scaling
        self.widget.refiner.eval()
        ref_sample = denormalize(self.widget.refiner(sample))
        print('Edit complete.')
        self.widget.new_seg  = ref_sample[0,0].cpu().data.numpy()
        self.finished.emit()
        
def edit_sample(refiner, sample):
    print('Editing segment...')
    sample = torch.clone(torch.from_numpy(sample.reshape(1, 1, *sample.shape).astype(np.float32)))
    # Normalize between -1 and 1
    sample = normalize(sample)
    # Run network on sample and denormalize back to original pixel scaling
    refiner.eval()
    with torch.no_grad():
        ref_sample = denormalize(refiner(sample.detach()))
    print('Edit complete.')
    return ref_sample[0,0].detach().cpu().data.numpy()

class LoadingWindow(QDialog):
    def __init__(self, text):
        super().__init__()
        self.ui = Ui_Loading()
        self.ui.setupUi(self)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
                
        self.ui.window_text.setText(text)
        self.ui.window_text.setAlignment(QtCore.Qt.AlignCenter)
          
class StartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('CT Editor')
        
        # File type options
        self.ui.filetypeBox.addItems(['.dcm','.mhd'])
        
        # Button functionalities
        self.ui.loadButton.setEnabled(False)
        self.ui.locatefileButtom.clicked.connect(self.locate_scan)
        self.ui.closeButton.clicked.connect(self.close)
        self.ui.loadButton.clicked.connect(self.load_scan)
        
    def locate_scan(self):
        if self.ui.filetypeBox.currentText()=='.mhd':
            # Look for .mhd files
            self.filename, _ = QFileDialog.getOpenFileName(self,'QFileDialog.getOpenFileName()', 
                                                           '../data/ct_scans/', 'mhd(*.mhd)')
        elif self.ui.filetypeBox.currentText()=='.dcm':
            # Look for dicom directories
            self.filename = QFileDialog.getExistingDirectory(self, "Open a folder", '../data/ct_scans/',
                                                             QFileDialog.ShowDirsOnly)
        if self.filename[-4:] == '.mhd':
            self.dicom_dir = None
            self.ui.fileLabel.setText('File name:\n%s' % self.filename)
            self.ui.loadButton.setEnabled(True)
        elif os.path.isdir(self.filename):
            self.dicom_dir = self.filename+'/'
            self.ui.fileLabel.setText('Dicom directory:\n%s' % self.filename)
            self.ui.loadButton.setEnabled(True)
        
    def load_scan(self):
        
        # Load CT data
        print('Loading scan...')
        if self.filename[-4:] == '.mhd':
            self.ct_scan, _, self.orig_spacing, _ = load_itk(self.filename)
        elif os.path.isdir(self.filename):
            self.ct_scan, self.orig_spacing = load_CTdicom(self.filename)
        
        # Rescale the CT scan to a common resolution        
        self.RescalingWindow = LoadingWindow('Rescaling CT Scan to common resolution...')
        self.RescalingWindow.show()
        
        # 1 - create Worker and Thread inside the Form
        self.obj = RescaleScan(self)  # no parent!
        self.thread = QThread()  # no parent!
        # 3 - Move the Worker object to the Thread object
        self.obj.moveToThread(self.thread)
        # 4 - Connect Worker Signals to the Thread slots
        self.obj.finished.connect(self.thread.quit)
        # 5 - Connect Thread started signal to Worker operational slot method
        self.thread.started.connect(self.obj.run)
        # * - Thread finished signal will close the app if you want!
        self.thread.finished.connect(self.open_ct_window)
        # 6 - Start the thread
        self.thread.start()
                
        '''
        self.ct_scan = np.load('../data/ct_scans/ct_scan.npy')
        self.new_spacing = np.load('../data/ct_scans/new_spacing.npy')
        self.orig_spacing = np.load('../data/ct_scans/orig_spacing.npy')
        self.open_ct_window()
        '''
        
    def open_ct_window(self):
        
        self.RescalingWindow.close()
        '''
        np.save('../data/ct_scans/ct_scan.npy', self.ct_scan)
        np.save('../data/ct_scans/new_spacing.npy', self.new_spacing)
        np.save('../data/ct_scans/orig_spacing.npy', self.orig_spacing)
        '''
        # Create CT display
        self.NewCTWindow = CTWindow(self.dicom_dir, self.ct_scan, 
                                    self.new_spacing, self.orig_spacing)
        
        # Plot scan
        slice_num = np.rint(self.ct_scan.shape[1]/2).astype(int)+1
        self.NewCTWindow.ui.slice.setText(str(slice_num))
        self.NewCTWindow.ui.sliceScrollbar.setValue(slice_num)
        self.NewCTWindow.update_figure()
        
        # Open CT window
        self.NewCTWindow.exec_()
        
        
class CTWindow(QDialog):
    def __init__(self, dicom_dir, ct_scan, new_spacing, orig_spacing):
        super().__init__()
        self.timer_id = -1
        self.dicom_dir = dicom_dir
        self.ct_scan = ct_scan
        self.new_spacing = new_spacing
        self.orig_spacing = orig_spacing
        
        self.refiner = load_model(model_name)
        
        # Aspect ratio of plot
        mm_extent = self.ct_scan.shape*self.new_spacing
        self.aspect = mm_extent[0]/mm_extent[1] * self.ct_scan.shape[1]/self.ct_scan.shape[0]
        
        self.ui = Ui_CTDisplay()
        self.ui.setupUi(self)
        
        # Button functionalities
        self.ui.displayButton.clicked.connect(self.update_figure)
        self.ui.closeButton.clicked.connect(self.close)
        self.ui.editButton.clicked.connect(self.apply_edit)
        self.ui.undoButton.clicked.connect(self.undo_edit)
        self.ui.clearButton.clicked.connect(self.clear_edits)
        self.ui.saveButton.clicked.connect(self.save_scan)
        self.ui.segmentButton.clicked.connect(self.load_structures)
        self.ui.editButton.setEnabled(False)
        self.ui.undoButton.setEnabled(False)
        self.ui.clearButton.setEnabled(False)
        self.ui.saveButton.setEnabled(False)
        self.ui.segmentButton.setEnabled(False)
        
        # Scrollbar functionalities
        self.ui.sliceScrollbar.setMaximum(self.ct_scan.shape[1])
        self.ui.sliceScrollbar.sliderMoved.connect(self.sliderval)
        self.ui.sliceScrollbar.valueChanged.connect(self.sliderval)
        self.ui.slice.textChanged.connect(self.sliceval)
        
        # Cube size functionalities
        self.ui.cubelengthBox.valueChanged.connect(self.cube_size)
        #self.ui.cubelengthBox.setMaximum(42.9)
        self.cube_max = self.ui.cubelengthBox.maximum()
        self.cube_min = self.ui.cubelengthBox.minimum()
        self.cube_step = self.ui.cubelengthBox.singleStep()
        self.cube_size()
        
        _translate = QtCore.QCoreApplication.translate
        #self.ui.cubelenLabel.setText(_translate("CTDisplay", "Cube Side Length (15.6-42.9)mm:"))
        
        # Point selection functionality
        self.cid = self.ui.mpl_ct.canvas.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        # Locations, sizes, and data of old segments
        self.seg_locations = []
        self.seg_boxsizes = []
        self.seg_old_data = []
                
    def timerEvent(self, event):
        self.killTimer(self.timer_id)
        self.timer_id = -1
        self.ui.slice.setText(str(self.ui.sliceScrollbar.value()))
        
    def sliderval(self):
        if self.timer_id != -1:
            self.killTimer(self.timer_id)
        self.timer_id = self.startTimer(10)
        
    def sliceval(self):
        update = True
        try:
            self.ui.sliceScrollbar.setValue(int(self.ui.slice.text()))
        except ValueError:
            update = False
        if update:
            self.update_figure()
            
    def update_axis(self, slice_indx, vmin, vmax):
        self.ui.mpl_ct.canvas.ax.clear()
        # Plot scan
        self.ui.mpl_ct.canvas.ax.imshow(self.ct_scan[:,slice_indx], origin='lower', cmap=plt.cm.bone,
                                        aspect=self.aspect, vmin=vmin, vmax=vmax)
        self.draw_cube()
        self.ui.mpl_ct.canvas.draw()
        
    def update_figure(self):
        update = True
        try:
            # Place slice indices within range
            if int(self.ui.slice.text())>self.ct_scan.shape[1]:
                self.ui.slice.setText(str(self.ct_scan.shape[1]))
            elif int(self.ui.slice.text())<1:
                self.ui.slice.setText('1')
        except ValueError:
            update = False
        
        if update:
            self.update_axis(int(self.ui.slice.text())-1,
                             float(self.ui.vmin.text()), 
                             float(self.ui.vmax.text()))
                            
    def onclick(self, event):
        if event.inaxes is None:
            return
        if event.canvas == self.ui.mpl_ct.canvas.fig.canvas:
            if self.ui.mpl_ct.toolbar._active is None:
                # Location of point
                self.x_loc = event.xdata
                self.z_loc = event.ydata
                self.y_loc = int(self.ui.slice.text())-1
                # Check if within limits of scan
                self.check_lims()
                # Update window
                self.ui.xvalLabel.setText(str(np.round(self.x_loc,1)))
                self.ui.yvalLabel.setText(str(np.round(self.y_loc+1,1)))
                self.ui.zvalLabel.setText(str(np.round(self.z_loc,1)))
                self.update_figure()
                self.collect_seg()
                self.ui.editButton.setEnabled(True)
            else:
                print('Toolbar in use.')
    def check_lims(self):
        if (self.x_loc-self.box_size[2]/2)<0:
            self.x_loc = self.box_size[2]/2
        if (self.y_loc-self.box_size[1]/2)<0:
            self.y_loc = self.box_size[2]/2
        if (self.z_loc-self.box_size[0]/2)<0:
            self.z_loc = self.box_size[2]/2
        if (self.x_loc+self.box_size[2]/2)>self.ct_scan.shape[2]:
            self.x_loc = self.ct_scan.shape[2]-self.box_size[2]/2-1
        if (self.y_loc+self.box_size[1]/2)>self.ct_scan.shape[1]:
            self.y_loc = self.ct_scan.shape[1]-self.box_size[2]/2-1
        if (self.z_loc+self.box_size[0]/2)>self.ct_scan.shape[0]:
            self.z_loc = self.ct_scan.shape[0]-self.box_size[2]/2-1
                
    def cube_size(self):
        # Reset to closest grid spacing
        #grid = np.arange(15.6, 31.2+3.9,3.9)
        grid = np.arange(self.cube_min, self.cube_max+self.cube_step, self.cube_step)
        self.ui.cubelengthBox.setValue(grid[np.argmin(np.abs(self.ui.cubelengthBox.value()-grid))])
        # Calculate the cube size in pixels
        z_side_length = np.rint(self.ui.cubelengthBox.value()/self.new_spacing[0]).astype(int)
        self.box_size = np.array([z_side_length, z_side_length*3, z_side_length*3])
        self.update_figure()
        try:
            self.collect_seg()
        except AttributeError:
            pass
                
    def draw_cube(self):
        try:
            if (np.abs(self.y_loc+1-int(self.ui.slice.text()))<=(self.box_size[1]/2)):
                # Plot cube boundaries
                self.ui.mpl_ct.canvas.ax.plot([self.x_loc-self.box_size[2]/2, self.x_loc-self.box_size[2]/2],
                                              [self.z_loc-self.box_size[0]/2, self.z_loc+self.box_size[0]/2], c='r')
                self.ui.mpl_ct.canvas.ax.plot([self.x_loc+self.box_size[2]/2, self.x_loc+self.box_size[2]/2],
                                              [self.z_loc-self.box_size[0]/2, self.z_loc+self.box_size[0]/2], c='r')
                self.ui.mpl_ct.canvas.ax.plot([self.x_loc-self.box_size[2]/2, self.x_loc+self.box_size[2]/2],
                                              [self.z_loc-self.box_size[0]/2, self.z_loc-self.box_size[0]/2], c='r')
                self.ui.mpl_ct.canvas.ax.plot([self.x_loc-self.box_size[2]/2, self.x_loc+self.box_size[2]/2],
                                              [self.z_loc+self.box_size[0]/2, self.z_loc+self.box_size[0]/2], c='r')
        except AttributeError:
            pass
        
    def collect_seg(self):
        # Grab pre-existing CT data
        self.cur_seg = np.copy(self.ct_scan[np.rint(self.z_loc-self.box_size[0]/2).astype(int): 
                                       np.rint(self.z_loc+self.box_size[0]/2).astype(int),
                                       np.rint(self.y_loc-self.box_size[1]/2).astype(int): 
                                       np.rint(self.y_loc+self.box_size[1]/2).astype(int),
                                       np.rint(self.x_loc-self.box_size[2]/2).astype(int): 
                                       np.rint(self.x_loc+self.box_size[2]/2).astype(int)])    
        
    def apply_edit(self):        
        # Apply edit
        self.ui.statusLabel.setText('Editing cube...')
        
        # 1 - create Worker and Thread inside the Form
        self.obj = EditSample(self)  # no parent!
        self.thread = QThread()  # no parent!
        # 3 - Move the Worker object to the Thread object
        self.obj.moveToThread(self.thread)
        # 4 - Connect Worker Signals to the Thread slots
        self.obj.finished.connect(self.thread.quit)
        # 5 - Connect Thread started signal to Worker operational slot method
        self.thread.started.connect(self.obj.run)
        # * - Thread finished signal will close the app if you want!
        self.thread.finished.connect(self.insert_edit)
        # 6 - Start the thread
        self.thread.start()
        
    def insert_edit(self):        
        self.ct_scan[np.rint(self.z_loc-self.box_size[0]/2).astype(int):
                     np.rint(self.z_loc+self.box_size[0]/2).astype(int),
                     np.rint(self.y_loc-self.box_size[1]/2).astype(int): 
                     np.rint(self.y_loc+self.box_size[1]/2).astype(int),
                     np.rint(self.x_loc-self.box_size[2]/2).astype(int): 
                     np.rint(self.x_loc+self.box_size[2]/2).astype(int)] = self.new_seg
        # Update figure
        self.update_figure()
        # Save data
        self.seg_locations.append([self.z_loc, self.y_loc, self.x_loc])
        self.seg_boxsizes.append(self.box_size)
        self.seg_old_data.append(self.cur_seg)
        # Edit window
        self.ui.statusLabel.setText('Edit complete.')
        self.ui.undoButton.setEnabled(True)
        self.ui.clearButton.setEnabled(True)
        self.ui.saveButton.setEnabled(True)
        if self.dicom_dir is not None:
            self.ui.segmentButton.setEnabled(True)
        self.ui.savenameLabel.setText('')
        
    def undo_edit(self):
        # Collect last edit data
        z_loc, y_loc, x_loc = self.seg_locations[-1]
        box_size = self.seg_boxsizes[-1]
        old_seg = self.seg_old_data[-1]
        # Replace edit with old data
        self.ct_scan[np.rint(z_loc-box_size[0]/2).astype(int):
                     np.rint(z_loc+box_size[0]/2).astype(int),
                     np.rint(y_loc-box_size[1]/2).astype(int): 
                     np.rint(y_loc+box_size[1]/2).astype(int),
                     np.rint(x_loc-box_size[2]/2).astype(int): 
                     np.rint(x_loc+box_size[2]/2).astype(int)] = old_seg
        
        # Update figure
        self.update_figure()
        
        # Delete last edit data
        del self.seg_locations[-1]
        del self.seg_boxsizes[-1]
        del self.seg_old_data[-1]
        
        # Edit window
        self.ui.savenameLabel.setText('') 
        self.ui.statusLabel.setText('')
        if len(self.seg_locations)<1:
            self.ui.undoButton.setEnabled(False)
            self.ui.clearButton.setEnabled(False)
            self.ui.saveButton.setEnabled(False)
            
    def clear_edits(self):
        while len(self.seg_locations)>0:
            self.undo_edit()
            
    def save_scan(self):
        # User input for file path
        self.filename, _ = QFileDialog.getSaveFileName(self, 'QFileDialog.getSaveFileName()', 
                                                       '../data/ct_scans/','*')
        # Save scan
        np.savez(self.filename, ct_scan=self.ct_scan, 
                 seg_locations=self.seg_locations, 
                 seg_boxsizes=self.seg_boxsizes)
        # Update window
        self.ui.savenameLabel.setText('Scan saved as %s.npz' % self.filename)
        
    def load_structures(self):
        
        # Load structure data
        
        # Rescale the CT scan to a common resolution        
        self.LoadingStWindow = LoadingWindow('Rescaling CT Scan to original resolution and loading RT structures...')
        self.LoadingStWindow.show()
        
        # 1 - create Worker and Thread inside the Form
        self.obj = LoadRT(self)  # no parent!
        self.thread = QThread()  # no parent!
        # 3 - Move the Worker object to the Thread object
        self.obj.moveToThread(self.thread)
        # 4 - Connect Worker Signals to the Thread slots
        self.obj.finished.connect(self.thread.quit)
        # 5 - Connect Thread started signal to Worker operational slot method
        self.thread.started.connect(self.obj.run)
        # * - Thread finished signal will close the app if you want!
        self.thread.finished.connect(self.open_segment_window)
        # 6 - Start the thread
        self.thread.start()
        
    def open_segment_window(self):
        self.LoadingStWindow.close()
        
        # Create Segmentation window
        self.NewSegmentWindow = SegmentWindow(self.dicom_dir, self.new_ct_scan, self.new_spacing, self.orig_spacing,
                                              self.seg_locations, self.seg_boxsizes, self.old_dcm_dir,
                                              self.rt_dcmfile, self.orig_locs, 
                                              self.new_locs, self.ROI_Name, self.ROI_colour, 
                                              self.ctr_indices)
        
        # Plot scan
        slice_num = np.rint(self.new_ct_scan.shape[0]/2).astype(int)+1
        self.NewSegmentWindow.ui.slice.setText(str(slice_num))
        self.NewSegmentWindow.ui.sliceScrollbar.setValue(slice_num)
        self.NewSegmentWindow.update_figure()
        
        # Open Segment window
        self.NewSegmentWindow.exec_()
        

class SegmentWindow(QDialog):
    def __init__(self, dicom_dir, ct_scan, new_spacing, orig_spacing, seg_locations, seg_boxsizes, old_dcm_dir,
                 rt_dcmfile, orig_locs, new_locs, ROI_Name, ROI_colour, ctr_indices):
        super().__init__()
        self.timer_id = -1
        self.dicom_dir = dicom_dir
        self.ct_scan = ct_scan
        self.new_spacing = new_spacing
        self.orig_spacing = orig_spacing
        #self.seg_locations = seg_locations
        #self.seg_boxsizes = seg_boxsizes
        
        # Rescale seg locations and boxsizes
        self.seg_locations = []
        self.seg_boxsizes = []
        for loc, size in zip(seg_locations, seg_boxsizes):
            self.seg_locations.append([np.rint(loc[0]*new_spacing[0]/orig_spacing[0]).astype(int), 
                                       np.rint(loc[1]*new_spacing[1]/orig_spacing[1]).astype(int),
                                       np.rint(loc[2]*new_spacing[2]/orig_spacing[2]).astype(int)])
            self.seg_boxsizes.append([np.rint(size[0]*new_spacing[0]/orig_spacing[0]).astype(int), 
                                       np.rint(size[1]*new_spacing[1]/orig_spacing[1]).astype(int),
                                       np.rint(size[2]*new_spacing[2]/orig_spacing[2]).astype(int)])
        
        self.old_dcm_dir = old_dcm_dir
        self.rt_dcmfile = rt_dcmfile
        self.orig_locs = orig_locs
        self.new_locs = new_locs
        self.ROI_Name = ROI_Name
        self.ROI_colour = ROI_colour
        self.ctr_indices = ctr_indices
                
        # Aspect ratio of plot
        #mm_extent = self.ct_scan.shape*self.new_spacing
        self.aspect = 1#mm_extent[1]/mm_extent[2] * self.ct_scan.shape[2]/self.ct_scan.shape[1]
        
        self.ui = Ui_Segmentation()
        self.ui.setupUi(self)
        
        # Button functionalities
        self.ui.displayButton.clicked.connect(self.update_figure)
        self.ui.closeButton.clicked.connect(self.close)
        self.ui.undoButton.clicked.connect(self.undo_point)
        self.ui.clearButton.clicked.connect(self.clear_points)
        self.ui.saveButton.clicked.connect(self.save_dicom)
        self.ui.undoButton.setEnabled(False)
        self.ui.clearButton.setEnabled(False)
        self.ui.saveButton.setEnabled(False)
        
        # Scrollbar functionalities
        self.ui.sliceScrollbar.setMaximum(self.ct_scan.shape[0])
        self.ui.sliceScrollbar.sliderMoved.connect(self.sliderval)
        self.ui.sliceScrollbar.valueChanged.connect(self.sliderval)
        self.ui.slice.textChanged.connect(self.sliceval)
        
        # Show the cubes around the edits
        self.ui.showCubeBox.setChecked(True)
        self.ui.showCubeBox.stateChanged.connect(self.update_figure)
        
        # ROI functionalities
        self.ui.StBox.addItems(['None']+self.ROI_Name)
        self.ui.StBox.currentIndexChanged.connect(self.change_ROI)
        self.ui.createStButton.clicked.connect(self.add_ROI)
        
        # Point selection functionality
        self.cid = self.ui.mpl_ct.canvas.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.xs = []
        self.ys = []
        self.zs = []
        self.c = np.array([128,255,0])
        self.new_ROI_Names = []
        self.new_ctr_indices = []
        self.scatter = self.ui.mpl_ct.canvas.ax.scatter(self.xs, self.ys)
        
    def timerEvent(self, event):
        self.killTimer(self.timer_id)
        self.timer_id = -1
        self.ui.slice.setText(str(self.ui.sliceScrollbar.value()))
        
    def sliderval(self):
        if self.timer_id != -1:
            self.killTimer(self.timer_id)
        self.timer_id = self.startTimer(10)
        
    def sliceval(self):
        update = True
        try:
            self.ui.sliceScrollbar.setValue(int(self.ui.slice.text()))
        except ValueError:
            update = False
        if update:
            self.update_figure()
            
    def update_axis(self, slice_indx, vmin, vmax):
        self.ui.mpl_ct.canvas.ax.clear()
        # Plot scan
        self.ui.mpl_ct.canvas.ax.imshow(self.ct_scan[slice_indx], cmap=plt.cm.bone,
                                        aspect=self.aspect, vmin=vmin, vmax=vmax)
        if self.ui.showCubeBox.isChecked():
            self.draw_cubes()
        self.scatter = self.ui.mpl_ct.canvas.ax.scatter(self.xs, self.ys)
        self.draw_points()
        self.ui.mpl_ct.canvas.draw()
        
    def update_figure(self):
        update = True
        try:
            # Place slice indices within range
            if int(self.ui.slice.text())>self.ct_scan.shape[0]:
                self.ui.slice.setText(str(self.ct_scan.shape[0]))
            elif int(self.ui.slice.text())<1:
                self.ui.slice.setText('1')
        except ValueError:
            update = False
        
        if update:
            self.update_axis(int(self.ui.slice.text())-1,
                             float(self.ui.vmin.text()), 
                             float(self.ui.vmax.text()))
                            
    def check_lims(self):
        if self.x_loc < 0:
            self.x_loc = 0
        if self.y_loc < 0:
            self.y_loc = 0
        if self.z_loc < 0:
            self.z_loc = 0
        if self.x_loc > self.ct_scan.shape[2]:
            self.x_loc = self.ct_scan.shape[2]-1
        if self.y_loc > self.ct_scan.shape[1]:
            self.y_loc = self.ct_scan.shape[1]-1
        if self.z_loc > self.ct_scan.shape[0]:
            self.z_loc = self.ct_scan.shape[0]-1
                
    def draw_cubes(self):
        for (z_loc, y_loc, x_loc), box_size in zip(self.seg_locations, self.seg_boxsizes):
            try:
                if (np.abs(z_loc+1-int(self.ui.slice.text()))<=(box_size[0]/2)):
                    # Plot cube boundaries
                    self.ui.mpl_ct.canvas.ax.plot([x_loc-box_size[2]/2, x_loc-box_size[2]/2],
                                                  [y_loc-box_size[1]/2, y_loc+box_size[1]/2], c='r')
                    self.ui.mpl_ct.canvas.ax.plot([x_loc+box_size[2]/2, x_loc+box_size[2]/2],
                                                  [y_loc-box_size[1]/2, y_loc+box_size[1]/2], c='r')
                    self.ui.mpl_ct.canvas.ax.plot([x_loc-box_size[2]/2, x_loc+box_size[2]/2],
                                                  [y_loc-box_size[1]/2, y_loc-box_size[1]/2], c='r')
                    self.ui.mpl_ct.canvas.ax.plot([x_loc-box_size[2]/2, x_loc+box_size[2]/2],
                                                  [y_loc+box_size[1]/2, y_loc+box_size[1]/2], c='r')
            except AttributeError:
                pass
            
    def change_ROI(self):
        roi = self.ui.StBox.currentText()
        if ((self.ui.StBox.currentText() in self.ROI_Name)):#roi is not 'None':
            indx = self.ROI_Name.index(roi)
            self.xs = self.ctr_indices[indx][:,2]
            self.ys = self.ctr_indices[indx][:,1]
            self.zs = self.ctr_indices[indx][:,0]
            self.c = np.array(self.ROI_colour[indx]).astype(int)
        elif (roi=='None'):
            self.xs = []
            self.ys = []
            self.zs = []
            self.c = np.array([128,255,0])
        else:
            indx = self.new_ROI_Names.index(roi)
            if len(self.new_ctr_indices[indx])>0:
                self.xs = list(np.array(self.new_ctr_indices[indx])[:,2])
                self.ys = list(np.array(self.new_ctr_indices[indx])[:,1])
                self.zs = list(np.array(self.new_ctr_indices[indx])[:,0])
            else:
                self.xs = []
                self.ys = []
                self.zs = []
            self.c = np.array([128,255,0])
        self.draw_points()
        
    def add_ROI(self):
        self.ui.StBox.addItems([self.ui.stNameLine.text()])
        self.xs = []
        self.ys = []
        self.zs = []
        self.c = np.array([128,255,0])
        self.new_ROI_Names.append(self.ui.stNameLine.text())
        self.new_ctr_indices.append([])
        index = self.ui.StBox.findText(self.ui.stNameLine.text(), QtCore.Qt.MatchFixedString)
        if index >= 0:
             self.ui.StBox.setCurrentIndex(index)
        
    def onclick(self, event):
        if event.inaxes is None:
            return
        if event.canvas == self.ui.mpl_ct.canvas.fig.canvas:
            if ((self.ui.mpl_ct.toolbar._active is None)&
                (self.ui.StBox.currentText() not in self.ROI_Name)&
                (self.ui.StBox.currentText() != 'None')):
                # Location of point
                self.x_loc = event.xdata
                self.y_loc = event.ydata
                self.z_loc = int(self.ui.slice.text())-1
                # Check if within limits of scan
                self.check_lims()
                # Save points
                self.xs.append(self.x_loc)
                self.ys.append(self.y_loc)
                self.zs.append(self.z_loc)      
                self.draw_points()
                
                indx = self.new_ROI_Names.index(self.ui.StBox.currentText())
                self.new_ctr_indices[indx].append([self.z_loc, self.y_loc, self.x_loc])
                
                self.ui.undoButton.setEnabled(True)
                self.ui.clearButton.setEnabled(True)
                self.ui.saveButton.setEnabled(True)
            else:
                print('Toolbar in use or not a new segment selected.')
            
    def draw_points(self):
        zs = np.array(self.zs)
        indices = np.where(zs==(int(self.ui.slice.text())-1))
        xs = np.array(self.xs)[indices]
        ys = np.array(self.ys)[indices]
        c = self.c/255
        try:
            self.scatter.set_offsets(np.array([xs,ys]).T)
            self.scatter.set_sizes(np.array([1]*len(xs)))
            self.scatter.set_color(np.array([c]*len(xs)))
            self.ui.mpl_ct.canvas.draw()
        except AttributeError:
            pass
        
    def undo_point(self):
        # Delete last point
        del self.xs[-1]
        del self.ys[-1]
        del self.zs[-1]
        
        # Update figure
        self.draw_points()
        
        # Edit window
        self.ui.savenameLabel.setText('') 
        self.ui.statusLabel.setText('')
        if len(self.seg_locations)<1:
            self.ui.undoButton.setEnabled(False)
            self.ui.clearButton.setEnabled(False)
            self.ui.saveButton.setEnabled(False)
            
    def clear_points(self):
        while len(self.xs)>0:
            self.undo_point()
            
    def save_dir(self):
        
        dialog = QFileDialog()
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setWindowTitle('New Patient Directory')
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        #dialog.setNameFilter(nameFilter)
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setDirectory('../data/ct_scans/')
        if dialog.exec_() == QFileDialog.Accepted:
            return dialog.selectedFiles()[0]
   
    def save_dicom(self):
        
        # Create new patient directory
        self.new_pat_dir = self.save_dir()
        
        # Rescale the CT scan to a common resolution        
        self.SavingWindow = LoadingWindow('Saving structures...')
        self.SavingWindow.show()
        
        # 1 - create Worker and Thread inside the Form
        self.obj = SaveDicom(self)  # no parent!
        self.thread = QThread()  # no parent!
        # 3 - Move the Worker object to the Thread object
        self.obj.moveToThread(self.thread)
        # 4 - Connect Worker Signals to the Thread slots
        self.obj.finished.connect(self.thread.quit)
        # 5 - Connect Thread started signal to Worker operational slot method
        self.thread.started.connect(self.obj.run)
        # * - Thread finished signal will close the app if you want!
        self.thread.finished.connect(self.finished_saving)
        # 6 - Start the thread
        self.thread.start()
        
    def finished_saving(self):
        self.SavingWindow.close()
        self.ui.savenameLabel.setText('Scan saved as %s' % self.new_pat_dir)
        
def change_UID(UID, new_id):
    new_uid = UID.split('.')
    new_uid[10] = new_id
    new_uid = '.'.join(new_uid)
    return new_uid
        
def save_CTdicom(old_pat_dir, new_pat_dir, ct_scan, orig_spacing, new_spacing,
                 new_ROI_Names, new_ctr_indices, orig_locs, new_locs):
    # Copy old dicom files over to new dir. These will be edited
    _ = copy_tree(old_pat_dir, new_pat_dir)
    
    #print('Rescaling scan...')
    #ct_scan, _ = resize_image(ct_scan, new_spacing, orig_spacing)
    
    print(ct_scan.shape)
    # Collect CT and RT locations
    dicoms = glob.glob(new_pat_dir+'/*')
    for dicom in dicoms:
        if os.path.isdir(dicom):
            ct_dir = dicom
        else:
            rt_file = dicom
            
    # Create random sequence of 10 integers to be used in new UIDs
    new_id = ''.join(np.random.randint(0, 10, 10).astype(str))
            
    # Save RT info
    
    # Load existing file
    rt_dcmfile = pydicom.read_file(rt_file, force=True)
    
    # Change UIDs
    sop_instances = []
    rt_dcmfile.SOPInstanceUID = change_UID(rt_dcmfile.SOPInstanceUID, new_id)
    rt_dcmfile.StudyInstanceUID = change_UID(rt_dcmfile.StudyInstanceUID , new_id) 
    rt_dcmfile.SeriesInstanceUID = change_UID(rt_dcmfile.SeriesInstanceUID, new_id)
    for file in rt_dcmfile.ReferencedFrameOfReferenceSequence:
        file.FrameOfReferenceUID = change_UID(file.FrameOfReferenceUID, new_id)
        for sub_file in file.RTReferencedStudySequence:
            sub_file.ReferencedSOPInstanceUID = change_UID(sub_file.ReferencedSOPInstanceUID, new_id)
            for subsub_file in sub_file.RTReferencedSeriesSequence:
                subsub_file.SeriesInstanceUID = change_UID(subsub_file.SeriesInstanceUID, new_id)
                for subsubsub_file in subsub_file.ContourImageSequence:
                    subsubsub_file.ReferencedSOPInstanceUID = change_UID(subsubsub_file.ReferencedSOPInstanceUID, new_id)
    for file in rt_dcmfile.StructureSetROISequence:
        file.ReferencedFrameOfReferenceUID = change_UID(file.ReferencedFrameOfReferenceUID, new_id)
    for file in rt_dcmfile.ROIContourSequence:
        try:
            for sub_file in file.ContourSequence:
                for subsub_file in sub_file.ContourImageSequence:
                    subsub_file.ReferencedSOPInstanceUID  = change_UID(subsub_file.ReferencedSOPInstanceUID, new_id)
                    sop_instances.append(subsub_file.ReferencedSOPInstanceUID)
        except AttributeError:
            continue
        try:
            for sub_file in file.ContourImageSequence:
                 sub_file.ReferencedSOPInstanceUID  = change_UID(sub_file.ReferencedSOPInstanceUID, new_id)
        except AttributeError:
            continue
            
    # Calculate max SOP Instance
    max_sop = 0
    for sop in sop_instances:
        if int(sop.split('.')[-2]) > max_sop:
            max_sop = int(sop.split('.')[-2])
    # SOP that we will change
    cur_sop_instance = sop_instances[0].split('.')
    
    for ctr_indices, name in zip(new_ctr_indices, new_ROI_Names):
        
        # Collect contour points
        xs = np.array(ctr_indices)[:,2]
        ys = np.array(ctr_indices)[:,1]
        zs = np.array(ctr_indices)[:,0]
        # Turn into pixel indices
        xs = np.rint(xs).astype(int)
        ys = np.rint(ys).astype(int)
        zs = np.rint(zs).astype(int)
        
        # Check existing ROI numbers
        all_ROI_nums = []
        for seq in rt_dcmfile.StructureSetROISequence:
            all_ROI_nums.append(seq.ROINumber)
        all_ROI_nums = np.array(all_ROI_nums).astype(int)
        # Create new ROI number
        new_num = (max(all_ROI_nums)+1).astype(str)
        # Create new dataset
        new_ds = pydicom.dataset.Dataset()
        # Save info
        new_ds.ROINumber = new_num
        new_ds.ReferencedFrameOfReferenceUID = rt_dcmfile.StructureSetROISequence[0].ReferencedFrameOfReferenceUID
        new_ds.ROIName = name
        new_ds.ROIGenerationAlgorithm = 'MANUAL'
        # Add new sequence to existing ROIs
        rt_dcmfile.StructureSetROISequence.append(new_ds)

        # Create new sequence
        new_ds = pydicom.dataset.Dataset()
        new_ds.ROIDisplayColor = ['128','255','0']
        new_ds.ReferencedROINumber = new_num

        # Save contour points
        ctr_pts = []
        for z in np.unique(zs):
            x = xs[zs==z]
            y = ys[zs==z]
            ctr_data = []
            for x_, y_ in zip(x,y):
                ctr_data.append(str(orig_locs[2][x_]))
                ctr_data.append(str(orig_locs[1][y_]))
                ctr_data.append(str(orig_locs[0][z]))
            row = pydicom.dataset.Dataset()
            row.ContourData = ctr_data
            row.NumberOfContourPoints = int(len(ctr_data)/3)
            row.ContourGeometricType = 'CLOSED_PLANAR'
            seq = pydicom.dataset.Dataset()
            seq.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
            # Create new SOP instance
            cur_sop_instance[-2] = str(max_sop+1)
            seq.ReferencedSOPInstanceUID = '.'.join(cur_sop_instance)
            max_sop+=1
            
            row.ContourImageSequence = pydicom.sequence.Sequence([seq])
            ctr_pts.append(row)
        new_ds.ContourSequence = pydicom.sequence.Sequence(ctr_pts)
        rt_dcmfile.ROIContourSequence.append(new_ds)
        # Save structure file
        rt_dcmfile.save_as(rt_file)
        
    # Save CT slices
    dcm_fns = np.sort(glob.glob(ct_dir+'/*'))
    for i, file in enumerate(dcm_fns):
        dcmfile = pydicom.read_file(file, force=True)
        
        # Change UIDs
        dcmfile.SOPInstanceUID = change_UID(dcmfile.SOPInstanceUID, new_id)
        dcmfile.StudyInstanceUID = change_UID(dcmfile.StudyInstanceUID, new_id)
        dcmfile.SeriesInstanceUID = change_UID(dcmfile.SeriesInstanceUID, new_id)
        dcmfile.FrameOfReferenceUID   = change_UID(dcmfile.FrameOfReferenceUID  , new_id)
        
        # Index into edited CT scan
        z_loc = float(dcmfile.ImagePositionPatient[2])
        cur_slice = ct_scan[np.where(orig_locs[0]==z_loc)[0]]
        # Rescale back
        dcmfile.PixelData = ((cur_slice - dcmfile.RescaleIntercept) / dcmfile.RescaleSlope).astype('uint16')[0].tobytes()
        # Save slice
        dcmfile.save_as(file)

class SaveDicom(QObject):
    def __init__(self, widget):
        super(SaveDicom, self).__init__()
        self.widget = widget

    finished = pyqtSignal()
    @pyqtSlot()
    def run(self):
        # Rescale the CT scan to a common resolution
        print('Saving scan as dicom...')
        save_CTdicom(self.widget.dicom_dir, self.widget.new_pat_dir, self.widget.ct_scan, 
                     self.widget.orig_spacing, self.widget.new_spacing, self.widget.new_ROI_Names, 
                     self.widget.new_ctr_indices, self.widget.orig_locs, self.widget.new_locs)
        print("Finished saving.")
        self.finished.emit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = StartWindow()
    w.show()
    sys.exit(app.exec_())
