#!/usr/bin/env python
#%%
#==============================================================================
# TRACTRAC -Masive Object Tracking Software
#==============================================================================
#usage: tractrac.py [-h] [-f FILE] [-tf TFILE] [-mmf MOTIONMODELFILE] [-a]
#                   [-o OUTPUT] [-opp] [-s] [-p PLOT] [-sp] [-par PARALLEL]
#
#TRACTRAC - Joris Heyman
#
#optional arguments:
#  -h, --help            show this help message and exit
#  -f FILE, --file FILE  Video Filename to track
#  -tf TFILE, --tfile TFILE
#                        Time of frame file
#  -mmf MOTIONMODELFILE, --motionmodelfile MOTIONMODELFILE
#                        Motion Model file
#  -a, --averages        Save average velocity maps
#  -o OUTPUT, --output OUTPUT
#                        Save tracking results in a file ASCII (1) or HDF5 (2)
#  -opp, --outputpp      Save Post Processing results in a file
#  -s, --silent          No tracking infos
#  -sp, --saveplot       Save plots in image sequence
#  -par PARALLEL, --parallel PARALLEL
#                        Visualization in a Parallel Thread

#==============================================================================
# # Get Help :
# # >> python tractrac.py --help
#==============================================================================


#==============================================================================
# # Example runs
# # Default video
# # >> python tractrac.py -a
# # >> python tractrac.py -p 1
# # WebCam
# # >> python tractrac.py -f '0' -p 1 -cmax=50
# # Other video file
# # >> python tractrac.py -f '../Sample_videos/videotest.avi' -p 1
# # Other image sequence 
# # >> python tractrac.py -f '../Sample_videos/PIVChallenge/*.tif' -a -p 2
# # >> python tractrac.py -f '../Sample_videos/RiverDrone/*.tif' -a -o 1-p 2
# # Type 
#==============================================================================
#=============================================
global version
version= '3.0 (22/05/2021) | J. Heyman'
#==============================================================================
#v3.0 __________________________________
# Graphical and live change of parameters (in _par.txt files)
#v2.0 __________________________________
# Fast Nearest Neighboor Search Integration (via scipy.spatial.DTree). No more paralelization windows are needed.

#version= '1.5 (03/01/2017)'
#%matplotlib auto
# Notes of versions
# v1.5 _________________________________________________________________________________
# Possibility to execute tractrac both as a module (import tractrac) or directly in bash (python tractrac.py)
# Integration of non-constant times step
# Possibility to read an "interpolated" motion model image of U and V
# Parameter file now called *_par.txt
# Fixed some difference between Matlab and Python version, some diff remain notaob_detectiony in the error thresholds

import time
import glob
import numpy as np
import numpy.matlib
import scipy.spatial.distance
import scipy.spatial as scp
import cv2
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
#import pdb
from parse import *
import sys
import os,os.path
import multiprocessing as mp
#from matplotlib.patches import Circle
#from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
#import imutils # To check opencv versions
from scipy.interpolate import griddata
import os,os.path
import argparse
import h5py # For saving binary format
#from tractrac_toolbox import *
#def run(**kwargs):
#% Tractrac Toolbox
import time
import argparse

import numpy as np
np.bool = np.bool_

from PyQt5 import QtWidgets, QtCore

from vispy.scene import SceneCanvas, visuals
from vispy.app import use_app

from scipy.fft import fft, ifft,fftfreq

from scipy.ndimage import gaussian_filter

from tractrac_lib import * 


# Define the parser
parser = argparse.ArgumentParser(description='TRACTRAC v'+version+' - Joris Heyman')
parser.add_argument('-f','--file', type=str, help='Video Filename to track',default='../Sample_videos/videotest.avi')
parser.add_argument('-tf','--tfile', type=str, help='Time of frame file',default='')
parser.add_argument('-mmf','--motionmodelfile', type=str, help='Motion Model file',default='')
parser.add_argument('-a','--averages', help='Save average velocity maps', action='store_true',default=False)
parser.add_argument('-o','--output', type=int, help='Save tracking results in a file ASCII (1) or HDF5 (2)',default=0)
parser.add_argument('-opp','--outputpp', help='Save Post Processing results in a file', action='store_true',default=False)
parser.add_argument('-s','--silent',help='No tracking infos', action='store_false',default=True)
parser.add_argument('-sp','--saveplot', help='Save plots in image sequence', action='store_true')


args = parser.parse_args()
filename=args.file
tfile=args.tfile
mmfilename=args.motionmodelfile
AVERAGES=args.averages
OUTPUT=args.output
OUTPUT_PP=args.outputpp
INFO=args.silent
SAVE_PLOT=args.saveplot

PARAMETERS=["vid_loop","ROIxmin",'ROIymin','ROIxmax','ROIymax','BG','BGspeed','noise','noise_size','peak_th_auto','peak_th'
,'peak_neigh','peak_conv','peak_conv_size','peak_minima','peak_subpix','motion','motion_av','motion_it','filter'
,'filter_std','motion','filter_time','plot','plot_image','plot_data','plot_data_type','plot_alpha','rescale']


# Read Video Stream or image sequence
flag_im=is_image(filename)
flag_web=0 # flag if videosource is webcam

# Check if projective transform file exist in folder
path,name=os.path.split(filename)

# Read Parameters or set to defaut values
if len(path)==0:
	path='./'
if flag_im:
	parameter_filename=path+'/' + name[-3:]+'seq_par.txt' # If list of image, default name different
else:
	parameter_filename=path+'/' + name[:-4]+'_par.txt'



version=' v3.0 GUI 06/2024'

COLORMAP_CHOICES = ["viridis", 'binary', 'gist_gray', 'plasma', 'inferno', 'magma', 'cividis',"reds", "blues"]
SIMULATION_CHOICES = ["Gradient","Decay","Source"]
IMAGE_CHOICES = ["Scalar","Vx","Vy","Vnorm"]

CONVOLUTION= ["None","Diff of Gaussian","Log of Gaussian"]
SUBPIX= ["Quadratic","Gaussian"]
IMAGE=["None","Raw","Convoluted","Background"]
DATA=["None","Velocity","Motion model","Model Error"]


# Read Projective transform file if it exist
if os.path.isfile(path+'/projection.txt'):
	proj=np.loadtxt(path+'/projection.txt')
else:
	proj=np.array([])


if flag_im: # Image list
	flist=sorted(glob.glob(filename))
	#pdb.set_trace()
	I0=imProj(cv2.imread(flist[0],2),proj)
	nFrames=len(flist)
	height,width=I0.shape[:2]
elif (filename=='0')|(filename=='1'): # WebCam
	flag_web=1
	cv2.destroyAllWindows()
	cap = cv2.VideoCapture(int(filename))
	nFrames=1000
	I0=cap.read()[1]
	I0=I0[::-1,:]
	I0=imProj(I0,proj)
	height,width=I0.shape[:2]
else:	# Video
	cv2.destroyAllWindows()
	cap = cv2.VideoCapture(filename)
	I0=cap.read()[1]
	I0=imProj(I0,proj)
	height,width=I0.shape[:2]
# 	if imutils.is_cv2():
# 		cap.set(cv2.cv.CAP_PROP_POS_FRAMES,0) # Rewind
# 		nFrames=int(cap.get(cv2.cv.CAP_PROP_FRAME_COUNT))
# 	elif imutils.is_cv3() or imutils.is_cv4() :
	cap.set(cv2.CAP_PROP_POS_FRAMES,0)# Rewind
	nFrames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# 	else:
# 		print('Bad OpenCV version. Please install opencv3')
# 		sys.exit()

if nFrames==0:
	print('No Frames to track. Check filename.')
	sys.exit()

# Read time file to get time step
DTFRAMES=np.ones(nFrames-1)
if os.path.isfile(tfile):
	Time_stamp=np.loadtxt(tfile)
	if Time_stamp.shape[0]==nFrames:
		DTFRAMES=np.diff(Time_stamp)

IMAGE_SHAPE = (height, width)  # (height, width)
CANVAS_SIZE = (width*3,height*4)  # (width, height)

class Controls(QtWidgets.QWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		
		
		self.set_default_parameter()
		# if exist, read parameter file
		self.read_parameter_file()
		# if exist, write parameter file with replaced default values
		self.write_parameter_file()
		
		self.layout = QtWidgets.QHBoxLayout()
		self.layout0 = QtWidgets.QVBoxLayout()
		self.layout1 = QtWidgets.QVBoxLayout()
		#self.layout2 = QtWidgets.QVBoxLayout()
		
		self.layout.addLayout(self.layout0 )
		self.layout.addLayout(self.layout1 )
		#self.layout.addLayout(self.layout2 )
		
		
		self.add_label("title",title=r"___ TracTrac GUI___\n")
		
		self.add_radio("start",title="Start Tracking",checked=0)
		
		self.add_button("save",title="Save Parameters")
# 		self.quit_bt = QtWidgets.QPushButton('Quit', self)
# 		layout.addWidget(self.quit_bt)
		
# 		self.pause_chooser = QtWidgets.QComboBox()
# 		self.pause_chooser.addItems(SIMULATION_CHOICES)
# 		layout.addWidget(self.pause_chooser)
		# RoixMin max : sliders
		self.add_label("imProc",title="___ Image Processing ___",layout='0')
		self.add_slider("ROIxmin",mini=0,maxi=100,inte=10,value=0,title="ROI xmin",layout='0')
		self.add_slider("ROIxmax",mini=0,maxi=100,inte=10,value=100,title="ROI xmax",layout='0')
		self.add_slider("ROIymin",mini=0,maxi=100,inte=10,value=0,title="ROI ymin",layout='0')
		self.add_slider("ROIymax",mini=0,maxi=100,inte=10,value=100,title="ROI ymax",layout='0')
		self.add_radio("BG",layout='0',title='Background subtraction',checked=self.BG)
		self.add_slider("BGspeed",mini=0,maxi=100,inte=10,value=2,title="Adaptation speed",layout='0')
		self.add_radio("noise",layout='0',title='Median noise filter',checked=self.noise)
		self.add_slider("noise_size",mini=1,maxi=100,inte=10,value=1,title="Median filter size",layout='0')
		self.add_slider("vid_loop",mini=1,maxi=10,inte=0,value=1,title="Video loops",layout='0')

		self.add_label("detect",title="___ Object Detection ___",layout='1')
		self.add_radio("peak_minima",layout='1',title='Track minima',checked=self.peak_minima)
		self.add_chooser("peak_conv",CONVOLUTION,title='Convolution kernel',layout='1')
		self.add_slider("peak_conv_size",mini=0,maxi=400,inte=40,value=22,title="Kernel size",layout='1')
		self.add_radio("peak_th_auto",layout='1',title='Automatic thresholding',checked=self.peak_th_auto)
		self.add_slider("peak_th",mini=-100,maxi=100,inte=20,value=2,title="Threshold value",layout='1')
		self.add_slider("peak_neigh",mini=1,maxi=20,inte=1,value=1,title="Min distance",layout='1')
		self.add_chooser("peak_subpix",SUBPIX,layout='1',title='Subpixel method')

		self.add_label("track",title="___ Object Tracking ___",layout='1')
		self.add_radio("motion",title="Motion model",layout='1',checked=self.motion)
		self.add_slider("motion_av",mini=1,maxi=100,inte=10,value=5,title="Spatial average",layout='1')
		self.add_slider("motion_it",mini=1,maxi=20,inte=5,value=1,title="Iterations",layout='1')
		self.add_radio("filter",title="Filter outliers",layout='1',checked=self.filter)
		self.add_slider("filter_std",mini=0,maxi=40,inte=1,value=20,title="Outlier threshold",layout='1')
		self.add_slider("filter_time",mini=0,maxi=100,inte=10,value=1,title="Threshold frame correlation",layout='1')

		self.add_label("pllabel",title="___ Plot and output ___",layout='0')
#		self.add_radio("plot",title="Plot",layout='0')
		self.add_chooser("plot_image",IMAGE,layout='0',title='Image type')
		self.add_chooser("plot_data",DATA,layout='0',title='Data type')
#		self.add_slider("plot_alpha",mini=0,maxi=10,inte=1,value=10,title="Transparency",layout='0')
		self.add_slider("rescale",mini=0,maxi=100,inte=10,value=10,title="Arrow Scale",layout='0')
		
		self.layout.addStretch(1)
		self.setLayout(self.layout)
	
	def read_parameter_file(self):
		if os.path.exists(parameter_filename):
			with open(parameter_filename) as f:
				for line in f:
					s=search('{varname:w} {varvalue:g}',line)
					if (s != None): 
						exec("%s = %d" % ('self.'+str(s['varname']),s['varvalue']))
			#	print('Parameter file read !')
		else:
			print('WARNING: no parameter file exists. Taking default values! ')

	def write_parameter_file(self):
		f = open(parameter_filename, 'w')
		f.write('# Parameter file generated by TracTrac Python v'+ version+'\n\n')
		f.write('# Video loops \n')
		f.write('vid_loop {} \t\t# Number of loops over frames to train motion model\n\n'.format(self.vid_loop))
		f.write('# Image Processing\n')
		f.write('ROIxmin {} \t\t# Region of interest for tracking (xmin) \n'.format(self.ROIxmin))
		f.write('ROIymin {} \t\t# Region of interest for tracking (ymin) \n'.format(self.ROIymin))
		f.write('ROIxmax {} \t\t# Region of interest for tracking (xmax) \n'.format(self.ROIxmax))
		f.write('ROIymax {} \t\t# Region of interest for tracking (ymax) \n'.format(self.ROIymax))
		f.write('BG {} \t\t# (0 or 1, default 0) Use background subtraction to remove static regions \n'.format(self.BG))
		f.write('BGspeed {} \t\t# (0 to 1, default 0.01) adaptation speed of background \n'.format(self.BGspeed))
		f.write('noise {} \t\t# (0 or 1, default 0) Use median filtering to remove noise \n'.format(self.noise))
		f.write('noise_size {} \t\t# (3 or 11, default 3) Size of median kernel \n\n'.format(self.noise_size))
		f.write('# Object Detection\n')
		f.write('peak_th_auto {}\t# (0 or 1) Automatic object detection threshold \n'.format(self.peak_th_auto))
		f.write('peak_th {} \t\t# (-inf to +inf) Manual object detection threshold \n'.format(self.peak_th))
		f.write('peak_neigh {} \t\t# (1 to inf, default 1) Minimum distance between object \n'.format(self.peak_neigh))
		f.write('peak_conv {} \t\t# Object detection kernel: 0 (none), 1 (Dif. of Gaussian), 2 (Log of Gaussian), default 1. \n'.format(self.peak_conv))
		f.write('peak_conv_size {} \t# (0 to +inf) Object typical size (pixels) \n'.format(self.peak_conv_size))
		f.write('peak_subpix {} \t# Subpixel precision method : 0 (quadratic), 1 (gaussian), default 1\n'.format(self.peak_subpix))
		f.write('peak_minima {} \t# (0 or 1) : Track dark objects (instead of bright), default 0\n\n'.format(self.peak_minima))
		f.write('# Motion Model \n')
		f.write('motion {} \t\t# (0 or 1) : Use motion model to predict object displacements, default 1\n'.format(self.motion))
		f.write('motion_av {} \t\t# (1 to +inf) : Number of neighboors over which model is averaged, default 5\n'.format(self.motion_av))
		f.write('motion_it {} \t\t# (0 to +inf) : Iterations of motion model \n'.format(self.motion_it))
		f.write('filter {} \t\t# (0 or 1) : Enable filtering of outliers \n'.format(self.filter))
		f.write('filter_std {} \t# (-inf to +inf, default 1.5) : Threshold for outliers filtering (in nb of standard deviation of motion model error) \n'.format(self.filter_std))
		f.write('filter_time {} \t# (0 to +inf, default 1) : Time speed of adaptation of outlier threshold \n\n'.format(self.filter_time))
		f.write('# Plotting \n')
		f.write('plot {} \t\t# (0 or 1) : Plot tracking process \n'.format(self.plot))
		f.write('plot_image {} \t# Plot Raw (0), Convoluted (1) or Background (2) image \n'.format(self.plot_image))
		f.write('plot_data {} \t\t# Color by velocity magnitude (0), motion model velocity (1), motion model error (2) \n'.format(self.plot_data))
		f.write('plot_data_type {} \t# Vectors (1) or scatter (2) plot \n'.format(self.plot_data))
		f.write('plot_alpha {} \t# (0,1) Transparency \n'.format(self.plot_alpha))
		f.close()
	

	def set_default_parameter(self):
		# Sine flow parameters
		self.vid_loop = 0
		self.ROIxmin = 0
		self.ROIymin= 0
		self.ROIxmax = width
		self.ROIymax=height
		self.BG=1
		self.BGspeed=0.05
		self.noise=1
		self.noise_size=20
		self.peak_th_auto=60
		self.peak_th=0.02
		self.peak_neigh=1
		self.peak_conv=1
		self.peak_conv_size=2.2
		self.peak_minima=0
		self.peak_subpix=1
		self.motion=1
		self.motion_av=3
		self.motion_it=1
		self.filter=1
		self.filter_std=2
		self.motion=1
		self.filter_time=1
		self.plot=1
		self.plot_image=0
		self.plot_data_type=1
		self.rescale=0.1
	
	def add_label(self,variable,title='label',layout='0'):
		print('self.'+variable+' =  QtWidgets.QLabel("'+title+'")')
		exec('self.'+variable+' =  QtWidgets.QLabel("'+title+'")')
		exec('self.layout'+layout+'.addWidget(self.'+variable+')')

	def add_slider(self,variable,mini=0,maxi=100,inte=1,value=50,title="Title",layout='0'):
#		print('self.'+variable+'_label = QtWidgets.QLabel("'+title+': '+fmt+'".format(self.'+variable+'))')
#		exec('self.'+variable+'_label = QtWidgets.QLabel("'+title+': '+fmt+'".format(self.'+variable+'))')
		exec('self.'+variable+'_label = QtWidgets.QLabel("'+title+'")')
		exec('self.layout'+layout+'.addWidget(self.'+variable+'_label)')
		exec('self.'+variable+'_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)')
		exec('self.'+variable+'_sl.setMinimum('+"{:d}".format(mini)+')')
		exec('self.'+variable+'_sl.setMaximum('+"{:d}".format(maxi)+')')
		exec('self.'+variable+'_sl.setValue('+"{:d}".format(value)+')')
		exec('self.'+variable+'_sl.setTickPosition(QtWidgets.QSlider.TicksAbove)')
		exec('self.'+variable+'_sl.setTickInterval('+"{:d}".format(inte)+')')
		exec('self.layout'+layout+'.addWidget(self.'+variable+'_sl)')

	def add_chooser(self,variable,CHOICES,title="title",layout='0'):
		exec('self.'+variable+'_label = QtWidgets.QLabel("'+title+'")')
		exec('self.layout'+layout+'.addWidget(self.'+variable+'_label)')
		exec('self.'+variable+'_chooser = QtWidgets.QComboBox()')
		exec('self.'+variable+'_chooser.addItems(CHOICES)')
		exec('self.layout'+layout+'.addWidget(self.'+variable+'_chooser)')
	
	def add_radio(self,variable,title="title",layout='0',checked=1):
		print('self.'+variable+'_radio = QtWidgets.QPushButton("'+title+'")')
		exec('self.'+variable+'_radio = QtWidgets.QPushButton("'+title+'")')
		exec('self.'+variable+'_radio.setChecked(False)')
		exec('self.'+variable+'_radio.setCheckable(True)')
		if checked:
			exec('self.'+variable+'_radio.setChecked(True)')
		exec('self.layout'+layout+'.addWidget(self.'+variable+'_radio)')

	def add_button(self,variable,title="button",layout='0'):
		print('self.'+variable+'_bt = QtWidgets.QPushButton("'+title+'", self)')
		exec('self.'+variable+'_bt = QtWidgets.QPushButton("'+title+'", self)')
		exec('self.layout'+layout+'.addWidget(self.'+variable+'_bt)')
		
	def set_ROIxmin(self,r):
		self.ROIxmin=int(np.minimum(int(width*r/100),self.ROIxmax-10))
		self.ROIxmin_label.setText("ROI xmin: {:d}".format(self.ROIxmin))

	def set_ROIxmax(self,r):
		self.ROIxmax=int(np.maximum(int(width*r/100),self.ROIxmin+10))
		self.ROIxmax_label.setText("ROI xmmax: {:d}".format(self.ROIxmax))

	def set_ROIymin(self,r):
		self.ROIymin=int(np.minimum(int(height*r/100),self.ROIymax-10))
		self.ROIymin_label.setText("ROI ymin: {:d}".format(self.ROIymin))

	def set_ROIymax(self,r):
		self.ROIymax=int(np.maximum(int(height*r/100),self.ROIymin+10))
		self.ROIymax_label.setText("ROI xmin: {:d}".format(self.ROIymax))

	def set_BGspeed(self,r):
		self.BGspeed=r/100.
		self.BGspeed_label.setText("Adaptation speed: {:1.2}".format(self.BGspeed))

	def set_noise_size(self,r):
		self.noise_size=r
		self.noise_size_label.setText("Median Filter size: {:d}".format(self.noise_size))

	def set_peak_conv_size(self,r):
		self.peak_conv_size=r/10.
		self.peak_conv_size_label.setText("Convolution Filter size: {:1.1f}".format(self.peak_conv_size))

	def set_peak_th(self,r):
		self.peak_th=r/100.
		self.peak_th_label.setText("Threshold value: {:1.2f}".format(self.peak_th))

	def set_peak_neigh(self,r):
		self.peak_neigh=r
		self.peak_neigh_label.setText("Min distance: {:d}".format(self.peak_neigh))

	def set_motion_av(self,r):
		self.motion_av=r
		self.motion_av_label.setText("Spatial Average: {:d}".format(self.motion_av))

	def set_rescale(self,r):
		self.rescale=r/10.
		self.rescale_label.setText("Arrow scale: {:1.1f}".format(self.rescale))
		
	def set_motion_it(self,r):
		self.motion_it=r
		self.motion_it_label.setText("Model Iterations: {:d}".format(self.motion_it))

	def set_filter_std(self,r):
		self.filter_std=r/10.
		self.filter_std_label.setText(r"Outlier threshold: {:1.1f}".format(self.filter_std))

	def set_filter_time(self,r):
		self.filter_time=r
		self.filter_time_label.setText("Threshold lag time: {:d}".format(self.filter_time))

	def set_plot_alpha(self,r):
		self.plot_alpha=r/10.
		self.plot_alpha_label.setText("Transparency: {:1.1f}".format(self.plot_alpha))

	def set_vid_loop(self,r):
		self.vid_loop=r
		self.vid_loop_label.setText("Video Loops: {:d}".format(self.vid_loop))

	def set_peak_th_auto(self):
		if self.peak_th_auto_radio.isChecked():
			self.peak_th_auto=1
		else:
			self.peak_th_auto=0

	def set_peak_minima(self):
		if self.peak_minima_radio.isChecked():
			self.peak_minima=1
		else:
			self.peak_minima=0
			

	def set_motion(self):
		if self.motion_radio.isChecked():
			self.motion=1
			print(1)
		else:
			self.motion=0
			print(0)

	def set_filter(self):
		if self.filter_radio.isChecked():
			self.filter=1
		else:
			self.filter=0
		
	def set_plot_image(self, _mode: str):
		if _mode=="None":
			self.plot_image = 0
		if _mode=="Raw":
			self.plot_image = 1
		if _mode=="Convoluted":
			self.plot_image = 2
		if _mode=="Background":
			self.plot_image = 3
		print(_mode,self.plot_image)
		

		
	def set_plot_data(self, _mode: str):
		if _mode=="None":
			self.plot_data = 0
		if _mode=="Velocity":
			self.plot_data = 1
		if _mode=="Motion Model":
			self.plot_data = 2
		if _mode=="Model Error":
			self.plot_data = 3
		#print(_mode,self.plot_image)


	def set_peak_conv(self, _mode: str):
		if _mode=="None":
			self.peak_conv = 0
		if _mode=="Log of Gaussian":
			self.peak_conv = 2
		if _mode=="Diff of Gaussian":
			self.peak_conv = 1

	def set_peak_subpix(self, _mode: str):
		if _mode=="Quadratic":
			self.peak_subpix = 1
		if _mode=="Gaussian":
			self.peak_subpix = 2
			
class CanvasWrapper:
	def __init__(self):
		
		self.canvas = SceneCanvas(size=CANVAS_SIZE)
#		self.grid = self.canvas.central_widget.add_grid()
	
		self.view_top = self.canvas.central_widget.add_view()

#		self.view_top = self.grid.add_view(0, 0, bgcolor='cyan')
		#image_data = np.zeros((h,w))
		image_data = I0
		
		self.image = visuals.Image(
			image_data,
			texture_format="auto",
			clim=[-1,1],
			cmap=COLORMAP_CHOICES[0],
			parent=self.view_top.scene,
			interpolation='bilinear'
		)
		
		vel=np.array([[0,0,-1],[width,height,-1]]).reshape(-1,3)
		vm=0.5
		col=plt.cm.jet(vm)
		self.arrows= visuals.Arrow(pos=vel,arrows=vel.reshape(-1,6),
								color=col,
								parent=self.view_top.scene,
								width=5, connect='segments', method='gl', antialias=True,
								arrow_type='triangle_30', arrow_size=5)
		
		#self.view_top.camera.PanZoomCamera(parent=self.view_top.scene, aspect=1, name='panzoom')
		self.view_top.camera = "panzoom"
		#self.view_top.camera = cameras.base_camera.BaseCamera(aspect=1,interactive=False)
		self.view_top.camera.set_range(x=(0, IMAGE_SHAPE[1]), y=(0, IMAGE_SHAPE[0]), margin=0)
		self.view_top.camera.interactive=False
		

# 		self.view_bot = self.grid.add_view(1, 0, bgcolor='#c0c0c0')
# 		line_data = _generate_random_line_positions(NUM_LINE_POINTS)
# 		self.line = visuals.Line(line_data, parent=self.view_bot.scene, color=LINE_COLOR_CHOICES[0])
# 		self.view_bot.camera = "panzoom"
# 		self.view_bot.camera.set_range(x=(0, NUM_LINE_POINTS), y=(0, 1))

# 	
# 	def set_image_colormap(self, cmap_name: str):
# 		print(f"Changing image colormap to {cmap_name}")
# 		self.image.cmap = cmap_name

# 		
# 	def set_line_color(self, color):
# 		print(f"Changing line color to {color}")
# 		self.line.set_data(color=color)

	def update_data(self, new_data_dict):
		#print("Updating data...")
#		self.line.set_data(new_data_dict["line"])
		self.image.set_data(new_data_dict["image"])
		self.arrows.set_data(pos=new_data_dict["vel"],arrows=new_data_dict["vel"].reshape(-1,6),
											 color=new_data_dict["color"])
		self.canvas.update()

class MyMainWindow(QtWidgets.QMainWindow):
	closing = QtCore.pyqtSignal()

	def __init__(self, canvas_wrapper: CanvasWrapper, *args, **kwargs):
		super().__init__(*args, **kwargs)

		central_widget = QtWidgets.QWidget()
		main_layout = QtWidgets.QHBoxLayout()

		self._controls = Controls()
		main_layout.addWidget(self._controls)
		self._canvas_wrapper = canvas_wrapper
		main_layout.addWidget(self._canvas_wrapper.canvas.native)

		central_widget.setLayout(main_layout)
		self.setCentralWidget(central_widget)

		self._connect_controls()

	def _connect_controls(self):
		print("connecting")
# 		self._controls.mode_chooser.currentTextChanged.connect(self._controls.set_mode)
# 		self._controls.colormap_chooser.currentTextChanged.connect(self._canvas_wrapper.set_image_colormap)
# 		self._controls.imtype_chooser.currentTextChanged.connect(self._controls.set_imtype)
# #		self._controls.line_color_chooser.currentTextChanged.connect(self._canvas_wrapper.set_line_color)
# 		self._controls.power_sl.valueChanged.connect(self._controls.set_roughness)
# 		self._controls.D_sl.valueChanged.connect(self._controls.set_diffusion)
# 		self._controls.a_sl.valueChanged.connect(self._controls.set_amplitude)
# 		self._controls.tcorr_sl.valueChanged.connect(self._controls.set_tcorr)
# 		self._controls.lmin_sl.valueChanged.connect(self._controls.set_lmin)
# 		self._controls.lmax_sl.valueChanged.connect(self._controls.set_lmax)
# 		self._controls.fps_sl.valueChanged.connect(self._controls.set_fps)
		self._controls.save_bt.clicked.connect(self._controls.write_parameter_file)
# 		#self._controls.quit_bt.clicked.connect(self._controls.set_quit)
# 		self._controls.rescale_bt.clicked.connect(self.set_rescale)
		self._controls.ROIxmin_sl.valueChanged.connect(self._controls.set_ROIxmin)
		self._controls.ROIxmax_sl.valueChanged.connect(self._controls.set_ROIxmax)
		self._controls.ROIymin_sl.valueChanged.connect(self._controls.set_ROIymin)
		self._controls.ROIymax_sl.valueChanged.connect(self._controls.set_ROIymax)
		self._controls.BGspeed_sl.valueChanged.connect(self._controls.set_BGspeed)
		self._controls.noise_size_sl.valueChanged.connect(self._controls.set_noise_size)
		self._controls.peak_th_sl.valueChanged.connect(self._controls.set_peak_th)
		self._controls.peak_conv_size_sl.valueChanged.connect(self._controls.set_peak_conv_size)
		self._controls.peak_neigh_sl.valueChanged.connect(self._controls.set_peak_neigh)
		self._controls.motion_av_sl.valueChanged.connect(self._controls.set_motion_av)
		self._controls.motion_it_sl.valueChanged.connect(self._controls.set_motion_it)
		self._controls.filter_std_sl.valueChanged.connect(self._controls.set_filter_std)
		self._controls.filter_time_sl.valueChanged.connect(self._controls.set_filter_time)
		self._controls.vid_loop_sl.valueChanged.connect(self._controls.set_vid_loop)
		self._controls.rescale_sl.valueChanged.connect(self._controls.set_rescale)
		# Choosers
		self._controls.plot_image_chooser.currentTextChanged.connect(self._controls.set_plot_image)
		self._controls.plot_data_chooser.currentTextChanged.connect(self._controls.set_plot_data)
		self._controls.peak_conv_chooser.currentTextChanged.connect(self._controls.set_peak_conv)
		
		self._controls.peak_th_auto_radio.clicked.connect(self._controls.set_peak_th_auto)
		self._controls.motion_radio.clicked.connect(self._controls.set_motion)
		self._controls.filter_radio.clicked.connect(self._controls.set_filter)
		self._controls.peak_minima_radio.clicked.connect(self._controls.set_peak_minima)

	def set_rescale(self):
		C=np.array(self._canvas_wrapper.image._data)
		self._canvas_wrapper.image.clim=[np.min(C),np.max(C)]
	
	def closeEvent(self, event):
		print("Closing main window!")
		self.closing.emit()
		return super().closeEvent(event)


class tractrac(QtCore.QObject):
	"""Object representing a complex data producer."""
	new_data = QtCore.pyqtSignal(dict)
	finished = QtCore.pyqtSignal()

	def __init__(self, myMainWindow: MyMainWindow, parent=None):
		super().__init__(parent)
		self._controls = Controls()
		self._image_data = np.zeros(IMAGE_SHAPE)
		#		self._line_data = _generate_random_line_positions(NUM_LINE_POINTS)
		self._myMainWindow = myMainWindow

	def self2dict(self): # convert GUI self. variable to dictionary for compatibility with tractrac
		th=[{}]
		for par in PARAMETERS:
			exec('th[0]["'+par+'"]=self._myMainWindow._controls.'+par)
		return th
	

	def stop_data(self):
		print("Data source is quitting...")
		self._should_end = True


	def run(self):
		# Rewind video
		cap.set(cv2.CAP_PROP_POS_FRAMES,0)# Rewind
		
		Pts=[]
		dt=DTFRAMES
		# Create dictionary of parameters
		th=self.self2dict()
		
		
		# Read 2 first frames and initialize
		if flag_web:
			time0=time.time()
	
		I0 = cv2.imread(flist[0],2) if flag_im else cap.read()[1];
		I0=imProj(I0,proj)
		I0f=imProc(I0,th)
		
			
		if flag_web:
			time1=time.time()
			dt[0]=time1-time0
			time0=time1
	
		I1 = cv2.imread(flist[1],2) if flag_im else cap.read()[1];
		I1=imProj(I1,proj)
		I1f=imProc(I1,th)
		
		# Read Mask file if any
		mask_file=path+'/' + name[:-4]+'_mask.tif'
		if os.path.isfile(mask_file):
			mask=cv2.imread(mask_file,2)
			mask=imProj(mask,proj)
			mask=imProc(mask,th)
		else:
			mask=np.ones(I1f.shape,dtype='float32')
	
		#print(mask_file,mask.shape)
		h,w=np.shape(I0f)
	
		#Initialize Background
		B=np.zeros((h,w),dtype=np.float32)
		if (th[0]['BG']==1)&(flag_web==0):
				nbgstart=np.minimum(100,nFrames)
				print('Initiating Background image over the firsts {:d} frames...'.format(nbgstart))
				for i in range(nbgstart):
					I = cv2.imread(flist[i],2) if flag_im else cap.read()[1];
					I=imProj(I,proj)
					I=imProc(I,th)
					B=B+I/nbgstart
			
			# Rewind video
				if (imutils.is_cv2()) & (not flag_im):
					cap.set(cv2.cv.CAP_PROP_POS_FRAMES,0);
					cap.read()
					cap.read()
				elif (imutils.is_cv3()) & (not flag_im):
					cap.set(cv2.CAP_PROP_POS_FRAMES,0);
					cap.read()
					cap.read()
						
		
		
		 # Initialize Foreground
		F0=I0f*mask-B
		F1=I1f*mask-B
	
	
		# get peaks and initialize centre, velocity and acc vectors
		# Feature to track detection
		F0f,x,y,z=feature_detection(F0,th)
		C0=np.transpose(np.array([x,y]))
		# Sort according to x
		#C0=C0[np.argsort(C0[:,0],axis=0),:]
		U0=np.zeros(C0.shape)
		A0=np.zeros(C0.shape)
		ID0=np.zeros(C0.shape[0])+np.nan # Vector of id traj
	
		F1f,x,y,z=feature_detection(F1,th)
		C1=np.transpose(np.array([x,y]))
		#C1=C1[np.argsort(C1[:,0],axis=0),:]
		U1=np.zeros(C1.shape)
		A1=np.zeros(C1.shape)
	
		# Replace threshold if auto and not shi and thomasi
		if (th[0]['peak_th_auto']==1)&~(th[0]['peak_conv']==3):
			th[0]['peak_th']=np.mean(F1f)+0.5*np.std(F1f)
	
	#	C1motion=C1
		errU_filt=np.zeros(C1.shape[0])
		id_traj=0
	
		# Initialize lists
	
		idgood=[]
		errU = np.array([])
		errU_th=2.
		Xmot, Umot,Xmot_temp,Umot_temp = [np.array([]).reshape((-1,2)) for j in range(4)]
		errmot, errmot_temp = [np.array([]) for j in range(2)]
		#X0,X1,X1motion,X2,um,Umotion,a= [np.array([]).reshape(0,2) for i in range(7)]
		# Vector of boundary known points
	#	BND=[]
	
		# Initialize Point Tree structure
		t0,t1=None,None
		if len(C0)>0: t0=scp.cKDTree(C0)
		if len(C1)>0: t1=scp.cKDTree(C1)
		ns01,ns10,i1001=[np.array([]) for i in range(3)]
		if (len(C0)>0)&(len(C1)>0):
			ns01=t0.query(C1)
			ns10=t1.query(C0)
			# First check of
			i1001=ns10[1][ns01[1]]==np.arange(C1.shape[0])
	
		# Check if MotionModel File existing
		MotionModelFile=False
		if os.path.exists(mmfilename):
			# Read File
			mmdata = np.loadtxt(mmfilename,delimiter=' ')
			# Interpolation on a grid made at the image size
			mm_x, mm_y = np.meshgrid( range(I0.shape[1]),range(I0.shape[0]))
			mm_U = griddata((mmdata[:,0],mmdata[:,1]), mmdata[:,2], (mm_x, mm_y), method='nearest')
			mm_V = griddata((mmdata[:,0],mmdata[:,1]), mmdata[:,3], (mm_x, mm_y), method='nearest')
			if mmdata.shape[1]>4:
				mm_E = griddata((mmdata[:,0],mmdata[:,1]), mmdata[:,4], (mm_x, mm_y), method='nearest')
			else:
				mm_E = griddata((mmdata[:,0],mmdata[:,1]), np.zeros(mmdata[:,1].shape), (mm_x, mm_y), method='nearest')
			MotionModelFile=True
			if INFO:
				print('! The Provided Motion Model File will be used !')
	# Initialize averages
				
		if AVERAGES:
			Av_U=np.zeros(F0f.shape)
			Av_V=np.zeros(F0f.shape)
			Av_N=np.zeros(F0f.shape)
	
	#====================Top Down Approach=========================================
		N=np.arange(nFrames)
		if ~flag_web:
			DT=dt
			for k in range(int(th[0]['vid_loop'])):
				if np.mod(k,2)==0:
					N=np.hstack((N,np.arange(nFrames-2,-1,-1))) # Vector of consecutive frames to read
					DT=np.hstack((DT,-dt[::-1])) # Vector of consecutive times
				else:
					N=np.hstack((N,np.arange(1,nFrames))) # Vector of consecutive frames to read
					DT=np.hstack((DT,dt[::1])) # Vector of consecutive times
			dt=DT
	#==============================================================================
		# MAIN LOOP OVER VIDEO FRAMES  #################################################
		#nFrames=5
		if INFO: print('0001 | Buffer frame...')
		for i in range(2,len(N)):
			# break if buttion start is unchecked
			#print(self._myMainWindow._controls.start_radio.isChecked())
			if not(self._myMainWindow._controls.start_radio.isChecked()):
				break
			
	#	for i in range(2,399):
			#print 'top'
			t = time.time()
			# Read parameter files to get live change of parameters
# 			th = read_parameter_file(parameter_filename)
# 			th = setup_parameters(th)
			# Create dictionary of parameters
			th=self.self2dict()
			
			
			# Replace threshold if auto and not shi and thomasi
			if (th[0]['peak_th_auto']==1)&~(th[0]['peak_conv']==3):
				th[0]['peak_th']=np.mean(F1f)+0.5*np.std(F1f)
			# PRE processing steps #################################################
			if flag_im:
				I2 = cv2.imread(flist[N[i]],2)
			elif flag_web:
				I2 = cap.read()[1];
				I2 = I2[::-1,:]
			else:
				cap.set(1,N[i])
				I2 = cap.read()[1];
	
			if I2 is None:
				print('WARNING: Video reader get None type. Exiting at frame {:d}.'.format(N[i]))
				break
	
			if flag_web:
				time1=time.time()
				dt[i-1]=time1-time0
				time0=time1
				print('Webcam Framerate: {:1.1f} fps'.format(1./dt[i-1]))
			
			I2=imProj(I2,proj)
			I2f=imProc(I2,th)
			F2 = I2f*mask-B
	
			F2f,x,y,z=feature_detection(F2,th)
			C2=np.transpose(np.array([x,y]))
			#C2=C2[np.argsort(C2[:,0],axis=0),:]
			U2=np.zeros(C0.shape)
			A2=np.zeros(C0.shape)
			# END PREPROCESSING  #################################################
	
			it=0
			while (it<=th[0]['motion_it']):
				# PREDICTION from motion model
				if (len(idgood)+Xmot.shape[0]>1)&(len(C1)>0)&(th[0]['motion']==1):
					if len(X1)==0:
						[Umotion,errU_filt,Xmot_temp,Umot_temp,errmot_temp]=Propagate_MotionModel_KdTree(C1,np.array([]).reshape(-1,2),np.array([]).reshape(-1,2),np.array([]),Xmot,Umot,errmot,th)
					else:
						[Umotion,errU_filt,Xmot_temp,Umot_temp,errmot_temp]=Propagate_MotionModel_KdTree(C1,X1[idgood,:],um[idgood,:],errU[idgood],Xmot,Umot,errmot,th)
				else:
					Umotion=np.zeros(C1.shape)
					errU_filt=np.zeros(C1.shape[0])
				# Start Iteration step #################################################
				#if len(C2)==0:
	
				# Initialize for each loop
				#X0,X1,X1motion,X2,um,Umotion,a,t2,ns12,i2112= (np.array([[],[]]).T for i in range(10))
				#idC2,errU,ID,ISgood,i_all,um= (np.array([]).T for i in range(6))
					# Build new trees
				if len(C1)>0: t1m=scp.cKDTree(C1+Umotion*dt[i-1]) # Points C1 moved by motion model # !!Check on original version
				if len(C2)>0:
					t2=scp.cKDTree(C2) # Points C2
				else:
					t2=None
				if (len(C1)>0)&(len(C2)>0):
					ns21=t2.query(C1+Umotion*dt[i-1])  # Nearest Neighboor link C1m -> C2
					ns12=t1m.query(C2) # Nearest Neighboor link C2 -> C1m
					i1221=ns12[1][ns21[1]]==np.arange(C1.shape[0])  # Check reversibility of link C2-> C1m [C1m -> C2 ]
					i2112=ns21[1][ns12[1]]==np.arange(C2.shape[0]) # For next iteration
				else:
					i1221=[];i2112=[];ns12=[];
				
				if (len(C0)>0)&(len(C1)>0)&(len(C2)>0):
					i_all=i1001&i1221 # Keep only Unilateral associations on 3 steps 0 -> 1 -> 2
					#print np.sum(i_all)/np.float(C1.shape[0])
					# Update Trajectories positions
					X0=C0[ns01[1][i_all],:]
					X1=C1[i_all,:]
					X2=C2[ns21[1][i_all],:]
					# Build traj ID
					ID=ID0[ns01[1][i_all]]
					# Velocities
					U0=(X1-X0)/dt[i-2]
					U1=(X2-X1)/dt[i-1]
					um=(U0+U1)/2.
					if (dt[i-1]+dt[i-2])==0:
						A=np.zeros(U0.shape)+np.nan
					else:
						A=(U1-U0)/(dt[i-1]+dt[i-2])*2.
					#print i_all.shape,Umotion.shape,X1.shape,np.unique(i_all).shape
					#print i_all
					# Error in Motion Model prediction
					#Umotion=(C1motion[i_all,:]-C1[i_all,:])/dt[n-1]
					#plt.quiver(C0[i_all,0],C0[i_all,1],Umotion[:,0],Umotion[:,1])
		# TO CHECK wether dt should appear or not in errU
					errU=np.maximum(-10.,np.log10(np.amax((np.abs((Umotion[i_all,:]-um)*dt[i-1])),1)))
					# if any value of error is nan, reinitialize it
					errU[np.isnan(errU)]=0
					# Filtering outliers
					ISgood=errU-errU_filt[i_all]<errU_th
					# Evolution of threshold
					errU_th=(th[0]['filter_time']*errU_th+th[0]['filter_std']*np.std(errU))/(th[0]['filter_time']+1.)
				else:
					i_all,ID,errU,ISgood=[[] for j in range(4)]
					X0,X1,X2,um,A=[np.array([]).reshape(-1,2) for j in range(5)]
					errU_th=2 # Reinitialize threshold
				# Filter Outliers if necessary
				if th[0]['filter']==1:
					idgood=np.where(ISgood==1)[0]
					idbad=np.where(ISgood==0)[0]
				elif len(X1)>0:
					idgood=np.arange(0,len(X1[:,1]))
					idbad=[];
				else:
					idgood=[]
					idbad=[]
	#					print id_traj
	#					break
				if np.isnan(errU_th): errU_th=2
				if len(errU)>0:
					infos= '     | Motion Model it %02d - %d pts - log Err. Av/Max %1.1f/%1.1f (px) ' % (it,len(idgood),np.mean(errU),np.mean(errU+errU_th))
				else:
					infos= '     | Motion Model it %02d - %d pts - log Err. Av/Max ??/%1.1f (px) ' % (it,0,errU_th)
				if INFO: print(infos)
	#			if (PLOT==2) & (th[0]['motion_it']>0):
	#				col = np.sqrt(Umotion[idgood,1]**2+Umotion[idgood,0]**2); vel=Umotion[idgood,:];
	#				if PAR:
	#					# the last argument send a stop signal to the worker
	#					q=[I2,X2[idgood,:],vel,col,n,np.mean(errU),n<nFrames-1]N
	#					q=[I2f,X2[idgood,:],vel,col,n,np.mean(errU)]
	#					plot(q)
				it+=1
			# END Iteration step #################################################
			
			#Replace Nan good ID by new values
			if len(ID)>0:
				idnan=np.where(np.isnan(ID[idgood]))[0]
				if len(idnan)>0: # Associate new ID
						ID[idgood[idnan]]=np.arange(0,len(idnan))+id_traj
						id_traj=id_traj+len(idnan)
			#if i==398: break
			# Keep track of last motion model used
			Xmot,Umot,errmot=Xmot_temp,Umot_temp,errmot_temp
			# Save ID for next iteration
			if len(C1)>0:
				ID0=np.zeros(C1.shape[0])+np.nan
				ID0[i_all]=ID
			
			# If last loop, reset ID to 0
			if (i==len(N)-nFrames) & ~(flag_web):
				ID0[i_all]=np.arange(int(sum(i_all)))
				id_traj=sum(i_all)
				A=A+np.nan
				
			# Save Frame ID Position Speed Acceleration
			if (OUTPUT>0) & (i>=len(N)-nFrames+2) :
				newPts=np.vstack(((np.zeros(len(idgood))+N[i]-np.sign(dt[i-1])).T,ID[idgood].T, X1[idgood,:].T, um[idgood,:].T,A[idgood,:].T, Umotion[idgood,:].T,errU[idgood].T)).T
				Pts.append(newPts) # We use list for dynamic resizing of arrays
	
			if AVERAGES & (i>=len(N)-nFrames+2) :
				xi=np.uint16(X1[idgood,:])
				Av_U[xi[:,1],xi[:,0]]=Av_U[xi[:,1],xi[:,0]]+um[idgood,0]
				Av_V[xi[:,1],xi[:,0]]=Av_V[xi[:,1],xi[:,0]]+um[idgood,1]
				Av_N[xi[:,1],xi[:,0]]=Av_N[xi[:,1],xi[:,0]]+1
				
				
		# Plotting
			
			PLOT_DATA=th[0]['plot_data']
			PLOT_TYPE=th[0]['plot_data_type']
			PLOT_IMAGE=th[0]['plot_image']
				
			if PLOT_IMAGE==0: Im=np.zeros(I1f.shape)
			if PLOT_IMAGE==1: Im=I1f
			if PLOT_IMAGE==2: Im=F2f
			if PLOT_IMAGE==3: Im=B
				
			Im=(Im-Im.min())/(Im.max()-Im.min()) # Normalisarion
# 				Cmin=th[0]['plot_cmin']
# 				Cmax=th[0]['plot_cmax']
			alpha=th[0]['plot_alpha']
			
			
			if PLOT_DATA==1:
				vel=-np.ones((len(idgood)*2,3))
				vel[::2,:2]=X1[idgood,:]
				vel[1::2,:2]=X1[idgood,:]+um[idgood,:]*th[0]['rescale']
				vm=np.sqrt(np.sum(um[idgood,:]**2,axis=1))
				col=np.ones((len(idgood)*2,4))
			if (PLOT_DATA==2):
				vel=-np.ones((X1.shape[0]*2,3))
				vel[::2,:2]=X1
				vel[1::2,:2]=X1+Umotion[i_all,:]*th[0]['rescale']
				vm=np.sqrt(np.sum(Umotion[i_all,:]**2,axis=1))
				col=np.ones((Xmot.shape[0]*2,4))
			if (PLOT_DATA==3):
				vel=-np.ones((X1.shape[0]*2,3))
				vel[::2,:2]=X1
				vel[1::2,:2]=X1+Umotion[i_all,:]*th[0]['rescale']
				vm=np.sqrt(np.sum(Umotion[i_all,:]**2,axis=1))
				col=np.ones((X1.shape[0]*2,4))
				vm=errU
			
			
			if len(vm)>0:
				vm=(vm-np.min(vm))/(np.max(vm)-np.min(vm))
				col[::2,:]=plt.cm.jet(vm)
				col[1::2,:]=plt.cm.jet(vm)
			else:
				col=np.array([]).reshape(-1,4)
				# statistics
# 				plot_time_stat[N[i],0]=C1.shape[0]
# 				plot_time_stat[N[i],1]=len(idgood)
# 				plot_time_stat[N[i],2]=np.exp(np.mean(errU))
# 				plot_time_stat[N[i],3]=np.nanmean(np.sqrt(np.sum(um[idgood,:]**2.,axis=1)))
# 				if PAR==0:
# 					if len(um)==0:
# 						col = []; vel= np.array([]);
# 					else:
# 						if PLOT_DATA==1: col = np.sqrt(um[idgood,1]**2+um[idgood,0]**2); vel=um[idgood,:];
# 						if PLOT_DATA==2: col = np.sqrt(Umotion[i_all,1]**2+Umotion[i_all,0]**2)[idgood]; vel=Umotion[i_all,:][idgood,:];
# 						if PLOT_DATA==3: col = 10**errU_filt[i_all][idgood]; vel=Umotion[i_all,:][idgood,:];
# 						if PLOT_TYPE==2: vel[:]=0;
# 					if len(X1)>0: q=[Im,X1[idgood,:],vel,col,N[i],np.mean(errU[idgood]),th,F2f,plot_time_stat,Cmin,Cmax,alpha]
# 					if len(X1)==0: q=[Im,[],vel,col,N[i],np.nan,th,F2f,plot_time_stat,Cmin,Cmax,alpha]
# 					plot(q)
# 				elif (queue.empty()) or (SAVE_PLOT==1): # Only plot when queue is empty or when the save order is given
# 					if len(um)==0:
# 						col = []; vel= np.array([]);
# 					else:
# 						if PLOT_DATA==1: col = np.sqrt(um[idgood,1]**2+um[idgood,0]**2); vel=um[idgood,:];
# 						if PLOT_DATA==2: col = np.sqrt(Umotion[i_all,1]**2+Umotion[i_all,0]**2)[idgood]; vel=Umotion[i_all,:][idgood,:];
# 						if PLOT_DATA==3: col = 10**errU_filt[i_all][idgood]; vel=Umotion[i_all,:][idgood,:];
# 						if PLOT_TYPE==2: vel[:]=0;
# 					# the last argument send a stop signal to the worker
# 					if len(X1)>0: q=[Im,X1[idgood,:],vel,col,N[i],np.mean(errU[idgood]),th,F2f,plot_time_stat,Cmin,Cmax,alpha,not(i==len(N)-1)]
# 					if len(X1)==0: q=[Im,[],vel,col,N[i],np.nan,th,F2f,plot_time_stat,Cmin,Cmax,alpha,not(i==len(N)-1)]

			data_dict = { "image": Im ,"vel": vel, "color":col}
			self.new_data.emit(data_dict)

			# Update Background
			if th[0]['BG']:
				r2=B-I2f
				r=-np.float32(r2>th[0]['BGspeed'])*th[0]['BGspeed']+np.float32(r2<-th[0]['BGspeed'])*th[0]['BGspeed']-np.float32(np.abs(r2)<=th[0]['BGspeed'])*r2;
				B=np.minimum(np.maximum(B+r,0),1)
	
			# Print Infos
			elapsed = time.time() - t
	
			infos= '%04d + %d Pts (%d%% recovered) - Time %1.2fs ' % (N[i],C1.shape[0],len(idgood)*100./np.maximum(0.1,C1.shape[0]),elapsed)+"="*20
			if INFO: print(infos)
			# Prepare matrix for next iteration
			# 1 -> 0
			#A0=A1;
			# KDTree
			t0=t1;t1=t2;
			ns01=ns12;
			i1001=i2112
			# 2 -> 1
			#print len(C0),len(C1),len(C2),len(ns12)
			C0=C1;C1=C2;U0=U1;U1=U2;I1=I2;I1f=I2f;
			#break
		############################ END of Main LOOPPP
		# local parameters
		#print(Np*p,self._tmax,self._tcorr)
		print('done')

if __name__ == "__main__":
	app = use_app("pyqt5")
	app.create()

	canvas_wrapper = CanvasWrapper()
	win = MyMainWindow(canvas_wrapper)
	data_thread = QtCore.QThread(parent=win)
	data_source = tractrac(win)
	data_source.moveToThread(data_thread)

	# update the visualization when there is new data
	data_source.new_data.connect(canvas_wrapper.update_data)
	
	# start data generation when the thread is started
	#data_thread.started.connect(data_source.run)
	
	win._controls.start_radio.clicked.connect(data_source.run)
	
	# if the data source finishes before the window is closed, kill the thread
	data_source.finished.connect(data_thread.quit, QtCore.Qt.DirectConnection)
	
	# if the window is closed, tell the data source to stop
	win.closing.connect(data_source.stop_data, QtCore.Qt.DirectConnection)
	
	# when the thread has ended, delete the data source from memory
	data_thread.finished.connect(data_source.deleteLater)

	win.show()
	data_thread.start()
	app.run()

	print("Waiting for data source to close gracefully...")
	data_thread.wait(5000)