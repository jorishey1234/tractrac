#%%
################ TRACTRAC ############
#==============================================================================
global version
version= '2.01 (??/??/??)'
#==============================================================================
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
# Fixed some difference between Matlab and Python version, some diff remain notably in the error thresholds

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
import imutils # To check opencv versions
from scipy.interpolate import griddata
import os,os.path
import argparse
import h5py # For saving binary format
#from tractrac_toolbox import *
#def run(**kwargs):
#% Tractrac Toolbox

def Propagate_MotionModel_KdTree(C,Xm,Um,Em,Xm_old,Um_old,Em_old,th):
	# KdTree
	Xref=np.vstack((Xm,Xm_old))
	Uref=np.vstack((Um,Um_old))
	Eref=np.hstack((Em,Em_old))
	#Make Tree
	tXref=scp.cKDTree(Xref)

	# Get firsts nn'th neighboors of query points
	nn=np.minimum(np.minimum(np.int8(th[0]['motion_av']),C.shape[0]),Xref.shape[0])
	#print nn
	distance,neighboors=tXref.query(C,k=nn)

	# Make matrices and compute local average
	neighboors=np.array(neighboors)
	values_U=Uref[neighboors.flatten(),:].reshape(-1,nn,2)
	values_E=Eref[neighboors.flatten()].reshape(-1,nn)

	# Averaging over neighboors values

	# Option 1 : simple average of neighbooring values
#	U_filt=np.nanmean(values_U,axis=1).reshape(-1,2)
#	E_filt=np.nanmean(values_E,axis=1).reshape(-1)

	# Option 2 : weighted average with weight on previous model
	W=np.ones(Xref.shape[0])
	#W[:Xm.shape[0]]=W[:Xm.shape[0]]*np.maximum(0,-Em)
	W[-Xm_old.shape[0]:]=W[-Xm_old.shape[0]:]*th[0]['filter_time']#*np.maximum(0,-Em_old)
	Wneighboors=W[neighboors.flatten()].reshape(-1,nn)
	U_filt=np.sum(values_U*Wneighboors.reshape(Wneighboors.shape[0],Wneighboors.shape[1],-1),axis=1).reshape(-1,2)/np.sum(Wneighboors,axis=1).reshape(Wneighboors.shape[0],-1)
	E_filt=np.sum(values_E*Wneighboors,axis=1)/np.sum(Wneighboors,axis=1)

	# Option 2 : weighted average with weight on previous model + distance + error


	# Find nan and replace by default 0 value
	idgood=np.isfinite(U_filt[:,0])

	# Save non nan to ponderate next iteration
	Xm_old=C[idgood,:]
	Um_old=U_filt[idgood,:]
	Em_old=E_filt[idgood]

	# Replace nan if no model points where given
	if len(U_filt)>0:
		idnan=np.isnan(U_filt[:,0])
		U_filt[idnan,:]=0
		E_filt[np.isnan(E_filt)]=2

	return U_filt,E_filt,Xm_old,Um_old,Em_old
#%

# Initialize plot window and cbar
def init_plot(w,h):
	global fig,ax,qui_c,Cmin,Cmax,alpha,img,qui
	size=10.0
	fig=plt.figure(figsize=(float(w)/float(h)*size,size))
	ax=fig.add_subplot(111)
	plt.show(block=False)
	qui=plt.quiver([],[],[],[], [], cmap = cm.rainbow, headlength=7,alpha=0.5)
	plt.clim(Cmin,Cmax)
	qui_c=plt.colorbar(qui,orientation='vertical', ax=ax,label='Velocity [px/frame]')
	img=plt.imshow(np.zeros((h,w)),cmap='gray',clim=[0,1],zorder=-1,interpolation='nearest')
	ax.axis([0,w,h,0])

def plot(q):
	global fig,ax,qui_c,Cmin,Cmax,alpha,th,SAVE_PLOT,plot_folder,img,qui
	image,points,vectors,col,n,err = q
	img.set_data(image)
	if len(vectors)>0:
		norm=np.sqrt(vectors[:,0]**2.+vectors[:,1]**2)
		norm[norm==0]=1 # To avoid division problem
		vectors[:,1]=vectors[:,1]/norm
		vectors[:,0]=vectors[:,0]/norm
#		vth=0.
#		idquiv=np.where(norm>vth)[0]
#		idscat=np.where(norm<vth)[0]
	#ax.cla()
	# Print Image
	# Print Scatter
	#sca=plt.scatter(points[:,0],points[:,1],s=th[0]['peak_conv_size']*20,c=col,alpha=alpha, cmap = cm.rainbow,vmin=Cmin,vmax=Cmax)
	# Print Quiver
#	angle=np.arctan2(vectors[:,0],vectors[:,1])*180/3.14
#	for i in range(len(col)):
#		plt.scatter(points[i,0],points[i,1],s=th[0]['peak_conv_size']*20,c=col[i],marker=(3,0,angle[i]),alpha=alpha, cmap = cm.rainbow,vmin=Cmin,vmax=Cmax)
	# Quver plot
	if len(vectors)==0:
		qui.remove()
		qui=plt.quiver([],[],[],[], [], cmap = cm.rainbow, headlength=7,alpha=0.5)
	else:
		#qui=plt.quiver(points[idquiv,0],points[idquiv,1],vectors[idquiv,0],-vectors[idquiv,1], col[idquiv], cmap = cm.rainbow,  pivot='middle', linewidth=.0,headwidth=1., headaxislength=1.,alpha=alpha); plt.clim(Cmin,Cmax)	
		qui.remove()
		qui=plt.quiver(points[:,0],points[:,1],vectors[:,0],-vectors[:,1], col[:], cmap = cm.rainbow,  pivot='middle', linewidth=.0,headwidth=1., headaxislength=1.,alpha=alpha); plt.clim(Cmin,Cmax)
		#plt.scatter(points[idscat,0],points[idscat,1],c=col[idscat], s=2., cmap = cm.rainbow, alpha=alpha,edgecolors=None); plt.clim(Cmin,Cmax)
	
	#ax.invert_yaxis()
	#vectors[:,1]=-vectors[:,1]
	if len(vectors)==0:
		title='TracTrac v2.0 | Frame {:04d} | No object found'.format(n)
	else:
		title='TracTrac v2.0 | Frame {:04d} | Points tracked {:5d} | Mean Err. {:1.4f}px'.format(n,points.shape[0],10**err)
	ax.set_title(title)
	fig.canvas.draw()
	if SAVE_PLOT:
		imname=plot_folder+'img{:04d}.png'.format(n)
		plt.savefig(imname,bbox_inches='tight',dpi=100)

# For plotting in Parralal process
def visualization_worker(q):
	global fig,ax,qui_c,Cmin,Cmax,alpha,th,plot_folder
	stop=True
	while stop:
		image,points,vectors,col,n,err,stop = q.get()
		plot([image,points,vectors,col,n,err])

def read_parameter_file(filename):
	par=[{}]
	if os.path.exists(filename):
		with open(filename) as f:
			for line in f:
				s=search('{varname:w} {varvalue:g}',line)
				if (s != None): par[0][str(s['varname'])]=s['varvalue']
			print 'Parameter file read !'
	else:
		print 'WARNING: no parameter file exists. Taking default values! '
	return par

def write_parameter_file(filename,th):
	global version
	f = open(filename, 'w')
	f.write('# Parameter file generated by TracTrac Python v'+ version+'\n\n')
	f.write('# Video loops \n')
	f.write('vid_loop {}\n\n'.format(th[0]['vid_loop']))
	f.write('# Image Processing\n')
	f.write('ROIxmin {}\n'.format(th[0]['ROIxmin']))
	f.write('ROIymin {}\n'.format(th[0]['ROIymin']))
	f.write('ROIxmax {}\n'.format(th[0]['ROIxmax']))
	f.write('ROIymax {}\n'.format(th[0]['ROIymax']))
	f.write('BG {}\n'.format(th[0]['BG']))
	f.write('BGspeed {}\n'.format(th[0]['BGspeed']))
	f.write('noise {}\n'.format(th[0]['noise']))
	f.write('noise_size {}\n\n'.format(th[0]['noise_size']))
	f.write('# Object Detection\n')
	f.write('peak_th {}\n'.format(th[0]['peak_th']))
	f.write('peak_th_auto {}\n'.format(th[0]['peak_th_auto']))
	f.write('peak_neigh {}\n'.format(th[0]['peak_neigh']))
	f.write('peak_conv {}\n'.format(th[0]['peak_conv']))
	f.write('peak_conv_size {}\n'.format(th[0]['peak_conv_size']))
	f.write('peak_subpix {}\n'.format(th[0]['peak_subpix']))
	f.write('peak_minima {}\n\n'.format(th[0]['peak_minima']))
	f.write('# Motion Model \n')
	f.write('motion {}\n'.format(th[0]['motion']))
	f.write('motion_av {}\n'.format(th[0]['motion_av']))
	f.write('motion_it {}\n'.format(th[0]['motion_it']))
	f.write('filter {}\n'.format(th[0]['filter']))
	f.write('filter_time {}\n'.format(th[0]['filter_time']))
	f.write('filter_std {}\n'.format(th[0]['filter_std']))
	f.close()


def set_default_parameter(th,w,h):
# set missing parameters to default fvalues
	if not('ROIxmin' in th[0]):th[0]['ROIxmin']=0
	if not('ROIxmax' in th[0]):th[0]['ROIxmax']=w
	if not('ROIymin' in th[0]):th[0]['ROIymin']=0
	if not('ROIymax' in th[0]):th[0]['ROIymax']=h
	if not('BG' in th[0]):th[0]['BG']=0
	if not('BGspeed' in th[0]):th[0]['BGspeed']=0.02
	if not('noise' in th[0]):th[0]['noise']=0
	if not('noise_size' in th[0]):th[0]['noise_size']=3
	if not('peak_th' in th[0]):th[0]['peak_th']=0.02
	if not('peak_th_auto' in th[0]):th[0]['peak_th_auto']=1
	if not('peak_neigh' in th[0]):th[0]['peak_neigh']=1
	if not('peak_conv_size' in th[0]):th[0]['peak_conv_size']=2.2
	if not('peak_conv' in th[0]):th[0]['peak_conv']=1
	if not('peak_subpix' in th[0]):th[0]['peak_subpix']=1
	if not('peak_minima' in th[0]):th[0]['peak_minima']=0
	if not('motion' in th[0]):th[0]['motion']=1
	if not('motion_av' in th[0]):th[0]['motion_av']=10
	if not('motion_it' in th[0]):th[0]['motion_it']=0
	if not('filter' in th[0]):th[0]['filter']=1
	if not('filter_time' in th[0]):th[0]['filter_time']=1.0
	if not('filter_std' in th[0]):th[0]['filter_std']=1.5
	if not('vid_loop' in th[0]):th[0]['vid_loop']=2
	return th


def times_f(a,b):
#	print a.shape,b.shape
	c=a*b
	return c

def blob_detection(F,th):
# LoG or DoG detection kernel
	scale=th[0]['peak_conv_size']
	size=(np.maximum(3,(int(scale*3.)/2)*2+1),np.maximum(3,(int(scale*3.)/2)*2+1))
	#print size
	Ff=-F # No kernel
	if th[0]['peak_conv']==1 : # DoG kernel
		Ff = cv2.GaussianBlur(F,size,scale*0.8,cv2.BORDER_REPLICATE)
		Ff = cv2.GaussianBlur(Ff,size,scale*1.2,cv2.BORDER_REPLICATE)-Ff
	if th[0]['peak_conv']==2 : # LoG kernel
		Ff_temp = cv2.GaussianBlur(F,size,scale*1.0,cv2.BORDER_REPLICATE)
		Ff = cv2.Laplacian(Ff_temp,cv2.CV_32F,cv2.BORDER_REPLICATE)
		#print Ff.shape,Ff.max()
	if th[0]['peak_minima']==0 : Ff=-Ff
	return Ff
	
def feature_detection(F,th):
# Detection of remarkable feature : Harris, Log and DoG kernel
	if th[0]['peak_conv']==3: # OPENCV implementation of Shi and Tomasi
		Ff=F
		corners = cv2.goodFeaturesToTrack(Ff,10000,th[0]['peak_th'],th[0]['peak_conv_size'])
		x,y=np.array([cc[0,0] for cc in corners]),np.array([cc[0,1] for cc in corners])
		z=np.zeros((len(x)))
	else: # Blob detection based on Log and Dog kernel followed by peak finder
		Ff=blob_detection(F,th)
		[x,y,z]=maximaThresh(Ff,1+2*int(th[0]['peak_neigh']),th)
	return Ff,x,y,z

def imProc(I,th):
# Default pre-processing steps on images
	if len(I.shape)>2: I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
	if I.dtype == 'uint8': Im = np.float32(I)/256. # ALways convert to float Images
	if I.dtype == 'uint16': Im = np.float32(I)/2.**16 # ALways convert to float
	# Crop
	ROIymin=np.minimum(0,th[0]['ROIymin'])
	ROIymax=np.minimum(I.shape[0],th[0]['ROIymax'])
	ROIxmin=np.maximum(0,th[0]['ROIxmin'])
	ROIxmax=np.minimum(I.shape[1],th[0]['ROIxmax'])
	Im = Im[ROIymin:ROIymax,ROIxmin:ROIxmax]
	#Filtering
	if th[0]['noise'] == 1: Im = cv2.medianBlur(Im,th[0]['noise_size'])
	#Im = cv2.GaussianBlur(Im,(5,5),2)
	return Im

def distanceMat(x,y):
	dist=scipy.spatial.distance.cdist(x,y)
	return dist

def cropPoints(x,y,ROI):
	idgood = np.where((x>ROI[0,0])&(x<ROI[1,0])&(y>ROI[0,1])&(y<ROI[1,1]))
	return idgood[0]

def cropPoints3d(x,y,z,ROI):
	idgood = np.where((x>ROI[0,0])&(x<ROI[1,0])&(y>ROI[0,1])&(y<ROI[1,1])&(z>ROI[0,2])&(z<ROI[1,2]))
	return idgood[0]

def initialize_ROI(size,n,nbnd):
	ROIbnd=[]
	ROI=[]
	for j in range(n[0]):
		for i in range(n[1]):
			for k in range(n[2]):
				ROIbnd.append(np.array([[(j)*size[0]/n[0]-nbnd[0],(i)*size[1]/n[1]-nbnd[1],k*size[2]/n[2]-nbnd[2]],[(j+1)*size[0]/n[0]+nbnd[0],(i+1)*size[1]/n[1]+nbnd[1],(k+1)*size[2]/n[2]+nbnd[2]]]))
				ROI.append(np.array([[(j)*size[0]/n[0],(i)*size[1]/n[1],k*size[2]/n[2]],[(j+1)*size[0]/n[0],(i+1)*size[1]/n[1],(k+1)*size[2]/n[2]]]))
	return ROIbnd,ROI


def imadjust(I):
	I=(I-np.min(I))/(np.max(I)-np.min(I))
	return I

def maximaThresh(a,n,th):
	# Find n*n local max
	method=th[0]['peak_subpix']
	a=np.float64(a)
	if a.max()<th[0]['peak_th']:
		print 'Warning : peak_th ({:1.4f}) is above the maximum image convoluted value ({:1.4f}).'.format(th[0]['peak_th'],a.max())
	r=np.random.rand(np.shape(a)[0],np.shape(a)[1])*1e-5
	mask=np.ones((n,n),np.uint8)
	b=cv2.dilate(a+r,mask,iterations = 1)
	c=((a+r==b)&(a>th[0]['peak_th']))
	[y,x]=np.where(c)
	# Remove points on border
	w=np.shape(a)[1]
	h=np.shape(a)[0]
	nb=np.floor(n/2.)
	idbnd=np.where((y<h-nb)&(y>=nb)&(x<w-nb)&(x>=nb))
	x=np.array(np.float64(x[idbnd])).reshape(-1)
	y=np.array(np.float64(y[idbnd])).reshape(-1)
	# Subpixel Refinement Method
	# 2nd order poly fit on the logarithm of diag of a
	Dx=np.zeros(x.shape);Dy=np.zeros(x.shape);z=np.zeros(x.shape);
	if method==1: [Dx,Dy,z]=subpix2nd(np.real(np.log(a-np.min(a)+1e-8)),x,y,n) #Gaussian
	if method==0: [Dx,Dy,z]=subpix2nd(a,x,y,n) # Quadratic
	# Take only peaks that moved less than 0.5
	idgood=np.where((np.abs(Dx)<0.5)&(np.abs(Dy)<0.5))
	#print x,y,Dx,Dy
	x=x[idgood]+Dx[idgood]
	y=y[idgood]+Dy[idgood]
	z=z[idgood]
	return x,y,z

def subpix2nd(a,x,y,n):
	#% Subpixel approximation of a 2nd order polynomial with a pencil of length
	#% np
	npen=np.floor(n/2.)
	pencil=np.arange(-npen,npen+1)
	X=np.matlib.repmat(pencil,np.size(x),1)
	YH=np.zeros(X.shape)
	YV=np.zeros(X.shape)
#	YD1=np.zeros(X.shape)
#	YD2=np.zeros(X.shape)
	n=np.float32(len(pencil))
	for i in range(0,len(pencil)):
		idV=sub2ind(np.shape(a),np.maximum(0,np.minimum(a.shape[0]-1,y+pencil[i])),x)
		idH=sub2ind(np.shape(a),y,np.maximum(0,np.minimum(a.shape[1]-1,x+pencil[i])))
		YV[:,i]=a.flat[idV]
		YH[:,i]=a.flat[idH]
	# 2nd order poly a+bx+cx^2=0
	s2=np.sum(pencil**2.)
	s4=np.sum(pencil**4.)
	bH=np.sum(YH*X,1)/s2
	cH=-(-s2*np.sum(YH,1)+n*np.sum(X**2.*YH,1))/(s2**2.-s4*n)
	bV=np.sum(YV*X,1)/s2
	cV=-(-s2*np.sum(YV,1)+n*np.sum(X**2.*YV,1))/(s2**2.-s4*n)
	cH[cH==0]=1e-8
	cV[cV==0]=1e-8
	# Peaks on hor and vert axis
	dH=-bH/cH/2.
	dV=-bV/cV/2.
	Dx=dH
	Dy=dV
	Z=YH[:,int((n-1)/2.)]
	return Dx,Dy,Z

def is_image(f):
	# check if f is an image
	flag_im=0
	if len(f)>=4:
		if (f[-3:]=='tif') |  (f[-3:]=='TIF') | (f[-4:]=='tiff') | (f[-3:]=='png') | (f[-3:]=='jpg') | (f[-4:]=='jpeg') :
			flag_im=1
	return flag_im

def sub2ind(array_shape, rows, cols):
	return np.uint32(rows*array_shape[1] + cols)


##############################################
# Function to initiate a computation Tractrac inside a python sript.
#
# Optional input arguments are :
# f=videofilename (string) : Pathname of the video file to analyse.
# tf=time_filename (string) : Pathname of the time stamp of video frames.
# p=plotting_option (0,1,2,3): for no plot, normal plot, motion model plot, or motion model error plot.
# o=output_file (0,1,2) : (1) save the tracking results in a ASCII file of name videofilename_track.txt or (2) in a binary hdf5 file videofilename_track.hdf5.
# s=silent (0,1) : verbose or silent computation.
# sp=save_plot (0,1) : Save images of each plots.
# clim=(float,float) : caxis color limits for plotting.
# alpha=float : Transparency of plotting
#
# Output arguments are:
# Pts (np.array) : Result array of tracklets. Columns are in order : 0 Frame, 1 Track ID, 2 X, 3 Y, 4 U, 5 V, 6 U motion model, 7 V motion model of columns, 8 X Acceleration, 9 Y acceleration, 10 Motion model error
# th (list dictionnary) : Tracking parameters
##############################################
#%%
def run(**kwargs):
	global SAVE_PLOT,Cmin,Cmax,alpha,INFO,PAR,plot_folder
	# Take arguments or Default values
	filename=kwargs.get('f','./Sample_videos/videotest.avi')	# Default Sample Video
	tfile=kwargs.get('tf','') # File with time stamp of video frames
	mmfilename=kwargs.get('mmf','')
	PLOT=kwargs.get('p',0)
	OUTPUT=kwargs.get('o',0)
	INFO=kwargs.get('s',0)
	SAVE_PLOT=kwargs.get('sp',0)
	Cmin,Cmax=kwargs.get('clim',(0.0,5.0))
	alpha=kwargs.get('calpha',0.5)
	th=kwargs.get('th',[{}])
	PAR=kwargs.get('par',1)
	Pts,th=tractrac(filename,th,mmfilename,tfile,PLOT,OUTPUT)
	return Pts,th

# Main tracking FUNCTION
def tractrac(filename,th,mmfilename,tfile,PLOT,OUTPUT):
	global SAVE_PLOT,Cmin,Cmax,alpha,INFO,PAR,version,plot_folder

#%%

	# Check if file exist
	#	if not(os.path.isfile(filename)):
	#		print 'Video File does not exist! Abord.'
	#		sys.exit()

	nchar=43
	sep="="*nchar
	if INFO:
		print sep
		print "|"+ ' TRACTRAC v'+version+' - Heyman J. '+ " |"
		print sep
		print("> OpenCV Version: {}".format(cv2.__version__)) 	# Check OpenCV version
		print '> file : '+filename
		print sep

	# Read Video Stream or image sequence
	flag_im=is_image(filename)
	flag_web=0 # flag if videosource is webcam
	if flag_im: # Image list
		flist=sorted(glob.glob(filename))
		#pdb.set_trace()
		I0=cv2.imread(flist[0],2)
		nFrames=len(flist)
		width=I0.shape[1]
		height=I0.shape[0]
	elif filename=='0': # WebCam
		flag_web=1
		cv2.destroyAllWindows()
		cap = cv2.VideoCapture(0)
		nFrames=10000
		width=int(cap.get(3))
		height=int(cap.get(4))
	else:	# Video
		cv2.destroyAllWindows()
		cap = cv2.VideoCapture(filename)
		if imutils.is_cv2():
			nFrames=int(cap.get(cv2.cv.CAP_PROP_FRAME_COUNT))
			width=int(cap.get(cv2.cv.CAP_PROP_FRAME_WIDTH))
			height=int(cap.get(cv2.cv.CAP_PROP_FRAME_HEIGHT))
		elif imutils.is_cv3():
			nFrames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		else:
			print 'Bad OpenCV version. Please install opencv3'
			sys.exit()

	if nFrames==0:
		print 'No Frames to track. Check filename.'
		sys.exit()

	# Read Parameters or set to defaut values
	path,name=os.path.split(filename)
	if len(path)==0:
		path='./'
	parameter_filename=path+'/' + name[:-4]+'_par.txt'

	if not th[0]: th = read_parameter_file(parameter_filename)
	# Set remaining Parameters and Save
	th = set_default_parameter(th,width,height)
	write_parameter_file(parameter_filename,th)

	# Convert to int several par (defaut is float)
	th[0]['ROIxmin']=np.maximum(int(th[0]['ROIxmin'])-1,0)
	th[0]['ROIxmax']=int(th[0]['ROIxmax'])
	th[0]['ROIymin']=np.maximum(int(th[0]['ROIymin'])-1,0)
	th[0]['ROIymax']=int(th[0]['ROIymax'])
	th[0]['noise_size']=int(th[0]['noise_size'])
	th[0]['peak_neigh']=int(th[0]['peak_neigh'])

#	if OUTPUT: write_parameter_file(parameter_filename,th)

	if INFO:
		print th
		print "="*(65)

	if SAVE_PLOT:
		plot_folder=path+'/'+name[:-4]+'_img/'
		if INFO:
			print 'Images will be saved in '+plot_folder
		if not(os.path.exists(plot_folder)): os.makedirs(plot_folder);

	# Read time file to get time step
	dt=np.ones(nFrames-1)
	if os.path.isfile(tfile):
		Time_stamp=np.loadtxt(tfile)
		if Time_stamp.shape[0]==nFrames:
			dt=np.diff(Time_stamp)

#%%

	Pts=[]
	
	# Initialize Plot
	if PLOT>0:
		init_plot(width,height)
		if PAR:
			queue = mp.Queue()
			p = mp.Process(target=visualization_worker, args=(queue,))
			p.start()
	
	#Initialize Background
	B=np.zeros((height,width),dtype=np.float32)
	if (th[0]['BG']==1)&(flag_web==0):
			nbgstart=np.minimum(100,nFrames)
			print 'Initiating Background image over the firsts {:d} frames...'.format(nbgstart)
			for i in range(nbgstart):
				I = cv2.imread(flist[i],2) if flag_im else cap.read()[1];
				I=imProc(I,th)
				B=B+I*mask/nbgstart
		# Rewind video
			if (imutils.is_cv2()) & (not flag_im):
				cap.set(cv2.cv.CAP_PROP_POS_FRAMES,0);
				cap.read()
				cap.read()
			elif (imutils.is_cv3()) & (not flag_im):
				cap.set(cv2.CAP_PROP_POS_FRAMES,0);
				cap.read()
				cap.read()


	# Read 2 first frames and initialize
	I0 = cv2.imread(flist[0],2) if flag_im else cap.read()[1];
	if flag_web: time0=time.time()
	I0f=imProc(I0,th)
	I1 = cv2.imread(flist[1],2) if flag_im else cap.read()[1];
	if flag_web:
		time1=time.time()
		dt[0]=time1-time0
		time0=time1
	I1f=imProc(I1,th)


	# Read Mask file if any
	mask_file=path+'/' + name[:-4]+'_mask.tif'
	if os.path.isfile(mask_file):
		mask=cv2.imread(mask_file,2)
		mask=imProc(mask,th)
	else:
		mask=np.ones(I1f.shape,dtype='float32')


	w=np.shape(I0f)[1]
	h=np.shape(I0f)[0]


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
			print '! The Provided Motion Model File will be used !'

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
	if INFO: print '0001 | Buffer frame...'
	for i in range(2,len(N)):
#	for i in range(2,399):
		#print 'top'
		t = time.time()

		# PRE processing steps #################################################
		if flag_im:
			I2 = cv2.imread(flist[N[i]],2)
		elif flag_web:
			I2 = cap.read()[1];
		else:
			cap.set(1,N[i])
			I2 = cap.read()[1];

		if I2 is None:
			if PAR and PLOT: p.join();
			print 'WARNING: Video reader get None type. Exiting at frame {:d}.'.format(N[i])
			break

		if flag_web:
			time1=time.time()
			dt[i-1]=time1-time0
			time0=time1
			print 'Webcam Framerate:',1/dt[i-1]

		I2f=imProc(I2,th)
		F2 = I2f*mask-B

		F2f,x,y,z=feature_detection(F2,th)
		C2=np.transpose(np.array([x,y]))
		#C2=C2[np.argsort(C2[:,0],axis=0),:]
		U2=np.zeros(C0.shape)
		A2=np.zeros(C0.shape)
		# END PREPROCESSING  #################################################

		it=0
		while it<=th[0]['motion_it']:
			# PREDICTION from motion model
			if (th[0]['motion']==1)&(len(idgood)+Xmot.shape[0]>1)&(len(C1)>0):
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
			if len(errU)>0:
				infos= '     | Motion Model it %02d - %d pts - log Err. Av/Max %1.1f/%1.1f (px) ' % (it,len(idgood),np.mean(errU),np.mean(errU+errU_th))
			else:
				infos= '     | Motion Model it %02d - %d pts - log Err. Av/Max ??/%1.1f (px) ' % (it,0,errU_th)
			if INFO: print infos
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
		if OUTPUT & (i>=len(N)-nFrames+2) :
			newPts=np.vstack(((np.zeros(len(idgood))+N[i]-np.sign(dt[i-1])).T,ID[idgood].T, X1[idgood,:].T, um[idgood,:].T,A[idgood,:].T, Umotion[idgood,:].T,errU[idgood].T)).T
			Pts.append(newPts) # We use list for dynamic resizing of arrays

	# Plotting
		if (PLOT>0):
			if PAR==0:
				if len(um)==0:
					col = []; vel= [];
				else:
					if PLOT==1: col = np.sqrt(um[idgood,1]**2+um[idgood,0]**2); vel=um[idgood,:];
					if PLOT==2: col = np.sqrt(Umotion[i_all,1]**2+Umotion[i_all,0]**2)[idgood]; vel=Umotion[i_all,:][idgood,:];
					if PLOT==3: col = 10**errU_filt[i_all][idgood]; vel=Umotion[i_all,:][idgood,:];
				if len(X1)>0: q=[I1f,X1[idgood,:],vel,col,N[i],np.mean(errU[idgood])]
				if len(X1)==0: q=[I1f,[],vel,col,N[i],np.nan]
				plot(q)
			elif (queue.empty()) or (SAVE_PLOT==1): # Only plot when queue is empty or when the save order is given
				if len(um)==0:
					col = []; vel= [];
				else:
					if PLOT==1: col = np.sqrt(um[idgood,1]**2+um[idgood,0]**2); vel=um[idgood,:];
					if PLOT==2: col = np.sqrt(Umotion[i_all,1]**2+Umotion[i_all,0]**2)[idgood]; vel=Umotion[i_all,:][idgood,:];
					if PLOT==3: col = 10**errU_filt[i_all][idgood]; vel=Umotion[i_all,:][idgood,:];
				# the last argument send a stop signal to the worker
				if len(X1)>0: q=[I1f,X1[idgood,:],vel,col,N[i],np.mean(errU[idgood]),not(i==len(N)-1)]
				if len(X1)==0: q=[I1f,[],vel,col,N[i],np.nan,not(i==len(N)-1)]
				queue.put(q)

		# Update Background
		if th[0]['BG']:
			r2=B-I2f
			r=-np.float32(r2>th[0]['BGspeed'])*th[0]['BGspeed']+np.float32(r2<-th[0]['BGspeed'])*th[0]['BGspeed']-np.float32(np.abs(r2)<=th[0]['BGspeed'])*r2;
			B=np.minimum(np.maximum(B+r,0),1)

		# Print Infos
		elapsed = time.time() - t

		infos= '%04d + %d Pts (%d%% recovered) - Time %1.2fs ' % (N[i],C1.shape[0],len(idgood)*100./np.maximum(0.1,C1.shape[0]),elapsed)+"="*20
		if INFO: print infos
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

#%%
	#cv2.destroyAllWindows()
	#if INFO: print '%04d | Buffer frame...' % (N[i+1])
	if PAR and PLOT:
		if INFO:
			print "="*(65)
			print 'Waiting for visualization thread end...'
		p.join()
		p.terminate()
		plt.close('all')
	if INFO: print 'End ! A total of {:d} objects have been tracked.'.format(id_traj)
	#print OUTPUT
	# SAVING ASCII ####################
	if OUTPUT==1:
		# Transform Pts into a numpy array
		Pts=np.concatenate(Pts)
		output_filename=path+'/' + name[:-4]+'_track.txt'
		if INFO:
			print 'Saving to ASCII file '+ output_filename +'...'
		head='TracTrac v'+version +' \n Parameters: '+str(th[0])+'\n Frame ID x y Vx Vy Ax Ay Vx(prediction) Vy(prediction) Error(prediction)'
		#newPts=np.vstack(((np.zeros(len(idgood))+n-1).T,ID[idgood].T, X1[idgood,:].T, um[idgood,:].T, Umotion[idgood,:].T,a[idgood,:].T,errU[idgood].T)).T
		np.savetxt(output_filename,Pts,fmt=('%d','%d','%.3f','%.3f','%.5f','%.5f','%.4f','%.4f','%.4f','%.4f','%.3f'),delimiter=' ', newline='\n', header=head, comments='# ')
		if INFO:
			print 'Saved !'
			print "="*(65)

	# SAVING HDF5 ####################
	if OUTPUT==2:
		# Transform Pts into a numpy array
		Pts=np.concatenate(Pts)
		output_filename=path+'/' + name[:-4]+'_track.hdf5'
		if INFO:
			print 'Saving to binary file '+ output_filename +'...'
		f = h5py.File(output_filename,'w')
		f.attrs['version']='HDF5 file made with TracTrac Python v'+version
		f.attrs['date']=time.strftime("%d/%m/%Y")
		f.attrs['nFrames']=nFrames
		f.attrs['size']=I0f.shape
		for items in th[0].keys(): f.attrs['th:'+items]=th[0][items]
		f.create_dataset("Frame", data=np.uint16(Pts[:,0]))
		f.create_dataset("Id", data=np.uint16(Pts[:,1]))
		f.create_dataset("x", data=np.float32(Pts[:,2:4]))
		f.create_dataset("u", data=np.float32(Pts[:,4:6]))
		f.create_dataset("a", data=np.float32(Pts[:,6:8]))
		f.create_dataset("u_motion", data=np.float32(Pts[:,8:10]))
		f.create_dataset("err_motion", data=np.uint32(Pts[:,10]))
		f.close()

		if INFO:
			print 'Saved !'
			print "="*(65)

	#	if OUTPUT_PP:
	#		#pdb.set_trace()
	#		output_filename=path+'/' + name[:-4]+'_post.txt'
	#		print 'PostProcessing will be saved to '+ output_filename +'...'
	#		post_save(output_filename,Pts,th)
	#		print 'Saved !'
	#		print "="*(65)
	return Pts,th



#%% Run as a script
if __name__ == "__main__":
	global SAVE_PLOT,Cmin,Cmax,alpha,INFO,PAR
	parser = argparse.ArgumentParser(description='TRACTRAC v'+version+' - Joris Heyman')
	parser.add_argument('-f','--file', type=str, help='Video Filename to track',default='./Sample_videos/videotest.avi')
	parser.add_argument('-tf','--tfile', type=str, help='Time of frame file',default='')
	parser.add_argument('-mmf','--motionmodelfile', type=str, help='Motion Model file',default='')
	parser.add_argument('-o','--output', type=int, help='Save tracking results in a file ASCII (1) or HDF5 (2)',default=0)
	parser.add_argument('-opp','--outputpp', help='Save Post Processing results in a file', action='store_true',default=False)
	parser.add_argument('-s','--silent',help='No tracking infos', action='store_false',default=True)
	# Plotting Options
	parser.add_argument('-p','--plot', type=int,help='Live plot of tracking results', default=0)
	parser.add_argument('-sp','--saveplot', help='Save plots in image sequence', action='store_true')
	parser.add_argument('-cmin','--cmin', type=float, help='Minimum velocity (px/frame) for plotting',default=0.)
	parser.add_argument('-cmax','--cmax', type=float, help='Maximum velocity (px/frame) for plotting',default=3.)
	parser.add_argument('-ca','--calpha', type=float, help='Alpha value for arrows',default=0.5)
	parser.add_argument('-par','--parallel', type=int,help='Visualization in a Parallel Thread', default=1)

	args = parser.parse_args()
	filename=args.file
	tfile=args.tfile
	mmfilename=args.motionmodelfile
	PLOT=args.plot
	OUTPUT=args.output
	OUTPUT_PP=args.outputpp
	INFO=args.silent
	SAVE_PLOT=args.saveplot
	Cmin=args.cmin
	Cmax=args.cmax
	alpha=args.calpha
	PAR=args.parallel
	th=[{}]
#%%
	Pts,th=tractrac(filename,th,mmfilename,tfile,PLOT,OUTPUT)

