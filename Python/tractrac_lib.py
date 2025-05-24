#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.matlib
import scipy.spatial.distance
import scipy.spatial as scp
import cv2
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import os,os.path

def Propagate_MotionModel_KdTree(C,Xm,Um,Em,Xm_old,Um_old,Em_old,th):
	# KdTree
	Xref=np.vstack((Xm,Xm_old))
	Uref=np.vstack((Um,Um_old))
	Eref=np.hstack((Em,Em_old))
	#Make Tree
	tXref=scp.cKDTree(Xref)

	# Get firsts nn'th neighboors of query points
	nn=np.minimum(np.minimum(np.uint16(th[0]['motion_av']),C.shape[0]),Xref.shape[0])
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
def init_plot(w,h,nFrames):
	global fig,ax,ax1,ax2,qui_c,Cmin,Cmax,alpha,img,qui,th,qui_hist,qui_th,plot_time
	size=10.0
#	fig=plt.figure(figsize=(float(w)/float(h)*size,size), dpi=80)
	plt.style.use(['dark_background'])
	fig=plt.figure(figsize=(size,size), dpi=80)
	fig.canvas.manager.set_window_title('TracTrac v'+version)
	ax=fig.add_subplot(211)
	ax1=fig.add_subplot(224)
	ax2=fig.add_subplot(223)
	plt.show(block=False)
	qui=ax.quiver([],[],[],[], [], cmap = cm.rainbow, headlength=7,alpha=0.5)
	qui_c=plt.colorbar(qui,orientation='vertical', ax=ax,label='Velocity [px/frame]')
	img=ax.imshow(np.zeros((h,w)),cmap='gray',clim=[0,1],zorder=-1,interpolation='nearest')
	img.set_clim(0,1)
	qui_c_ticks = np.linspace(0, 1, 4)
	qui_c.set_ticks(qui_c_ticks)
	ax.axis([0,w,h,0])
	# Histogram
	qui_hist=ax1.plot([0],[0],'.')
	qui_th=ax1.plot([0,0],[1e-3,1e2],'r-',label='Detection threshold (peak_th)')
	ax1.set_yscale('log')
	ax1.set_xlim([-0.1,0.1])
	ax1.set_ylim([1e-3,1e2])
	ax1.set_xlabel('Object intensities')
	ax1.legend()
	#plt.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=0.05, hspace=0.01)
	# Plot statistics through time
	plot_time=[ax2.plot(np.arange(nFrames),np.zeros(nFrames),'r.',label='Objects (detected)'),
	ax2.plot(np.arange(nFrames),np.zeros(nFrames),'b.',label='Object (tracked)'),
	ax2.plot(np.arange(nFrames),np.zeros(nFrames),'g.',label='Model error (px)'),
	ax2.plot(np.arange(nFrames),np.zeros(nFrames),'c.',label='Mean speed (px/frame)')]
	ax2.set_xlim([0,nFrames])
	ax2.set_yscale('log')
	ax2.set_ylim([1e-1,1e5])
	ax2.set_xlabel('Frames')
	ax2.legend()
	fig.canvas.draw()
	#ax1.axis([0,w,h,0])
	

def plot(q):
	global fig,ax,ax1,ax2,qui_c,Cmin,Cmax,alpha,SAVE_PLOT,plot_folder,img,qui,qui_hist,qui_th,plot_time
	image,points,vectors,col,n,err,th,Im,plot_time_stat,Cmin,Cmax,alpha = q
	img.set_data(image)
	if vectors.all()==0: # scatter plot
		qui.remove()
		points_per_particles=((th[0]['peak_conv_size']*ax.get_window_extent().width  / image.shape[0] * 72./fig.dpi) ** 2)
		qui=ax.scatter(points[:,0],points[:,1],c=col[:],s=points_per_particles, cmap = cm.rainbow,alpha=alpha);# plt.clim(Cmin,Cmax)	
	else:
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
			qui=ax.quiver([],[],[],[], [], cmap = cm.rainbow, headlength=7,alpha=0.5)
		else:
			#qui=plt.quiver(points[idquiv,0],points[idquiv,1],vectors[idquiv,0],-vectors[idquiv,1], col[idquiv], cmap = cm.rainbow,  pivot='middle', linewidth=.0,headwidth=1., headaxislength=1.,alpha=alpha); plt.clim(Cmin,Cmax)	
			qui.remove()
			qui=ax.quiver(points[:,0],points[:,1],vectors[:,0],-vectors[:,1], col[:], cmap = cm.rainbow,  pivot='middle', linewidth=.0,headwidth=1., headaxislength=1.,alpha=alpha);
			qui.set_clim(Cmin,Cmax)
			#plt.scatter(points[idscat,0],points[idscat,1],c=col[idscat], s=2., cmap = cm.rainbow, alpha=alpha,edgecolors=None); plt.clim(Cmin,Cmax)
		
	#ax.invert_yaxis()
	#vectors[:,1]=-vectors[:,1]
	if len(points)==0:
		title='Frame {:04d} | No object found'.format(n)
	else:
		title='Frame {:04d} | Points tracked {:5d} |'.format(n,points.shape[0])
	ax.set_title(title)
	qui_c_ticks = ['{:1.1f}'.format(i) for i in np.linspace(Cmin, Cmax, 4)]
	qui_c.set_ticklabels(qui_c_ticks)
	########### Plot Histogram
	h,x=np.histogram(Im,300,density=True)
	qui_hist[0].set_xdata(x[1:])
	qui_hist[0].set_ydata(h)
	qui_th[0].set_xdata([th[0]['peak_th'],th[0]['peak_th']])
	########### plot statistics
	plot_time[0][0].set_ydata(plot_time_stat[:,0])
	plot_time[1][0].set_ydata(plot_time_stat[:,1])
	plot_time[2][0].set_ydata(plot_time_stat[:,2])
	plot_time[3][0].set_ydata(plot_time_stat[:,3])
	fig.canvas.draw()
	if SAVE_PLOT:
		imname=plot_folder+'img{:04d}.png'.format(n)
		fig.savefig(imname,bbox_inches='tight',dpi=100)

# For plotting in Parralal process
def visualization_worker(q):
	global fig,ax,qui_c,SAVE_PLOT,plot_folder,img,qui
	stop=True
	while stop:
		image,points,vectors,col,n,err,th,Im,plot_time_stat,Cmin,Cmax,alpha,stop = q.get()
		plot([image,points,vectors,col,n,err,th,Im,plot_time_stat,Cmin,Cmax,alpha])

def read_parameter_file(filename):
	par=[{}]
	if os.path.exists(filename):
		with open(filename) as f:
			for line in f:
				s=search('{varname:w} {varvalue:g}',line)
				if (s != None): par[0][str(s['varname'])]=s['varvalue']
		#	print('Parameter file read !')
	else:
		print('WARNING: no parameter file exists. Taking default values! ')
	return par
	
def setup_parameters(th):
	# Convert to int several par (defaut is float)
	th[0]['ROIxmin']=np.maximum(int(th[0]['ROIxmin'])-1,0)
	th[0]['ROIxmax']=int(th[0]['ROIxmax'])
	th[0]['ROIymin']=np.maximum(int(th[0]['ROIymin'])-1,0)
	th[0]['ROIymax']=int(th[0]['ROIymax'])
	th[0]['noise_size']=int(th[0]['noise_size'])
	th[0]['peak_neigh']=int(th[0]['peak_neigh'])
	return th

def write_parameter_file(filename,th):
	global version
	f = open(filename, 'w')
	f.write('# Parameter file generated by TracTrac Python v'+ version+'\n\n')
	f.write('# Video loops \n')
	f.write('vid_loop {} \t\t# Number of loops over frames to train motion model\n\n'.format(th[0]['vid_loop']))
	f.write('# Image Processing\n')
	f.write('ROIxmin {} \t\t# Region of interest for tracking (xmin) \n'.format(th[0]['ROIxmin']))
	f.write('ROIymin {} \t\t# Region of interest for tracking (ymin) \n'.format(th[0]['ROIymin']))
	f.write('ROIxmax {} \t\t# Region of interest for tracking (xmax) \n'.format(th[0]['ROIxmax']))
	f.write('ROIymax {} \t\t# Region of interest for tracking (ymax) \n'.format(th[0]['ROIymax']))
	f.write('BG {} \t\t# (0 or 1, default 0) Use background subtraction to remove static regions \n'.format(th[0]['BG']))
	f.write('BGspeed {} \t\t# (0 to 1, default 0.01) adaptation speed of background \n'.format(th[0]['BGspeed']))
	f.write('noise {} \t\t# (0 or 1, default 0) Use median filtering to remove noise \n'.format(th[0]['noise']))
	f.write('noise_size {} \t\t# (3 or 11, default 3) Size of median kernel \n\n'.format(th[0]['noise_size']))
	f.write('# Object Detection\n')
	f.write('peak_th_auto {}\t# (0 or 1) Automatic object detection threshold \n'.format(th[0]['peak_th_auto']))
	f.write('peak_th {} \t\t# (-inf to +inf) Manual object detection threshold \n'.format(th[0]['peak_th']))
	f.write('peak_neigh {} \t\t# (1 to inf, default 1) Minimum distance between object \n'.format(th[0]['peak_neigh']))
	f.write('peak_conv {} \t\t# Object detection kernel: 0 (none), 1 (Dif. of Gaussian), 2 (Log of Gaussian), default 1. \n'.format(th[0]['peak_conv']))
	f.write('peak_conv_size {} \t# (0 to +inf) Object typical size (pixels) \n'.format(th[0]['peak_conv_size']))
	f.write('peak_subpix {} \t# Subpixel precision method : 0 (quadratic), 1 (gaussian), default 1\n'.format(th[0]['peak_subpix']))
	f.write('peak_minima {} \t# (0 or 1) : Track dark objects (instead of bright), default 0\n\n'.format(th[0]['peak_minima']))
	f.write('# Motion Model \n')
	f.write('motion {} \t\t# (0 or 1) : Use motion model to predict object displacements, default 1\n'.format(th[0]['motion']))
	f.write('motion_av {} \t\t# (1 to +inf) : Number of neighboors over which model is averaged, default 5\n'.format(th[0]['motion_av']))
	f.write('motion_it {} \t\t# (0 to +inf) : Iterations of motion model \n'.format(th[0]['motion_it']))
	f.write('filter {} \t\t# (0 or 1) : Enable filtering of outliers \n'.format(th[0]['filter']))
	f.write('filter_std {} \t# (-inf to +inf, default 1.5) : Threshold for outliers filtering (in nb of standard deviation of motion model error) \n'.format(th[0]['filter_std']))
	f.write('filter_time {} \t# (0 to +inf, default 1) : Time speed of adaptation of outlier threshold \n\n'.format(th[0]['filter_time']))
	f.write('# Plotting \n')
	f.write('plot {} \t\t# (0 or 1) : Plot tracking process \n'.format(th[0]['plot']))
	f.write('plot_image {} \t# Plot Raw (0), Convoluted (1) or Background (2) image \n'.format(th[0]['plot_image']))
	f.write('plot_data {} \t\t# Color by velocity magnitude (0), motion model velocity (1), motion model error (2) \n'.format(th[0]['plot_data']))
	f.write('plot_data_type {} \t# Vectors (1) or scatter (2) plot \n'.format(th[0]['plot_data']))
	f.write('plot_cmin {} \t\t# (-inf,+inf) Minimum value of colorscale \n'.format(th[0]['plot_cmin']))
	f.write('plot_cmax {} \t\t# (-inf,+inf) Maximum value of colorscale \n'.format(th[0]['plot_cmax']))
	f.write('plot_alpha {} \t# (0,1) Transparency \n'.format(th[0]['plot_alpha']))
	f.close()


def set_default_parameter(th,w,h):
	iscomplete=1
# set missing parameters to default fvalues
	if not('ROIxmin' in th[0]):
		th[0]['ROIxmin']=0
		iscomplete=0
	if not('ROIxmax' in th[0]):
		th[0]['ROIxmax']=w
		iscomplete=0
	if not('ROIymin' in th[0]):
		th[0]['ROIymin']=0
		iscomplete=0
	if not('ROIymax' in th[0]):
		th[0]['ROIymax']=h
		iscomplete=0
	if not('BG' in th[0]):
		th[0]['BG']=0
		iscomplete=0
	if not('BGspeed' in th[0]):
		th[0]['BGspeed']=0.01
		iscomplete=0
	if not('noise' in th[0]):
		th[0]['noise']=0
		iscomplete=0
	if not('noise_size' in th[0]):
		th[0]['noise_size']=3
		iscomplete=0
	if not('peak_th' in th[0]):
		th[0]['peak_th']=0.02
		iscomplete=0
	if not('peak_th_auto' in th[0]):
		th[0]['peak_th_auto']=1
		iscomplete=0
	if not('peak_neigh' in th[0]):
		th[0]['peak_neigh']=1
		iscomplete=0
	if not('peak_conv_size' in th[0]):
		th[0]['peak_conv_size']=2.
		iscomplete=0
	if not('peak_conv' in th[0]):
		th[0]['peak_conv']=1
		iscomplete=0
	if not('peak_subpix' in th[0]):
		th[0]['peak_subpix']=1
		iscomplete=0
	if not('peak_minima' in th[0]):
		th[0]['peak_minima']=0
		iscomplete=0
	if not('motion' in th[0]):
		th[0]['motion']=1
		iscomplete=0
	if not('motion_av' in th[0]):
		th[0]['motion_av']=10
		iscomplete=0
	if not('motion_it' in th[0]):
		th[0]['motion_it']=0
		iscomplete=0
	if not('filter' in th[0]):
		th[0]['filter']=1
		iscomplete=0
	if not('filter_time' in th[0]):
		th[0]['filter_time']=1.0
		iscomplete=0
	if not('filter_std' in th[0]):
		th[0]['filter_std']=1.5
		iscomplete=0
	if not('vid_loop' in th[0]):
		th[0]['vid_loop']=2
		iscomplete=0
	if not('plot' in th[0]):
		th[0]['plot']=1
		iscomplete=0
	if not('plot_image' in th[0]):
		th[0]['plot_image']=1
		iscomplete=0
	if not('plot_data' in th[0]):
		th[0]['plot_data']=1
		iscomplete=0
	if not('plot_data_type' in th[0]):
		th[0]['plot_data_type']=1
		iscomplete=0
	if not('plot_cmin' in th[0]):
		th[0]['plot_cmin']=0
		iscomplete=0
	if not('plot_cmax' in th[0]):
		th[0]['plot_cmax']=5
		iscomplete=0
	if not('plot_alpha' in th[0]):
		th[0]['plot_alpha']=0.5
		iscomplete=0
	return th,iscomplete


def times_f(a,b):
#	print a.shape,b.shape
	c=a*b
	return c

def blob_detection(F,th):
# LoG or DoG detection kernel
	scale=th[0]['peak_conv_size']
	size=(int(np.maximum(3,(int(scale*3.)//2)*2+1)),int(np.maximum(3,(int(scale*3.)//2)*2+1)))
#	print(size)
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

def imProj(I,proj):
	if proj.shape[0]>=4:		
		if (proj.shape[1]!=4)|(proj.shape[0]<4):
			print('ERROR: Bad formating of _proj.txt file. See documentation')		
			return I
	# Projective transform if file found
		src_points = np.float32([proj[i,:2] for i in range(proj.shape[0])])
		dst_points = np.float32([proj[i,2:] for i in range(proj.shape[0])])
		projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
		
		# Compute image size
		pts_bnd = np.float32([[0,0],[0,I.shape[0]],[I.shape[1],I.shape[0]],[I.shape[1],0]]).reshape(-1,1,2)
		pts_bnd_= cv2.perspectiveTransform(pts_bnd, projective_matrix)
		[xmin, ymin] = np.int32(pts_bnd_.min(axis=0).ravel() - 0.5)
		[xmax, ymax] = np.int32(pts_bnd_.max(axis=0).ravel() + 0.5)
		t = [-xmin,-ymin]
		Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # Translate
		I = cv2.warpPerspective(I,Ht.dot(projective_matrix), (xmax-xmin,ymax-ymin))	
		# Only square
		#I = cv2.warpPerspective(I, projective_matrix, (int(proj[:,2].max()),int(proj[:,3].max())))
	return I

def imProc(I,th):
# Default pre-processing steps on images
# To Greyscale
	if len(I.shape)>2: I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
	if I.dtype == 'uint8': Im = np.float32(I)/256. # ALways convert to float Images
	if I.dtype == 'uint16': Im = np.float32(I)/2.**16 # ALways convert to float
	# Crop
	ROIymin=np.maximum(0,th[0]['ROIymin'])
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
		print('Warning : peak_th ({:1.4f}) is above the maximum image convoluted value ({:1.4f}).'.format(th[0]['peak_th'],a.max()))
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
	# Take pencil length in accordance with peak_conv_size
	#n_subpix=int(np.maximum(2,th[0]['peak_conv_size']))
	n_subpix=2
	if method==1: [Dx,Dy,z]=subpix2nd(np.real(np.log(a-np.min(a)+1e-8)),x,y,n_subpix) #Gaussian
	if method==0: [Dx,Dy,z]=subpix2nd(a,x,y,n_subpix) # Quadratic
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
	#pencil=np.array([-npen,0,npen])
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

