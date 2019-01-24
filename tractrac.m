function varargout = tractrac(varargin)
version='xx';
% TracTrac is GUI application for tracking a large number of objects in a movie. 
% Version: 1.5, February 2017
%
% TracTrac was initially designed for fluids, but can be used with grains, stars, birds....
% Tested on Matlab 2012b and followings 
% Author: Joris Heyman, hey.joris@gmail.com
% 
% How to run tractrac ? 
% >> tractrac
% 
% How to use tractrac ?
% a) Load a video filemenu by clicking in button 1. If the filemenu exists, it should appear in field 2 and 3 and the first frame in the visualization window. 
% 
% b) Change tracking parameters in the left panels. You can also provide a parameter filemenu with the same name as the video and the extension .txt. Parameters are (from top to bottom)
% - ROI (ROIxmin,ROIxmax,ROIymin,ROIymax) : crop the video to the region of interest defined by top left (4 & 5) and bottom right corners (6 & 7).
% - Nb. proc. (ncol,nlin,nbnd): Number of parallel windows for computing points distances.  8 is the number of columns, 9 of lines, and 10 the overlapping of windows.
% - Background Model (BG,BG_speed): Use of a background subtraction method (static zones of the image). 11 gives the relative adaptation speed of the background (values less than 1).
% - Noise Filtering (noise,noise_size): Median filter to remove noise, the size of the filter is given in 12
% - Minima: to be checked if we do not look for bright points but dark
% - Convolution (peak_conv,peak_conv_size): Select the convolution kernel to detect peak. Kernel  size is given in 13
% - Peak Neighbors (peak_neigh): Number of neighboor pixels to consider to detect peak given in 14. Values are 3, 5 or 7.
% - Intensity threshold (peak_th): only select peak that are brighter than this value (15).
% - Sub-pixel method (peak_subpix): 16 defines the fitting method to find the peak's position at a sub-ixel resolution. Default is gaussian.- Motion Model (motion,motion_av): Use of a motion model to predict peak motions. A simple constant velocity model is used.  17 gives the extent (in pixels) of the spatial averaging performed on predicted velocity vectors. ! Averaging do not cross processing windows !-Filtering Outliers (filter,filter_time,filter_std) : Filtering of bad velocity vectors according to the typical error between predicted velocity and computed velocity. 18 changes the typical adaptation time of error thresholds (default 1) and 19 the number of standard deviations above which to reject a vector.
% 
% c) Select the desired visualization among the possible choices (20). Note that color vectors might take longer to be plotted.
% 
% d) Press button Start (21). While the computation is done, you can adjust if necessary the tracking parameters to improve the result. Some live indications about the computations are given in field 3.
% 
% e) Wait for the end of the video or press Stop (22) whenever wanted. Save the computation by pressing button 23. 
% 
% f) You are now able to post-process the data. First, provide parameters of the borrom right panel:
% - fps (fps): 24 gives the number of frame per second the video was taken.
% - res (res): 25 gives the pixel resolution (in meter per pixel) .
% - Binning (binX,binY): reshape the data on a mesh with grid spaced horizontally by X (26) and vertically by Y (27), in pixels.
% - rot (rot): 28 allows you to rotate the data with a given angle (in degree), the positive direction being counterclockwise.
% Once the parameters are definde, press Post-Processing button (29) and wait for the process to complete. Finally Save then the data (30) for later use.
% 
% g) Choose a statistic to be plotted in menu 31 and press Plot button (32) to display the result. You can save the figure in png by simply pressing Print (33). You can now access a previous computation by loading first the video and go directly to step g).
% 
% h) If you want to treat several video files in a row, Press Process FileList (34) and load a text filemenu with the path of all videos, one per line. It is then recommanded to prepare parameter files for each video.
% 
% Enjoy and improve !

% Last Modified by GUIDE v2.5 23-Jan-2019 13:37:31

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @tractrac_OpeningFcn, ...
                   'gui_OutputFcn',  @tractrac_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before tractrac is made visible.
function tractrac_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to tractrac (see VARARGIN)

% Choose default command line output for tractrac
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% screen welcom screen
if exist('tractrac.png')
I=imread('tractrac.png');
imshow(I,'Parent',handles.axes1);
end

% set defaut filemenu path to current folder
% fpath=pwd;
% if exist(fpath)
% set(handles.Load_vid,'String',fpath);
% else
% set(handles.Load_vid,'String','/');
% end
% drawnow

%Prevent a bug Segmentation fault in matlab when calling rand
r=rand(1000,1000);
clear r
% UIWAIT makes tractrac wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = tractrac_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
%set(gcf, 'units','normalized','outerposition',[0 0 1 1]);


% --- Executes on button press in Load_vid.
function FileName_Callback(hObject, eventdata, handles)
% hObject    handle to Load_vid (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of Load_vid

function Im=imProc(I,th,ROI)
% Check image bit depth
if isa(I,'uint16')
bitdepth=2.^16;
else
bitdepth=2.^8;
end
% In case of color images : take mean
if length(size(I))>2,
I=mean(I,3);
end
% Back to floating point precision
Im=double(I)/bitdepth;
Im=crop(Im,ROI);
% Invert images ifblack dots are searched
% if th.MinimaOn 
% Im=1-Im;
% end
% Filtering
if th.noise == 1, Im = medfilt2(Im,[th.noise_size th.noise_size]);
end

function I=crop(I0,ROI)
I=I0(ROI(1,2):ROI(2,2),ROI(1,1):ROI(2,1));

function th=read_parameters(hObject,eventdata,handles)
% Video loops
th.vid_loop=str2double(get(handles.vid_loop,'String')); 

% Image processing
th.PlotOn=get(handles.PlotOn,'Value'); 
th.PlotMotion=get(handles.PlotMotion,'Value'); 
%th.PlotColor=get(handles.PlotOn,'Value'); 
th.PlotPoints=get(handles.PlotPoints,'Value'); 
%th.PlotVectors=get(handles.PlotVectors,'Value'); 
th.PlotError=get(handles.PlotError,'Value'); 

th.CMin=str2double(get(handles.CMin,'String'));
th.CMax=str2double(get(handles.CMax,'String'));
th.PlotScale=str2double(get(handles.PlotScale,'String'));
th.noise=get(handles.NoiseOn,'Value');
th.noise_size=max(1,round(str2double(get(handles.noisefilt,'String'))));

th.BgOn=get(handles.BgOn,'Value');
th.bw=str2double(get(handles.bw,'String')); % Threshold for peak detection
th.BWAuto=get(handles.BWAuto,'Value'); 
th.BGspeed=str2double(get(handles.BGspeed,'String')); % Background adaptation speed

% Peak detection
ConvKern=get(handles.ConvKern,'String');
th.ConvKern=ConvKern(get(handles.ConvKern,'Value'),:); 
th.KerSize=str2double(get(handles.KerSize,'String'));
th.neigh=1+2*str2double(get(handles.neigh,'String')); % nb of Neighboors in peak detection (3 5 7...)  
subpix=get(handles.subpix,'String');
th.subpix=subpix(get(handles.subpix,'Value'),:);
th.subpix=th.subpix{1};
th.MinimaOn=get(handles.MinimaOn,'Value'); % Find minimum instead of maximums

% % Motion model
th.motion=get(handles.MotionOn,'Value');  
th.motion_av=max(0.01,str2double(get(handles.MotionFilt,'String'))); % Averaging distance in motion model
th.motion_it=str2double(get(handles.motion_it,'String'));

% % Filtering
th.filter=get(handles.OutliersOn,'Value');
th.filter_time=max(0.01,str2double(get(handles.udt,'String'))); % Adaptation speed of typical error
th.filter_std=str2double(get(handles.nstd,'String')); % n std max error

% --- Executes on button press in Start.
function Start_Callback(hObject, eventdata, handles)
clc
%set(handles.Info,'String',sprintf('Starting ... Please wait.'));
set(handles.Save,'Enable','On');
set(handles.SaveASCII,'Enable','On');

drawnow 

set(gcf,'PaperPositionMode','auto');
set(gcf,'InvertHardcopy','off');

%Get data
data=guidata(hObject);

path=data.path;
if ~isfield(data,'cam')
    nFrames=data.nFrames;
    vid1=data.vid1;
else
    nFrames=1000;
end

data.infos_all=[];
dt=ones(nFrames-1,1);

data.StopOn=0;
if isfield(data,'recROI'); data=rmfield(data,'recROI'); end
if isfield(data,'Prot'); data=rmfield(data,'Prot'); end
guidata(hObject,data);

if isfield(data,'cam'), 
I00=snapshot(data.cam);
elseif data.isvid,
I00=read(vid1,1);
else
I00=imread([path '/' vid1.flist(1).name]);
end

ImW=size(I00,2);
ImH=size(I00,1);


%%% Region of Interest
xmin=max(1,floor(str2double(get(handles.ROIxmin,'String'))));
ymin=max(1,floor(str2double(get(handles.ROIymin,'String'))));
xmax=min(ImW,floor(str2double(get(handles.ROIxmax,'String'))));
ymax=min(ImH,floor(str2double(get(handles.ROIymax,'String'))));
ROI=[xmin ymin;xmax ymax];

w=diff(ROI(:,1))+1;
h=diff(ROI(:,2))+1;

th=read_parameters(hObject,eventdata,handles);


%%  INITIALISATION
STD=[];

infos_all=[];

% Read first 2 images
if isfield(data,'cam'), 
I0=snapshot(data.cam);
I1=snapshot(data.cam);
elseif data.isvid
I0=read(vid1,1);
I1=read(vid1,2);
else
I0=imread([path '/' vid1.flist(1).name]);
I1=imread([path '/' vid1.flist(2).name]);
end

% Filter images
I0=imProc(I0,th,ROI);
I1=imProc(I1,th,ROI);

% Pre-initialize memory for points saving
%maxmem=1e5;
%Pts=nan(maxmem,11,'single');
Pts=[];
pts_ind=1; % initialize index of trajectory

% Initialize Background
B=zeros(size(I0));
if th.BgOn,
    
    set(handles.Info,'String','... Initializing Background ... Please wait');
    drawnow
    nbgstart=min(200,nFrames);
    % start with an average of n frames
    for i=1:nbgstart
        
    set(handles.Info,'String',sprintf('... Initializing Background ... %02i',floor(i/nbgstart*100)));
    drawnow
        if isfield(data,'cam'), 
        I=snapshot(data.cam)
        elseif data.isvid
        I=read(vid1,i);
        else
        I=imread([path '/' vid1.flist(i).name]);
        end    
        I=imProc(I,th,ROI);
        B=B+I/nbgstart;    
    end
end

% Forground
F0=I0-B;
F1=I1-B;


% Blob detection
F0f=blob_detection(F0,th);
F1f=blob_detection(F1,th);

% scale between 0 and 1
%F0f=imadj(F0f);
%F1f=imadj(F1f);

% replace intensity threshold
if th.BWAuto
 th.bw=mean(F1f(:))+0.5*std(F1f(:));
end

% get peaks and initiate centre, velocity and acc vectors
 [xm,ym,zm0]=maximaThresh(F0f,th.neigh,th.bw,th.subpix);
 C0=[xm ym];
 U0=zeros(size(C0));
 A0=zeros(size(C0));
 ID0=nan(size(C0,1),1); % Vector of id traj
 
 [xm,ym,zm1]=maximaThresh(F1f,th.neigh,th.bw,th.subpix);
 C1=[xm ym];
 U1=zeros(size(C1));
 A1=zeros(size(C1));

 errU_filt=zeros(size(C1,1),1)';
 id_traj=0;

% Initialize list
idgood=[];
errU=[];
errU_th=2;
Xmot=[];Umot=[];Xmot_temp=[];Umot_temp=[];
errmot=[];errmot_temp=[];

% Initialize Point Tree structure
t0 = KDTreeSearcher(C0);
t1 = KDTreeSearcher(C1);
[ns01,d01] = knnsearch(t0,C1,'k',1);
[ns10,d10] = knnsearch(t0,C0,'k',1);
% Unambigous choices
if size(C0,1)>0&size(C1,1)>0,
i1001=ns10(ns01)'==(1:size(C1,1));
else
i1001=[];
end

% Create Boundary Point Matrix [x y ux uy]
BND=[];
if ~isempty(data.BND)
BND=[data.BND(:,1)-ROI(1,1) data.BND(:,2)-ROI(1,2) data.BND(:,3:4)];
end

% reset current axes
%axes(handles.axes1)
cla reset
cla(handles.axes1,'reset')
set(handles.axes2,'Layer','Top');
set(handles.axes1,'Layer','Bottom');
colormap(handles.axes1,gray)
% Initialize Plot
hold(handles.axes1,'on')
handles.h_image=imshow(I1,'Parent',handles.axes1)
set(handles.axes1,'Xlim',[0,size(I1,2)]);
set(handles.axes1,'Ylim',[0,size(I1,1)]);

% Initialize scatter object
nscat_max=100;
 handles.h_scatter=gobjects(nscat_max,1);
 for i=1:nscat_max
  handles.h_scatter(i)=scatter([],[],th.PlotScale,[],'filled','Parent',handles.axes1,'MarkerFaceAlpha',0.1,'MarkerEdgeColor','None');
 end
nscat=min(nscat_max,max(1,ceil(str2double(get(handles.PlotTail,'String')))));

% Initialize suiver object
handles.h_quiver=quiver([],[],[],[],0,'r-','Parent',handles.axes1);

% Histogram object
[h,x]=hist(F1f(:),100);
%cla(handles.axes5);
cla(handles.axes5,'reset')
hold(handles.axes5,'on');
set(handles.axes5,'XColor',[1 1 1]);
set(handles.axes5,'YColor',[1 1 1]);
set(handles.axes5,'color','none');
set(handles.axes5,'color','k');
set(handles.axes5,'Yscale','log');
handles.h_histo=bar(x,h,'w','Parent',handles.axes5);
handles.h_threshold=plot([0.0,0.0],get(handles.axes5,'Ylim'),'r-','Parent',handles.axes5,'linewidth',3);
%handles.h_threshold_text=text(1,1,'','color','r','fontsize',16,'Parent',handles.axes5);
set(handles.axes5,'Xlim',get(handles.axes5,'Xlim'));
set(handles.axes5,'Ylim',get(handles.axes5,'Ylim'));

set(handles.axes1,'TickDir','in');
set(handles.axes1,'XColor',[1 1 1]);
set(handles.axes1,'YColor',[1 1 1]);

cla(handles.axes2,'reset');
set(handles.axes2,'Visible','off');
set(handles.axes2,'Layer','Top');
% color Legend set(handles.Info,'String',sprintf('Finishing... Please Wait'));
%  xcol=10:size(I00,2)/50:size(I00,2)/4;
%  colorLeg=hsv2rgb([linspace(0.7,0,length(xcol))' ones(size(xcol))' ones(size(xcol))']);
%  set(gca,'colororder',colorLeg);
%  plot([xcol;xcol],[zeros(size(xcol))+10;zeros(size(xcol))+10+size(I00,1)/30],'-','linewidth',10,'Parent',handles.axes1);
%  text(xcol(1),0,num2str(th.CMin,'%1.1f'),'color','w','fontweight','bold','fontsize',10,'Parent',handles.axes1);
%  text(xcol(end)-size(I00,2)/50,0,num2str(th.CMax,'%1.1f'),'color','w','fontweight','bold','fontsize',10,'Parent',handles.axes1);
%  text((xcol(end)+xcol(1))/2,0,'|v|','color','w','fontweight','bold','fontsize',10,'Parent',handles.axes1);
colormap(handles.axes2,jet);
set(handles.axes2,'Clim',[th.CMin th.CMax]);
cc=colorbar(handles.axes2,'Location','east','Position',[0.68 0.05 0.02 0.9],'Color','w');
set(handles.CMax,'Visible','On');
set(handles.CMin,'Visible','On');
xlabel(cc, 'Velocity (pix/frame)','Color','w');
%set(cc,'Layer','top')


%scat=scatter([],[],th.PlotScale,[],'filled','Parent',handles.axes1);
%set(handles.Info,'String',sprintf('Starting loop...'));

flag_web=0;

%%%% Top-down approach
N=1:nFrames;
if ~flag_web
		dt_temp=dt;
		for k=1:th.vid_loop
			if mod(k,2)==1
				N=[N(:)' nFrames-1:-1:1]; % Vector of consecutive frames to read
				dt_temp=[dt_temp(:)' -dt(end:-1:1)']; % Vector of consecutive times
            else
				N=[N(:)' 2:nFrames]; % Vector of consecutive frames to read
				dt_temp=[dt_tem(:)' dt(2:end)']; % Vector of consecutive times
            end
        end

		dt=dt_temp;
end


infos_all=[sprintf('0001 | Buffer Frame... \n') infos_all];
set(handles.Info,'String',infos_all);
disp('0001 | Buffer Frame...')


n=3;  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Running loop over frames
while n<=length(N)&~data.StopOn
% Read parameters at each frames
th=read_parameters(hObject,eventdata,handles);
tic

% PRE processing steps #################################################
if isfield(data,'cam')
I2=snapshot(data.cam);
elseif data.isvid
I2=read(vid1,N(n));
else
I2=imread([path '/' vid1.flist(n).name]);
end

Iraw=I2;

I2=imProc(I2,th,ROI); % Preprocesing
F=I2-B; % Remove Background
Ff=blob_detection(F,th); % blob detection

% Replace intensity threshold
if th.BWAuto
 th.bw=mean(Ff(:))+0.5*std(Ff(:));
end
% Find new peaks
[xm,ym,zm2]=maximaThresh(Ff,th.neigh,th.bw,th.subpix);
C2=[xm ym];
U2=zeros(size(C2));
A2=zeros(size(C2));
% END PREPROCESSING  #################################################

it=0;
while it<th.motion_it+1,
	if th.motion&(n<nFrames)&(length(idgood)+size(Xmot,1)>1),
	[Umotion,errU_filt,Xmot_temp,Umot_temp,errmot_temp]=Propagate_MotionModel_KdTree(C1,X1(idgood,:),um(idgood,:),errU(idgood),BND,Xmot,Umot,errmot,th);
	else
	Umotion=zeros(size(C1));
	errU_filt=zeros(size(C1,1),1);
	end

% 	if length(C2)==0,
% 		errU=[];ID=[];ISgood=[];X1=[];X0=[];um=[];A=[];t1m=[];t2=[];ns21=[];ns12=[];i_all=[];i2112=[];
% 	else
		 %Build new trees 
		t1m = KDTreeSearcher(C1+Umotion*dt(n-1));
		t2 = KDTreeSearcher(C2);
		[ns21,d21] = knnsearch(t2,C1+Umotion*dt(n-1),'k',1);
		[ns12,d12] = knnsearch(t1m,C2,'k',1);
        
     
    if size(C1,1)>0&size(C2,1)>0,   
		i1221=ns12(ns21)'==1:size(C1,1);  % Check reversibility of link C2-> C1m [C1m -> C2 ]
		i2112=ns21(ns12)'==1:size(C2,1); % For next iteration
    else
        i1221=[];
        i2112=[];
    end
    
    if size(C0,1)>0&size(C1,1)>0&size(C2,1)>0,   
		i_all=(i1001&i1221)'; % Keep only Unilateral associations on 3 steps 0 -> 1 -> 2
		% Update Trajectories positions
		X0=C0(ns01(i_all),:);
		X1=C1(i_all,:);
		X2=C2(ns21(i_all),:);

		% Velocities
		U0=(X1-X0)/dt(n-2);
		U1=(X2-X1)/dt(n-1);
		um=(U0+U1)/2.;
		if (dt(n-1)+dt(n-2))==0
			A=nan(size(U0));
        else
			A=(U1-U0)/(dt(n-1)+dt(n-2))*2.;
        end
% TO CHECK wether dt should appear or not in errU
		errU=max(-10.,log10(max((abs((Umotion(i_all,:)-um)*dt(n-1)))')'));
		% Filtering outliers
		ISgood=(errU-errU_filt(i_all))<errU_th;
		% Evolution of threshold
		errU_th=(th.filter_time*errU_th+th.filter_std*std(errU))/(th.filter_time+1.);

    else
        i_all=[];
        % Update Trajectories positions
		X0=[];X1=[];X2=[];ID=[];um=[];
		A=[];errU=[];ISgood=[];
		% Reinitialize threshold
		errU_th=2;
    end
    


	% Filter Outliers if necessary
	if th.filter
		idgood=find(ISgood);
		idbad=find(~ISgood);
	else
		if ~isempty(X1)    
			idgood=1:length(X1(:,1));
			idbad=[];
		else
		    idgood=[];
		    idbad=[];
		end
	end
	infos=sprintf('     | Motion Model it %02i - %i pts - Err. Av/Max %1.3f/%1.3f px ',it,length(idgood),10.^(mean(errU)),10.^(mean(errU+errU_th)));
	disp(infos)
	it=it+1;
end
% END Iteration step #################################################
% Keep track of last motion model used
Xmot=Xmot_temp;
Umot=Umot_temp;
errmot=errmot_temp;

% Build traj ID and replace nan
ID=ID0(ns01(i_all));
idnan=find(isnan(ID));
if length(idnan)>0, % Associate new ID
	ID(idnan)=(1:length(idnan))-1+id_traj;
	id_traj=id_traj+length(idnan);
end
% Save ID for next iteration
ID0=nan(size(C1,1),1);
ID0(i_all)=ID;

% If last loop reset IDs to 0
if (n==length(N)-nFrames) & ~(flag_web)
	ID0(i_all)=0:sum(i_all)-1;
	id_traj=sum(i_all);
end

% Save Frame Position Speed Acceleration 
%if pts_ind+length(idgood)-1<maxmem
if (n>=length(N)-nFrames+2)
	Pts(pts_ind:pts_ind+length(idgood)-1,:)=[zeros(length(idgood),1)+n-1 ID(idgood) X1(idgood,:) um(idgood,:) A(idgood,:) Umotion(idgood,:) errU_filt(idgood)];
	pts_ind=pts_ind+length(idgood);
end
% else % break
%     set(handles.Info,'String',sprintf('Max memory limit reached ! Preventive stop...'));
%     data.StopOn=1;
% end
    


% Vizualization
PlotImageType=get(handles.PlotImageType,'Value');
if get(handles.PlotOn,'Value')
switch PlotImageType
    case 1
    set(handles.h_image,'CData',I2);
    case 2
    set(handles.h_image,'CData',imadjust(F));
    case 3
    set(handles.h_image,'CData',imadjust(Ff));
    case 4
    set(handles.h_image,'CData',imadjust(B));
    case 5
     set(handles.h_image,'CData',zeros(size(Ff)));
end

if ~isempty(BND)
plot(BND(:,1),BND(:,2),'b*')
end

% Check tail length
nscat_temp=min(nscat_max,max(2,ceil(str2double(get(handles.PlotTail,'String')))));
if nscat_temp~=nscat, % if tail as changed
    for i=nscat_temp:length(handles.h_scatter)
    set(handles.h_scatter(i),'XData',[],'YData',[],'ZData',[],'CData',[]);
    end
end
nscat=nscat_temp;

color_points=[];
if ~isempty(idgood),
    if th.PlotPoints 
    Umag=(sqrt(um(idgood,1).^2+um(idgood,2).^2));
    Umag_norm=max(0,min(0.7,(Umag-th.CMin)/(th.CMax-th.CMin)));
    color_points=hsv2rgb([0.7-Umag_norm ones(size(Umag)) ones(size(Umag))]);
    end
    Emag=max(0,min(0.7,(10.^(errU_filt)-th.CMin)/(th.CMax-th.CMin)));
    if th.PlotError&th.motion&size(Emag,2)<=1,
        Emag=max(0,min(0.7,(10.^(errU(idgood))-th.CMin)/(th.CMax-th.CMin)));
        color_points=hsv2rgb([0.7-Emag ones(size(Emag)) ones(size(Emag))]);
    end
    if get(handles.PlotID,'Value')&~isempty(idgood),
        IDg=mod(ID(idgood),1000)/1000;
        color_points=hsv2rgb([IDg ones(size(IDg)) ones(size(IDg))]);
    end
end

if length(color_points)>0,
    set(handles.h_scatter(mod(n,nscat)+1),'XData',X2(idgood,1),'YData',X2(idgood,2),'ZData',zeros(size(X2(idgood,2))),'CData',color_points,'MarkerFaceAlpha',0.8);
    set(handles.h_scatter(mod(n-1,nscat)+1),'MarkerFaceAlpha',0.2);
    set(handles.h_scatter(mod(n+1,nscat)+1),'MarkerFaceAlpha',0.1);
%    uistack(handles.h_scatter(mod(n,nscat)+1),'up',1)
%    uistack(handles.h_scatter(mod(n-1,nscat)+1),'down',1)
else
    set(handles.h_scatter(mod(n,nscat)+1),'XData',[],'YData',[],'ZData',[],'CData',[]);
end

%Remove quiver model if not motion model
if th.PlotMotion&~isempty(idgood)
set(handles.h_quiver,'XData',C1(:,1),'YData',C1(:,2),'UData',(Umotion(:,1))*th.PlotScale,'VData',(Umotion(:,2))*th.PlotScale);    
else
delete(handles.h_quiver);
handles.h_quiver=quiver([],[],[],[],0,'r-','Parent',handles.axes1);
end

% Modify size if changed
if get(handles.h_scatter(1),'SizeData')~=th.PlotScale,
for i=1:length(handles.h_scatter)
set(handles.h_scatter(i),'SizeData',th.PlotScale);
end
end

% Histogram
if get(handles.histOn,'Value')
[h,x]=hist(Ff(:),100);
set(handles.h_histo,'XData',x,'YData',h);
set(handles.h_threshold,'XData',[1,1]*th.bw);
end

if get(handles.RecOn,'Value'), % Reccord
% Get defaut path
filename=data.fname;
[pathstr,name,ext] = fileparts(filename);
path_img=[pathstr '/' name '_img/'];
if exist(path_img)==0,
    mkdir(path_img)
end
% Print frame
print(gcf,[path_img 'img_' num2str(N(n),'%05i') '.jpg'],'-djpeg','-r0');
end

end

drawnow;
pause(0.001);

% Update Background
if th.BgOn
   r2=B-I2;
   r=-double(r2>th.BGspeed)*th.BGspeed+double(r2<-th.BGspeed)*th.BGspeed-double(abs(r2)<=th.BGspeed).*r2;
   B=min(max(B+r,0),1);   
else
   B=zeros(size(I2));
end

% Prepare matrix for next iteration
% 1 -> 0
C0=C1;
U0=U1;
A0=A1;
%KDTree
t0=t1;t1=t2;ns01=ns12;i1001=i2112;
zm1=zm2;
% 2-> 1
C1=C2;
U1=U2;
A1=A2;
I1=I2;

elapsed=toc;
infos= [sprintf('%04d | %i Pts (%1.0f%% recovered) | Time %1.2fs | Nb traj %1.0e |',N(n),size(C1,1),length(idgood)*100./max(0.1,size(C1,1)),elapsed,id_traj)];
infos2= [sprintf('%04d | %i Pts (%1.0f%% recovered) |\n',N(n),size(C1,1),length(idgood)*100./max(0.1,size(C1,1)))];
infos_all=[infos2 infos_all];
set(handles.Info,'String',infos_all);
disp(infos)


n=n+1;
data=guidata(hObject);
end
% END OF WHILE LOOOP


set(handles.Info,'String','... Finalizing Run ... Please wait');
drawnow

disp(sprintf('%04i | Buffer Frame.',N(n-1)))
if ~isempty(Pts)
% Remove extra lines (if any)
Pts=Pts(1:pts_ind-1,:);

% add ROI to x and y coordinates
Pts(:,3)=Pts(:,3)+ROI(1,1)-1;
Pts(:,4)=Pts(:,4)+ROI(1,2)-1;

% save data

infos_all=[sprintf('> Tracking Stopped. Please save results! \n') infos_all];
set(handles.Info,'String',infos_all);

data.infos_all=infos_all;
data.Pts=Pts;
data.Frames=n;
guidata(hObject,data);
else
infos_all=[sprintf('> Finished. No trajectories found ! \n') infos_all];
set(handles.Info,'String',infos_all);
end

function Ff=blob_detection(F,th)
scale=th.KerSize;
if strcmp(th.ConvKern{1},'DoG')
fG1=fspecial('gaussian',[1 1]*ceil(scale)*2+1,scale*0.8);
fG2=fspecial('gaussian',[1 1]*ceil(scale)*2+1,scale*1.2);
fDoG=fG2-fG1;
if th.MinimaOn
Ff=imfilter(F,fDoG,'replicate');
else  
Ff=imfilter(F,-fDoG,'replicate');
end
elseif strcmp(th.ConvKern{1},'LoG')
fG = fspecial('gaussian',[1 1]*ceil(scale)*2+1,scale);
fL = fspecial('laplacian');
if th.MinimaOn
Ff=imfilter(F,fG,'replicate','same');
else 
Ff=imfilter(F,-fG,'replicate','same');
end
Ff=imfilter(Ff,fL,'replicate','same');
else
if th.MinimaOn
Ff=-F;
else
Ff=F;
end

end

function I=imadj(I)
I=(I-min(I(:)))/(max(I(:))-min(I(:)));

% --- Executes on button press in Pause.
function Pause_Callback(hObject, eventdata, handles)
% hObject    handle to Pause (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in Stop.
function Stop_Callback(hObject, eventdata, handles)
% hObject    handle to Stop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
data=guidata(hObject);
data.StopOn=1;
guidata(hObject,data);


% --- Executes on button press in SaveASCII.
function SaveASCII_Callback(hObject, eventdata, handles)
% hObject    handle to Save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
data=guidata(hObject);
if isfield(data,'Pts'),
 filename=[data.path '/' data.fname '_track.txt'];
 set(handles.Info,'String','... Saving ... Please wait');
 drawnow
 Frames=data.Frames;
 th=read_parameters(hObject,eventdata,handles);
 %save(Load_vid,'th','Pts','Frames','-v7.3');
 fid=fopen(filename,'w');
 fprintf(fid,['# Raw Tracking Data File. Computed with TracTrac (Matlab), the ' date '\n']);
 fprintf(fid,['# Parameters: see parameter file. \n']);
 fprintf(fid,['# Frame ID x y Vx Vy Ax Ay Vx(motion) Vy(motion) Error(motion) \n']);
 for i=1:size(data.Pts,1)
     fprintf(fid, '%i %i %.2f %.2f %.3f %.3f %.4f %.4f %.3f %.3f %.3f\n',data.Pts(i,:));
 end
 fclose(fid);
 
 data.infos_all=[sprintf('> Raw data saved to %s_track.txt \n',data.fname) data.infos_all];
 set(handles.Info,'String',data.infos_all);
 disp(sprintf('Raw Data saved to  %s',filename))
 rmfield(data,'Pts'); % remove field to save memory
 guidata(hObject,data);
 set(handles.PostMenu,'Enable','On');
else
 set(handles.Info,'String',sprintf('No data to save !'));
end
 
% --- Executes on button press in Save.
function Save_Callback(hObject, eventdata, handles)
% hObject    handle to Save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
data=guidata(hObject);
if isfield(data,'Pts'),
 filename=data.fname;
 filename=[data.path '/' data.fname '_track'];
 set(handles.Info,'String','... Saving ... Please wait');
 drawnow
 Frames=data.Frames;
 Pts=data.Pts;
 th=read_parameters(hObject,eventdata,handles);
 save(filename,'th','Pts','Frames','-v7.3');
 data.infos_all=[sprintf('> Raw data saved to %s_track.mat \n',data.fname) data.infos_all];
 set(handles.Info,'String',data.infos_all);
 disp(sprintf('> Raw Data saved to  %s.mat',filename))
 rmfield(data,'Pts'); % remove field to save memory
 guidata(hObject,data);
 set(handles.PostMenu,'Enable','On');
else
 set(handles.Info,'String',sprintf('No data to save !'));
 end
    

% --- Executes on button press in PlotOn.
function PlotOn_Callback(hObject, eventdata, handles)
% hObject    handle to PlotOn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hint: get(hObject,'Value') returns toggle state of PlotOn


% --- Executes on button press in BgOn.
function BgOn_Callback(hObject, eventdata, handles)
% hObject    handle to BgOn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
A=get(handles.BgOn,'Value');
if A
    set(handles.text_BGspeed,'Visible','on');
    set(handles.BGspeed,'Visible','on');
else
    set(handles.text_BGspeed,'Visible','off');
    set(handles.BGspeed,'Visible','off');
end
% Hint: get(hObject,'Value') returns toggle state of BgOn


% --- Executes on button press in MotionOn.
function MotionOn_Callback(hObject, eventdata, handles)
% hObject    handle to MotionOn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of MotionOn
A=get(handles.MotionOn,'Value');
if A
    set(handles.text_MotionFilt,'Visible','on');
    set(handles.MotionFilt,'Visible','on');
else
    set(handles.text_MotionFilt,'Visible','off');
    set(handles.MotionFilt,'Visible','off');
end

% --- Executes on button press in NoiseOn.
function NoiseOn_Callback(hObject, eventdata, handles)
% hObject    handle to NoiseOn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
A=get(handles.NoiseOn,'Value');
if A
    set(handles.text_noisefilt,'Visible','on');
    set(handles.noisefilt,'Visible','on');
else
    set(handles.text_noisefilt,'Visible','off');
    set(handles.noisefilt,'Visible','off');
end

% Hint: get(hObject,'Value') returns toggle state of NoiseOn


% --- Executes on selection change in ConvKern.
function ConvKern_Callback(hObject, eventdata, handles)
% hObject    handle to ConvKern (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns ConvKern contents as cell array
%        contents{get(hObject,'Value')} returns selected item from ConvKern


% --- Executes during object creation, after setting all properties.
function ConvKern_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ConvKern (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function KerSize_Callback(hObject, eventdata, handles)
% hObject    handle to KerSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of KerSize as text
%        str2double(get(hObject,'String')) returns contents of KerSize as a double

nf=max(0.01,(str2double(get(handles.KerSize,'String'))));
set(handles.KerSize,'String',num2str(nf));


% --- Executes during object creation, after setting all properties.
function KerSize_CreateFcn(hObject, eventdata, handles)
% hObject    handle to KerSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function BGspeed_Callback(hObject, eventdata, handles)
% hObject    handle to BGspeed (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


nf=max(0,str2double(get(handles.BGspeed,'String')));
set(handles.BGspeed,'String',num2str(nf));

% Hints: get(hObject,'String') returns contents of BGspeed as text
%        str2double(get(hObject,'String')) returns contents of BGspeed as a double


% --- Executes during object creation, after setting all properties.
function BGspeed_CreateFcn(hObject, eventdata, handles)
% hObject    handle to BGspeed (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function MotionFilt_Callback(hObject, eventdata, handles)
% hObject    handle to MotionFilt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

nf=max(0.1,str2double(get(handles.MotionFilt,'String')));
set(handles.MotionFilt,'String',num2str(nf,'%i'));
% Hints: get(hObject,'String') returns contents of MotionFilt as text
%        str2double(get(hObject,'String')) returns contents of MotionFilt as a double


% --- Executes during object creation, after setting all properties.
function MotionFilt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to MotionFilt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function bw_Callback(hObject, eventdata, handles)
% hObject    handle to bw (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of bw as text
%        str2double(get(hObject,'String')) returns contents of bw as a double


% --- Executes during object creation, after setting all properties.
function bw_CreateFcn(hObject, eventdata, handles)
% hObject    handle to bw (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function neigh_Callback(hObject, eventdata, handles)
% hObject    handle to neigh (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

nf=max(1,round(str2double(get(handles.neigh,'String'))));
set(handles.neigh,'String',num2str(nf,'%i'));
% Hints: get(hObject,'String') returns contents of neigh as text
%        str2double(get(hObject,'String')) returns contents of neigh as a double


% --- Executes during object creation, after setting all properties.
function neigh_CreateFcn(hObject, eventdata, handles)
% hObject    handle to neigh (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function udt_Callback(hObject, eventdata, handles)
% hObject    handle to udt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of udt as text
%        str2double(get(hObject,'String')) returns contents of udt as a double
nf=max(1,str2double(get(handles.udt,'String')));
set(handles.udt,'String',num2str(nf,'%i'));

% --- Executes during object creation, after setting all properties.
function udt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to udt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function nstd_Callback(hObject, eventdata, handles)
% hObject    handle to nstd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of nstd as text
%        str2double(get(hObject,'String')) returns contents of nstd as a double


% --- Executes during object creation, after setting all properties.
function nstd_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nstd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in OutliersOn.
function OutliersOn_Callback(hObject, eventdata, handles)
% hObject    handle to OutliersOn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
A=get(handles.OutliersOn,'Value');
if A
    set(handles.text_udt,'Visible','on');
    set(handles.udt,'Visible','on');
    set(handles.text_nstd,'Visible','on');
    set(handles.nstd,'Visible','on');
else
    set(handles.text_udt,'Visible','off');
    set(handles.udt,'Visible','off');
    set(handles.text_nstd,'Visible','off');
    set(handles.nstd,'Visible','off');
end
% Hint: get(hObject,'Value') returns toggle state of OutliersOn


% --- Executes on selection change in subpix.
function subpix_Callback(hObject, eventdata, handles)
% hObject    handle to subpix (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns subpix contents as cell array
%        contents{get(hObject,'Value')} returns selected item from subpix


% --- Executes during object creation, after setting all properties.
function subpix_CreateFcn(hObject, eventdata, handles)
% hObject    handle to subpix (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% Function to load parameter filemenu and save it into a structure par
function par=load_parfile(file)
    par=struct();
    fid=fopen(file,'r');
    tline = fgets(fid);
    while ischar(tline)
        %disp(tline);
        if tline(1)~='#';
        C = textscan(tline,'%s %f');
        if ~isempty(C{1,1})&~isempty(C{1,2})
        if isstr(C{1}{:})&isnumeric(C{2})&~isnan(C{2})
        eval(['par.' C{1}{:} '=' num2str(C{2}) ';']);
        end
        end
        end
        tline = fgets(fid);
    end
    disp(sprintf('Parameter file %s loaded !',file))
    fclose(fid);
   

% --- Executes on button press in SavePar.
function SavePar_Callback(hObject, eventdata, handles)
% Save parameters in txt filemenu
data=guidata(hObject);
if isfield(data,'fname')
filename=data.fname;
filename=[data.path '/' data.fname '_par.txt'];
fid=fopen(filename,'w'); % if parameter file exist, overwrite
fprintf(fid,'# Parameter file generated by TracTrac Matlab v2.0\n\n')
% get parameters;
fps=1;res=1;binX=1;binY=1;rot=0;
if isfield(data,'par')
    if isfield(data.par,'fps'); fps=data.par.fps; end
    if isfield(data.par,'res'); res=data.par.res; end
    if isfield(data.par,'binX'); binX=data.par.binX; end
    if isfield(data.par,'binY'); binY=data.par.binY; end
    if isfield(data.par,'rot'); rot=data.par.rot; end
end

    fprintf(fid,'# Video Loops #\n');
    fprintf(fid,['vid_loop ' num2str(get(handles.vid_loop,'Value')) '\n']);
    fprintf(fid,'# Image Processing #\n');
    fprintf(fid,['ROIxmin ' get(handles.ROIxmin,'String') '\n']); 
    fprintf(fid,['ROIxmax ' get(handles.ROIxmax,'String') '\n']); 
    fprintf(fid,['ROIymin ' get(handles.ROIymin,'String') '\n']); 
    fprintf(fid,['ROIymax ' get(handles.ROIymax,'String') '\n']);
    fprintf(fid,['BG ' num2str(get(handles.BgOn,'Value')) '\n']);
    fprintf(fid,['BG_speed ' get(handles.BGspeed,'String') '\n']); 
    fprintf(fid,['noise ' num2str(get(handles.NoiseOn,'Value')) '\n']); 
    fprintf(fid,['noise_size ' get(handles.noisefilt,'String') '\n\n']); 

    fprintf(fid,'# Peak detection #\n');
    fprintf(fid,['peak_th ' get(handles.bw,'String') '\n']);
    fprintf(fid,['peak_th_auto ' num2str(get(handles.BWAuto,'Value')) '\n']);
    fprintf(fid,['peak_neigh ' get(handles.neigh,'String') '\n']); 
    fprintf(fid,['peak_conv_size ' get(handles.KerSize,'String') '\n']); 
    fprintf(fid,['peak_conv ' num2str(get(handles.ConvKern,'Value')) '\n']); 
    fprintf(fid,['peak_subpix ' num2str(get(handles.subpix,'Value')) '\n']); 
    fprintf(fid,['peak_minima ' num2str(get(handles.MinimaOn,'Value')) '\n\n']); 

    fprintf(fid,'# Motion Model #\n');
    fprintf(fid,['motion ' num2str(get(handles.MotionOn,'Value')) '\n']);
    fprintf(fid,['motion_it ' get(handles.motion_it,'String') '\n']); 
    fprintf(fid,['motion_steady ' num2str(get(handles.motion_steady,'Value')) '\n']); 
    fprintf(fid,['motion_av ' get(handles.MotionFilt,'String') '\n\n']); 

    fprintf(fid,'# Filter Outliers #\n');
    fprintf(fid,['filter ' num2str(get(handles.OutliersOn,'Value')) '\n']); 
    fprintf(fid,['filter_time ' get(handles.udt,'String') '\n']); 
    fprintf(fid,['filter_std ' get(handles.nstd,'String') '\n\n']); 

    fprintf(fid,'# Post Processing #\n');
    fprintf(fid,['fps ' num2str(fps) '\n']); 
    fprintf(fid,['res ' num2str(res) '\n']); 
    fprintf(fid,['binX ' num2str(binX) '\n']); 
    fprintf(fid,['binY ' num2str(binY) '\n']); 
    fprintf(fid,['rot ' num2str(rot) '\n']); 
fclose(fid);
set(handles.Info,'String',sprintf('Parameter file saved !'));
    disp(sprintf('Parameter file %s saved !',filename))
end

    
% Function to load video filemenu
function load_img(hObject, eventdata, handles)

data=guidata(hObject);
if isfield(data,'cam'), delete(data.cam); data=rmfield(data,'cam'); end

filename=data.fname;
path=data.path;
ext=data.ext;
flist=dir([path '/*' ext]);
vid1.flist=flist;
nFrames=length(flist);
I00=imread([path '/' flist(1).name]);


% reset current axes
axes(handles.axes1)
cla reset
imshow(I00, 'Parent', handles.axes1);
set(handles.axes1,'TickDir','in');
set(handles.axes1,'XColor',[1 1 1]);
set(handles.axes1,'YColor',[1 1 1]);
axis on
% Defaut values
    set(handles.ROIxmin,'String','1');
    set(handles.ROIxmax,'String',num2str(size(I00,2),'%i'));
    set(handles.ROIymin,'String','1');
    set(handles.ROIymax,'String',num2str(size(I00,1),'%i'));
    drawnow
% Load Parameter filemenu if exist
par=struct();
filename_par=[data.path '/' data.fname '_par.txt'];
if exist(filename_par)
    par=load_parfile(filename_par);
    % Update handles with par data
    if isfield(par,'vid_loop'), set(handles.vid_loop,'String',num2str(par.vid_loop)); end
    if isfield(par,'ROIxmin'), set(handles.ROIxmin,'String',num2str(par.ROIxmin)); end
    if isfield(par,'ROIxmax'), set(handles.ROIxmax,'String',num2str(par.ROIxmax)); end
    if isfield(par,'ROIymin'), set(handles.ROIymin,'String',num2str(par.ROIymin)); end
    if isfield(par,'ROIymax'), set(handles.ROIymax,'String',num2str(par.ROIymax)); end
    if isfield(par,'fps'), par.fps ; end
    if isfield(par,'res'),  par.res ; end
    if isfield(par,'peak_th'), set(handles.bw,'String',num2str(par.peak_th)); end
    if isfield(par,'peak_th_auto'), set(handles.BWAuto,'Value',par.peak_th_auto==1); end
    if isfield(par,'peak_neigh'), set(handles.neigh,'String',num2str(par.peak_neigh)); end
    if isfield(par,'peak_conv_size'), set(handles.KerSize,'String',num2str(par.peak_conv_size)); end
    if isfield(par,'peak_conv'), set(handles.ConvKern,'Value',par.peak_conv); end
    if isfield(par,'peak_subpix'), set(handles.subpix,'Value',par.peak_subpix); end
    if isfield(par,'peak_minima'), set(handles.MinimaOn,'Value',par.peak_minima==1); set(handles.MinimaOff,'Value',par.peak_minima==0); end
    if isfield(par,'motion'), set(handles.MotionOn,'Value',par.motion==1); end
    if isfield(par,'motion_steady'), set(handles.motion_steady,'Value',par.motion_steady);
        set(handles.motion_unsteady,'Value',~par.motion_steady); end
    if isfield(par,'motion_it'), set(handles.motion_it,'String',num2str(par.motion_it)); end
    if isfield(par,'motion_av'), set(handles.MotionFilt,'String',num2str(par.motion_av)); end
    if isfield(par,'rot'), par.rot; end
    if isfield(par,'noise_size'), set(handles.noisefilt,'String',num2str(par.noise_size)); end
    if isfield(par,'noise'), set(handles.NoiseOn,'Value',par.noise==1); end
    if isfield(par,'binX'), par.binX; end
    if isfield(par,'binY'), par.binY; end
    if isfield(par,'BG_speed'), set(handles.BGspeed,'String',num2str(par.BG_speed)); end
    if isfield(par,'BG'), set(handles.BgOn,'Value',par.BG==1); end
    if isfield(par,'filter_time'), set(handles.udt,'String',num2str(par.filter_time)); end
    if isfield(par,'filter_std'), set(handles.nstd,'String',num2str(par.filter_std)); end
    if isfield(par,'filter'), set(handles.OutliersOn,'Value',par.filter==1); end

    set(handles.Info,'String',sprintf('Found %i frames to process! \n Custom parameters read from *_par.txt file. \n',nFrames));

    drawnow
else
    set(handles.Info,'String',sprintf('Found %i frames to process! \n Default parameters were loaded. \n',nFrames));

end


%data=struct();

% FileMenu for boundary conditions
filename_Bnd=[data.path '/' data.fname '_bnd.csv'];
BND=[];
if exist(filename_Bnd)
BND=dlmread(filename_Bnd,',',0,0);
end
data.BND=BND;

% Plot ROI rectangle²
xmin=max(1,floor(str2double(get(handles.ROIxmin,'String'))));
ymin=max(1,floor(str2double(get(handles.ROIymin,'String'))));
xmax=floor(str2double(get(handles.ROIxmax,'String')));
ymax=floor(str2double(get(handles.ROIymax,'String')));
ROI=[xmin ymin;xmax ymax];
rec=[ROI(1,1) ROI(1,2) ROI(2,1)-ROI(1,1) ROI(2,2)-ROI(1,2)];

%if isfield(data,'recROI'), delete(data.recROI); end
data.recROI=rectangle('Position',rec,'edgecolor','g','linestyle','-','Parent',handles.axes1);

% Save data
data.vid1=vid1;
data.nFrames=nFrames;
data.par=par;
guidata(hObject,data);

% Function to load video filemenu
function load_video(hObject, eventdata, handles)

data=guidata(hObject);

if isfield(data,'cam'), delete(data.cam); data=rmfield(data,'cam'); end

filename=[data.path '/' data.fname data.ext];
disp(filename)
if exist(filename)
vid1 = VideoReader(filename);
nFrames=vid1.NumberOfFrames;
I00=read(vid1,1);


% reset current axes
axes(handles.axes1)
cla reset
imshow(I00, 'Parent', handles.axes1);
set(handles.axes1,'TickDir','in');
set(handles.axes1,'XColor',[1 1 1]);
set(handles.axes1,'YColor',[1 1 1]);
axis on
% Defaut values
    set(handles.ROIxmin,'String','1');
    set(handles.ROIxmax,'String',num2str(size(I00,2),'%i'));
    set(handles.ROIymin,'String','1');
    set(handles.ROIymax,'String',num2str(size(I00,1),'%i'));
    drawnow
% Load Parameter filemenu if exist
par=struct();
filename_par=[data.path '/' data.fname '_par.txt'];
if exist(filename_par)
    par=load_parfile(filename_par);
    % Update handles with par data
    if isfield(par,'vid_loop'), set(handles.vid_loop,'String',num2str(par.vid_loop)); end
    if isfield(par,'ROIxmin'), set(handles.ROIxmin,'String',num2str(par.ROIxmin)); end
    if isfield(par,'ROIxmax'), set(handles.ROIxmax,'String',num2str(par.ROIxmax)); end
    if isfield(par,'ROIymin'), set(handles.ROIymin,'String',num2str(par.ROIymin)); end
    if isfield(par,'ROIymax'), set(handles.ROIymax,'String',num2str(par.ROIymax)); end
    if isfield(par,'peak_th'), set(handles.bw,'String',num2str(par.peak_th)); end
    if isfield(par,'peak_th_auto'), set(handles.BWAuto,'Value',par.peak_th_auto==1); end
    if isfield(par,'peak_neigh'), set(handles.neigh,'String',num2str(par.peak_neigh)); end
    if isfield(par,'peak_conv_size'), set(handles.KerSize,'String',num2str(par.peak_conv_size)); end
    if isfield(par,'peak_conv'), set(handles.ConvKern,'Value',par.peak_conv); end
    if isfield(par,'peak_subpix'), set(handles.subpix,'Value',par.peak_subpix); end
    if isfield(par,'peak_minima'), set(handles.MinimaOn,'Value',par.peak_minima==1); set(handles.MinimaOff,'Value',par.peak_minima==0); end
    if isfield(par,'motion'), set(handles.MotionOn,'Value',par.motion==1); end
    if isfield(par,'motion_av'), set(handles.MotionFilt,'String',num2str(par.motion_av)); end
    if isfield(par,'motion_it'), set(handles.motion_it,'String',num2str(par.motion_it)); end
    if isfield(par,'motion_steady'), set(handles.motion_steady,'Value',par.motion_steady);  set(handles.motion_unsteady,'Value',~par.motion_steady); end
    if isfield(par,'noise_size'), set(handles.noisefilt,'String',num2str(par.noise_size)); end
    if isfield(par,'noise'), set(handles.NoiseOn,'Value',par.noise==1); end
    if isfield(par,'BG_speed'), set(handles.BGspeed,'String',num2str(par.BG_speed)); end
    if isfield(par,'BG'), set(handles.BgOn,'Value',par.BG==1); end
    if isfield(par,'filter_time'), set(handles.udt,'String',num2str(par.filter_time)); end
    if isfield(par,'filter_std'), set(handles.nstd,'String',num2str(par.filter_std)); end
    if isfield(par,'filter'), set(handles.OutliersOn,'Value',par.filter==1); end

    set(handles.Info,'String',sprintf(' Found %i frames to process. \n Custom parameters read from existing par file. \n\n >> Press Start Button to start tracking!',nFrames));

    drawnow
else
    set(handles.Info,'String',sprintf(' Found %i frames to process. \n Default parameters were loaded. \n\n >> Press Start Button to start tracking!',nFrames));

end


%data=struct();

% FileMenu for boundary conditions
filename_Bnd=[data.path '/' data.fname '_bnd.csv'];
BND=[];
if exist(filename_Bnd)
BND=dlmread(filename_Bnd,',',0,0);
end
data.BND=BND;

% Plot ROI rectangle²
xmin=max(1,floor(str2double(get(handles.ROIxmin,'String'))));
ymin=max(1,floor(str2double(get(handles.ROIymin,'String'))));
xmax=floor(str2double(get(handles.ROIxmax,'String')));
ymax=floor(str2double(get(handles.ROIymax,'String')));
ROI=[xmin ymin;xmax ymax];
rec=[ROI(1,1) ROI(1,2) ROI(2,1)-ROI(1,1) ROI(2,2)-ROI(1,2)];

%if isfield(data,'recROI'), delete(data.recROI); end
data.recROI=rectangle('Position',rec,'edgecolor','g','linestyle','-','Parent',handles.axes1);

% Save data
data.vid1=vid1;
data.nFrames=nFrames;
data.par=par;
guidata(hObject,data);

else
set(handles.Info,'String',sprintf('!!! Cannot load video file !!!'));
end


function Load_vid_Callback(hObject, eventdata, handles)
% Load Video Object
data=guidata(hObject);
[file, path] = uigetfile({'*.avi;*.mj2;*.mp4;*.tif;*.tiff;*.png;*.jpg;*.jpeg','Video/Images Files'});
filename=[path file];
[path, name, ext] = fileparts(filename);
data.path=path;data.fname=name;data.ext=ext;
if sum(strcmpi(ext,{'.avi','.mj2','.mp4'}))>0
    data.isvid=1; guidata(hObject,data);
    load_video(hObject, eventdata, handles);
    set(handles.TrackMenu,'Enable','On');
    set(handles.Start,'Enable','On');
    set(handles.Stop,'Enable','On');
    if exist([path '/' data.fname '_track.mat']) || exist([path '/' data.fname '_track.txt']) 
    set(handles.PostMenu,'Enable','On');
    end
    if exist([path '/' data.fname '_PostProc.mat']) 
    set(handles.PlotMenu,'Enable','On');
    end
elseif sum(strcmpi(ext,{'.tiff','.tif','.png','.jpg','.jpeg'}))>0
    data.isvid=0; data.fname=[ext(2:end) '_seq']; guidata(hObject,data);
    load_img(hObject, eventdata, handles);
    set(handles.TrackMenu,'Enable','On');
    set(handles.Start,'Enable','On');
    set(handles.Stop,'Enable','On');
    if exist([path '/' data.fname '_track.mat']) || exist([path '/' data.fname '_track.txt']) 
    set(handles.PostMenu,'Enable','On');
    end
    if exist([path '/' data.fname '_PostProc.mat']) 
    set(handles.PlotMenu,'Enable','On');
    end
else
    disp('Error, file unknown')
end

% Hints: get(hObject,'String') returns contents of Load_vid as text
%        str2double(get(hObject,'String')) returns contents of Load_vid as a double

% --- Executes during object creation, after setting all properties.
function Load_vid_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Load_vid (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Info_Callback(hObject, eventdata, handles)
% hObject    handle to Info (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% --- Executes during object creation, after setting all properties.
function Info_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Info (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function ROIxmin_Callback(hObject, eventdata, handles)
% hObject    handle to ROIxmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ROIxmin as text
%        str2double(get(hObject,'String')) returns contents of ROIxmin as a double
data=guidata(hObject);
xmin=max(1,floor(str2double(get(handles.ROIxmin,'String'))));
ymin=max(1,floor(str2double(get(handles.ROIymin,'String'))));
xmax=floor(str2double(get(handles.ROIxmax,'String')));
ymax=floor(str2double(get(handles.ROIymax,'String')));
ROI=[xmin ymin;xmax ymax];
rec=[ROI(1,1) ROI(1,2) ROI(2,1)-ROI(1,1) ROI(2,2)-ROI(1,2)];
if isfield(data,'recROI'), delete(data.recROI);rmfield(data,'recROI'); end
data.recROI=rectangle('Position',rec,'edgecolor','g','linestyle','-','Parent',handles.axes1);
drawnow;
guidata(hObject,data);

% --- Executes during object creation, after setting all properties.
function ROIxmin_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ROIxmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function ROIxmax_Callback(hObject, eventdata, handles)
% hObject    handle to ROIxmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ROIxmax as text
%        str2double(get(hObject,'String')) returns contents of ROIxmax as a double
data=guidata(hObject);
xmin=max(1,floor(str2double(get(handles.ROIxmin,'String'))));
ymin=max(1,floor(str2double(get(handles.ROIymin,'String'))));
xmax=floor(str2double(get(handles.ROIxmax,'String')));
ymax=floor(str2double(get(handles.ROIymax,'String')));
ROI=[xmin ymin;xmax ymax];
rec=[ROI(1,1) ROI(1,2) ROI(2,1)-ROI(1,1) ROI(2,2)-ROI(1,2)];

if isfield(data,'recROI'), delete(data.recROI);rmfield(data,'recROI'); end
data.recROI=rectangle('Position',rec,'edgecolor','g','linestyle','-','Parent',handles.axes1);
drawnow;
guidata(hObject,data);

% --- Executes during object creation, after setting all properties.
function ROIxmax_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ROIxmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function ROIymin_Callback(hObject, eventdata, handles)
% hObject    handle to ROIymin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ROIymin as text
%        str2double(get(hObject,'String')) returns contents of ROIymin as a double

data=guidata(hObject);
xmin=max(1,floor(str2double(get(handles.ROIxmin,'String'))));
ymin=max(1,floor(str2double(get(handles.ROIymin,'String'))));
xmax=floor(str2double(get(handles.ROIxmax,'String')));
ymax=floor(str2double(get(handles.ROIymax,'String')));
ROI=[xmin ymin;xmax ymax];
rec=[ROI(1,1) ROI(1,2) ROI(2,1)-ROI(1,1) ROI(2,2)-ROI(1,2)];

if isfield(data,'recROI'), delete(data.recROI); rmfield(data,'recROI'); end
data.recROI=rectangle('Position',rec,'edgecolor','g','linestyle','-','Parent',handles.axes1);
drawnow;
guidata(hObject,data);

% --- Executes during object creation, after setting all properties.
function ROIymin_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ROIymin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function ROIymax_Callback(hObject, eventdata, handles)
% hObject    handle to ROIymax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ROIymax as text
%        str2double(get(hObject,'String')) returns contents of ROIymax as a double
data=guidata(hObject);
xmin=max(1,floor(str2double(get(handles.ROIxmin,'String'))));
ymin=max(1,floor(str2double(get(handles.ROIymin,'String'))));
xmax=floor(str2double(get(handles.ROIxmax,'String')));
ymax=floor(str2double(get(handles.ROIymax,'String')));
ROI=[xmin ymin;xmax ymax];
if isfield(data,'recROI'), delete(data.recROI); rmfield(data,'recROI'); end
rec=[ROI(1,1) ROI(1,2) ROI(2,1)-ROI(1,1) ROI(2,2)-ROI(1,2)];
data.recROI=rectangle('Position',rec,'edgecolor','g','linestyle','-','Parent',handles.axes1);
drawnow;
guidata(hObject,data);

% --- Executes during object creation, after setting all properties.
function ROIymax_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ROIymax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Nlin_Callback(hObject, eventdata, handles)
% hObject    handle to ncol (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ncol as text
%        str2double(get(hObject,'String')) returns contents of ncol as a double
% 
nf=max(1,round(str2double(get(handles.Nlin,'String'))));
set(handles.Nlin,'String',num2str(nf,'%i'));

% --- Executes during object creation, after setting all properties.
function Nlin_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ncol (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Ncol_Callback(hObject, eventdata, handles)
% hObject    handle to Ncol (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% 
nf=max(1,round(str2double(get(handles.Ncol,'String'))));
set(handles.Ncol,'String',num2str(nf,'%i'));

% Hints: get(hObject,'String') returns contents of Ncol as text
%        str2double(get(hObject,'String')) returns contents of Ncol as a double
% --- Executes during object creation, after setting all properties.
function Ncol_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Ncol (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Nbnd_Callback(hObject, eventdata, handles)
% hObject    handle to Nbnd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% 
% nf=max(1,round(str2double(get(handles.Nbnd,'String'))));
% set(handles.Nbnd,'String',num2str(nf,'%i'));

% Hints: get(hObject,'String') returns contents of Nbnd as text
%        str2double(get(hObject,'String')) returns contents of Nbnd as a double
% --- Executes during object creation, after setting all properties.
function Nbnd_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Nbnd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function noisefilt_Callback(hObject, eventdata, handles)
% hObject    handle to noisefilt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% nf=max(1,round(str2double(get(handles.noisefilt,'String'))));
% set(handles.noisefilt,'String',num2str(nf,'%i'));

% Hints: get(hObject,'String') returns contents of noisefilt as text
%        str2double(get(hObject,'String')) returns contents of noisefilt as a double

% --- Executes during object creation, after setting all properties.
function noisefilt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to noisefilt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function [xm,ym,zm] = maximaThresh(a,n,th,method)
% take the locals maxima of a 2d field
% interpolate the local minima position with barycenter method
%% 8 neighbour local max 
% |1 2 3|
% |8 0 4|
% |7 6 5|
% 
% b=a(2:end-1,2:end-1);
% I=  (b>a(1:end-2,2:end-1)) .* ... % 8
%     (b>a(3:end,2:end-1)) .* ... % 4
%     (b>a(2:end-1,1:end-2)) .* ... % 2
%     (b>a(2:end-1,3:end)) .* ... % 6
%     (b>a(1:end-2,1:end-2)) .* ... % 1
%     (b>a(1:end-2,3:end)) .* ... % 7
%     (b>a(3:end,1:end-2)) .* ... % 3
%     (b>a(3:end,3:end)) .* ... % 5
%     (b>th); % Threshold
% 
% [y,x]=find(I); % y vertical (rows) x (horiz) columns
% x=x+1;
% y=y+1;

% add a litlle bit of noise in case 2 pixels have the same intensity
% r=rand(size(a));
% a=a+r*1e-6;


%% Find n*n local max
r=rand(size(a))*1e-5;
% mask=true(n);
% mask(ceil(n/2),ceil(n/2))=0;
% b = ordfilt2(a+r,n^2-1,mask);
% [y,x]=find((a+r>b)&(a>th));
%SE = strel('disk',1)
SE = strel('square',n);
b = imdilate(a+r,SE);
[y,x]=find((a+r==b)&(a>th));

% Remove points on border
w=size(a,2);h=size(a,1);
nb=floor(n/2);
id=find(y<h-nb&y>=nb+1&x<w-nb&x>=nb+1);
x=x(id); y=y(id);

%% Subpixel Refinement Method

if strcmp(method,'gaussian')
% 2nd order poly fit on the logarithm of diag of a
[Dx,Dy,Z]=subpix2nd(real(log(a-min(a(:))+1e-8)),x,y,n);
% Take only peaks that moved into the pixel
idgood=find(abs(Dx)<0.5&abs(Dy)<0.5);
xb=x+Dx;
yb=y+Dy;
zb=Z;
    
elseif strcmp(method,'quadratic')
% 2nd order poly fit on the logarithm of diag of a
[Dx,Dy,Z]=subpix2nd(a,x,y,n);
% Take only peaks that moved into the pixel
idgood=find(abs(Dx)<0.5&abs(Dy)<0.5);
xb=x+Dx;
yb=y+Dy;
zb=Z;
% Take only peaks that moved into the pixel
idgood=find(abs(Dx)<0.5&abs(Dy)<0.5);
xb=x+Dx;
yb=y+Dy;
zb=Z;

elseif strcmp(method,'barycenter')
    % SHOULD NOT BE USED, do not converge to true value !!
% Center of gravity
S=zeros(size(x)); % for sum
Min=ones(size(x)); % for min
Dx=zeros(size(x));
Dy=zeros(size(x));

for i=-floor(n/2):floor(n/2)
   for  j=-floor(n/2):floor(n/2);
     idin=find((x+i)>0&(x+i)<=size(a,2)&(y+j)>0&(y+j)<=size(a,1)); % points inside the images
     idx=sub2ind(size(a),y(idin)+j,x(idin)+i);
     S(idin)=S(idin)+a(idx);
     Dx(idin)=Dx(idin)+i*a(idx);
     Dy(idin)=Dy(idin)+j*a(idx);
     Min(idin)=min(Min(idin),a(idx));
     end    
end
% 
Dx=Dx./(S-n^2*Min);
Dy=Dy./(S-n^2*Min);
% Dx=Dx./(S);
% Dy=Dy./(S);

% Take only motion inside one pixel
idgood=find(Dx<0.5&Dy<0.5);
xb=x+Dx;
yb=y+Dy;
zb=a(sub2ind(size(a),y,x));
else
  xb=x;
  yb=y;
  idgood=1:length(x);
  zb=zeros(size(xb));
end

xm=xb(idgood);
ym=yb(idgood);
zm=zb(idgood);

function [Dx,Dy,Z]=subpix2nd(a,x,y,n)
% Subpixel approximation of a 2nd order polynomial with a pencil of length
% np
np=floor(n/2);
pencil=-np:np;
X=repmat(pencil,length(x),1);
YH=zeros(size(X));
YV=zeros(size(X));
YD1=zeros(size(X));
YD2=zeros(size(X));

n=length(pencil);
YV=[];
YH=[];
%YD1=[];YD2=[];

for i=1:length(pencil) 
idV=sub2ind(size(a),max(1,min(size(a,1),y+pencil(i))),x);
idH=sub2ind(size(a),y,max(1,min(size(a,2),x+pencil(i))));
% Diagonals
%idDiag1=sub2ind(size(a),max(1,min(size(a,1),y+pencil(i))),max(1,min(size(a,2),x+pencil(i))));
%idDiag2=sub2ind(size(a),max(1,min(size(a,1),y-pencil(i))),max(1,min(size(a,2),x+pencil(i))));
YV(:,i)=a(idV);
YH(:,i)=a(idH);
% YD1(:,i)=a(idDiag1);
% YD2(:,i)=a(idDiag2);
end

% 2nd order poly a+bx+cx^2=0
s2=sum(pencil.^2);
s4=sum(pencil.^4);

bH=sum((YH.*X),2)'./s2;
cH=-(-s2*sum(YH,2)'+n*sum(X.^2.*YH,2)')/(s2^2-s4*n);
    
% Get third coefficient    
%    aH=-(s4*sum(YH')-s2*sum((X.^2.*YH)'))/(s2^2-s4*n);   
%     xi=-np:0.01:np;
%     z=500
%     hold on
%     plot(pencil,YH(z,:),'r*')
%     plot(xi,xi.^2*cH(z)+xi*bH(z)+aH(z)) 

bV=sum(YV.*X,2)'./s2;
cV=-(-s2*sum(YV,2)'+n*sum(X.^2.*YV,2)')/(s2^2-s4*n);

%     %Diagonals     
%     bD1=sum(YD1.*X,2)'./s2;
%     cD1=-(-s2*sum(YD1,2)'+n*sum(X.^2.*YD1,2)')/(s2^2-s4*n);
%     
%     bD2=sum(YD2.*X,2)'./s2;
%     cD2=-(-s2*sum(YD2,2)'+n*sum(X.^2.*YD2,2)')/(s2^2-s4*n);
%     
    % Peaks on hor and vert axis
    dH=-bH./2./cH;
    dV=-bV./2./cV;
    
%     % Peaks on diagonals
%     dD1=-bD1./2./cD1;
%     dD2=-bD2./2./cD2;
% Rotate -45 diagonals to get cartesian coordinates
%     dV2=(dD1-dD2);
%     dH2=(dD2+dD1);
%  Take average
%    Dx=(dH'+dH2')/2;
%    Dy=(dV'+dV2')/2;

      Dx=(dH');
      Dy=(dV');
      Z=YH(:,(n-1)/2);

function Id=cropPoints(Pts,ROI)
Id=[];
if ~isempty(Pts)
Id=find(Pts(:,2)>ROI(1,2)&Pts(:,2)<ROI(2,2)&Pts(:,1)>ROI(1,1)&Pts(:,1)<ROI(2,1));
end

% --- Executes on button press in PlotOn.
function PlotColor_Callback(hObject, eventdata, handles)
% hObject    handle to PlotOn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of PlotOn

% --- Executes on button press in PlotMotion.
function PlotMotion_Callback(hObject, eventdata, handles)
% hObject    handle to PlotMotion (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of PlotMotion

% --- Executes on button press in PlotVectors.
function PlotVectors_Callback(hObject, eventdata, handles)
% hObject    handle to PlotVectors (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of PlotVectors

function CMin_Callback(hObject, eventdata, handles)
% hObject    handle to CMin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(handles.axes2,'Clim',[str2double(get(handles.CMin,'String')) str2double(get(handles.CMax,'String'))]);
% Hints: get(hObject,'String') returns contents of CMin as text
%        str2double(get(hObject,'String')) returns contents of CMin as a double

% --- Executes during object creation, after setting all properties.
function CMin_CreateFcn(hObject, eventdata, handles)
% hObject    handle to CMin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function CMax_Callback(hObject, eventdata, handles)
% hObject    handle to CMax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.axes2,'Clim',[str2double(get(handles.CMin,'String')) str2double(get(handles.CMax,'String'))]);
% Hints: get(hObject,'String') returns contents of CMax as text
%        str2double(get(hObject,'String')) returns contents of CMax as a double

% --- Executes during object creation, after setting all properties.
function CMax_CreateFcn(hObject, eventdata, handles)
% hObject    handle to CMax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function PlotScale_Callback(hObject, eventdata, handles)
nf=max(0,str2double(get(handles.PlotScale,'String')));
set(handles.PlotScale,'String',num2str(nf));

function PlotScale_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function PlotPoints_Callback(hObject, eventdata, handles)
if get(handles.PlotPoints,'Value')==1;
   set(handles.PlotError,'Value',0);
   set(handles.PlotID,'Value',0);
end
% -- Executes on button press in PostProc.
function PostProc_Callback(hObject, eventdata, handles)
% hObject    handle to PostProc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Check if a mat exist


data=guidata(hObject);
filename_mat=[data.path '/' data.fname '_track.mat'];
filename_txt=[data.path '/' data.fname '_track.txt'];

if isfield(data,'infos_all'),
data.infos_all=[sprintf('> Loading Raw Track file ... \n') data.infos_all];
else
data.infos_all=[sprintf('> Loading Raw Track file ... \n')];
end 
set(handles.Info,'String',data.infos_all);
drawnow


%disp(Load_vid)
POST=0;
if exist(filename_txt);
Pts=dlmread(filename_txt,' ',3,0);   
POST=1; 
elseif exist(filename_mat)
load(filename_mat);
POST=1;
% Old mat files
elseif exist([data.path '/' data.fname '.mat'])
load([data.path '/' data.fname '.mat']);
POST=1;
end

if POST==1
% gET pARAMETERS
fps=1;res=1;binX=1;binY=1;rot=0;
if isfield(data,'par')
    if isfield(data.par,'fps'); fps=data.par.fps; end
    if isfield(data.par,'res'); res=data.par.res; end
    if isfield(data.par,'binX'); binX=data.par.binX; end
    if isfield(data.par,'binY'); binY=data.par.binY; end
    if isfield(data.par,'rot'); rot=data.par.rot; end
end
Frames=max(Pts(:,1));
% Rotate Points & velocity by -rot
Xrot=Pts(:,3)*cosd(rot)-Pts(:,4)*sind(rot);
Yrot=Pts(:,3)*sind(rot)+Pts(:,4)*cosd(rot);
Urot=Pts(:,5)*cosd(rot)-Pts(:,6)*sind(rot);
Vrot=Pts(:,5)*sind(rot)+Pts(:,6)*cosd(rot);
Erot=Pts(:,11);

xmin=floor(min(Xrot));
xmax=ceil(max(Xrot));
ymin=floor(min(Yrot));
ymax=ceil(max(Yrot));

dx=binX;
dy=binY;

[X,Y]=meshgrid(xmin-dx:dx:xmax+dx,ymin-dy:dy:ymax+dy);

xf=ceil((Xrot-xmin+1)/dx);
yf=ceil((Yrot-ymin+1)/dy);

U=cell(size(X));
V=cell(size(X));
E=cell(size(X));
%Ax=cell(size(X));
%Ay=cell(size(X));

for i=1:length(xf)
    U{yf(i),xf(i)}=[U{yf(i),xf(i)} Urot(i)*res*fps];
    V{yf(i),xf(i)}=[V{yf(i),xf(i)} Vrot(i)*res*fps];
    E{yf(i),xf(i)}=[E{yf(i),xf(i)} Erot(i)];
    
    if mod(i,10000)==0,
    set(handles.Info,'String',sprintf('Post Processing: %02i perc... \n',round(i/length(xf)*100)));
    drawnow
    end
end
% 
% for i=1:size(X,1)
%     for j=1:size(X,2)
%         %[i j]
%         id=find(xf==j&yf==i);
%         U{i,j}=Urot(id)*res*fps;
%         V{i,j}=Vrot(id)*res*fps;
%       %  Ax{i,j}=Pts(id,9);
%       %  Ay{i,j}=Pts(id,10); 
%         adv=((i-1)*size(X,2)+j)/size(X,2)/size(X,1);
%        
%         set(handles.Info,'String',sprintf('Post Processing... %1.2f percent',adv*100));
%         drawnow
%     end
% end


data=guidata(hObject);
data.U=U;
data.V=V;
data.E=E;
data.X=X*res;
data.Y=Y*res;

data.dx=dx;
data.dy=dy;
data.res=res;
data.fps=fps;
data.Frames=Frames;

data.infos_all=[sprintf('> Post-Processing Done. Please save it ! \n') data.infos_all];
set(handles.Info,'String',data.infos_all);
guidata(hObject,data);
    
%handles.Info.String=sprintf('Post Processing Done');


set(handles.PostSave,'Enable','On');
else

set(handles.Info,'String',sprintf('No tracking file found! You need to Save your results before post-processing !'));
end

% --- Executes on button press in PostSave.
function PostSave_Callback(hObject, eventdata, handles)
% hObject    handle to PostSave (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
data=guidata(hObject);
if isfield(data,'U')&isfield(data,'V')&isfield(data,'X')&isfield(data,'Y'),
 filename=data.fname;
 

 set(handles.Info,'String','... Saving ... Please wait');
drawnow
 U=data.U;
 V=data.V;
 E=data.E;
 X=data.X;
 Y=data.Y;
 dx=data.dx;
 dy=data.dy;
 res=data.res;
 fps=data.fps;
 Frames=data.Frames;
 save([data.path '/' data.fname '_PostProc.mat'],'U','V','E','X','Y','dx','dy','res','fps','Frames','-v7.3');
 
data.infos_all=[sprintf('> Post-Proc saved to %s_PostProc.mat \n',data.fname) data.infos_all];
set(handles.Info,'String',data.infos_all);

 disp(sprintf('PostProcessing saved to  %s_PostProc.mat',filename))
 
 set(handles.PlotMenu,'Enable','On');
 
 guidata(hObject,data);
 else

data.infos_all=['!!!! No PostProcessing to save !!!!' data.infos_all];
set(handles.Info,'String',data.infos_all);
 guidata(hObject,data);
end
 
% --- Executes on button press in PostPlot.
function PostPlot_Callback(hObject, eventdata, handles, PostPlotType)
% hObject    handle to PostPlot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

data=guidata(hObject);
filenamePost=[data.path '/' data.fname '_PostProc.mat'];
if exist(filenamePost)

 set(handles.Info,'String','... Plotting ... Please wait');
drawnow
load(filenamePost);

if PostPlotType<7
switch PostPlotType, % 3D plots
    case 1,
        
         data.nfig=PostPlotType;
         data.figname='Phi';
        % Plot Ph
        P=cellfun(@length,V)/Frames/(dx*res)/(dy*res)*(pi*0.0005^2);
        cmin=max(0,min(min(P)));
        cmax=min(1,max(max(P)));
        tit='<\Phi>';

    case 2,
        
         data.nfig=PostPlotType;
         data.figname='U';
        % Plot Um
        P=cellfun(@mean,U);
        cmin=nanmean(nanmean(P))-3*nanstd(nanstd(P));
        cmax=nanmean(nanmean(P))+3*nanstd(nanstd(P));
        tit='<U> (m/s)';
    case 3,
        
         data.nfig=PostPlotType;
         data.figname='V';
        % Plot Vm
        P=cellfun(@mean,V);
        cmin=nanmean(nanmean(P))-2*nanstd(nanstd(P));
        cmax=nanmean(nanmean(P))+2*nanstd(nanstd(P));
        tit='<V> (m/s)';
        
    case 4,
        
         data.nfig=PostPlotType;
         data.figname='Up2';
    P=cellfun(@var,U);
        cmin=max(0,nanmean(nanmean(P))-nanstd(nanstd(P)));
        cmax=nanmean(nanmean(P))+nanstd(nanstd(P));
        tit='<U''^2 (m^2/s^2)>';
    
    case 5,
        
         data.nfig=PostPlotType;
         data.figname='Vp2';
    P=cellfun(@var,V);
        cmin=max(0,nanmedian(nanmean(P))-nanstd(nanstd(P)));
        cmax=nanmean(nanmedian(P))+nanstd(nanstd(P));
        tit='<V''^2> (m^2/s^2)';
        
    case 6,
        
         data.nfig=PostPlotType;
         data.figname='UpVp';
    P=cellfun(@(x,y) mean((x-mean(x)).*(y-mean(y))),U,V);
        cmin=nanmean(nanmedian(P))-nanstd(nanstd(P));
        cmax=nanmean(nanmedian(P))+nanstd(nanstd(P));
        tit='<U''V''> (m^2/s^2)';   
        
end
        figure(PostPlotType+1)
        clf
        colormap jet
        set(gcf,'Name',data.figname,'Position',[100 100 size(P,2)*4 size(P,1)*4])
        subplot('position',[0.1 0.05 0.85 0.92]);
        surf(X,Y,zeros(size(P)),P,'edgecolor','none')
        hold on
        box
        colorbar
        view(2)
        caxis([cmin cmax])
        xlabel('x (m)')
        ylabel('y (m)')
        axis ij
        %axis([min(X(1,:)) max(X(1,:)) min(Y(:,1)) max(Y(:,1))])
        axis tight
        axis equal
        title(tit)
     %  set(gca,'Position',[.1 .08 .78 .9])
set(handles.Info,'String',data.infos_all);
        drawnow
end

if PostPlotType>6 % 2D plots
    switch PostPlotType, % 3D plots
    case 7,
        
         data.nfig=PostPlotType;
         data.figname='UxVxPhix';
        Pu=nanmean(cellfun(@mean,U),1);
        Pv=nanmean(cellfun(@mean,V),1);
        Pphi=nanmean(cellfun(@length,V)/Frames/(dx*res)/(dy*res)*(pi*0.0005^2),1);
        figure(PostPlotType)
        clf
        plot(X(1,:),Pu,'-o')
        hold on
        plot(X(1,:),Pv,'-d')
        plot(X(1,:),Pphi,'-s')
        legend('<U(x)>','<V(x)>','<Phi(x)>')
        xlabel('x (m)')
        ylabel('<U(x)>,<V(x)> (m/s), \Phi')

    case 8,
        
         data.nfig=PostPlotType;
         data.figname='UyVyPhiy';
        Pu=nanmean(cellfun(@mean,U),2);
        Pv=nanmean(cellfun(@mean,V),2);
        Pphi=nanmean(cellfun(@length,V)/Frames/(dx*res)/(dy*res)*(pi*0.0005^2),2);
             figure(PostPlotType)
        clf
        plot(Pu,Y(:,1),'-o')
        hold on
        plot(Pv,Y(:,1),'-d')
        plot(Pphi,Y(:,1),'-s')
        ylabel('y (m)')
        xlabel('<U(y)>,<V(y)> (m/s), \Phi')
        legend('<U(y)>','<V(y)>','<Phi(y)>')
        axis ij
        
    case 9,
        
         data.nfig=PostPlotType;
         data.figname='UpxVpx';
        Pu=nanmedian(cellfun(@var,U),1);
        Pv=nanmedian(cellfun(@var,V),1);
        Puv=nanmedian(cellfun(@(x,y) mean((x-mean(x)).*(y-mean(y))),U,V),1);
             figure(PostPlotType)
        clf
        plot(X(1,:),Pu)
        hold on
        plot(X(1,:),Pv)
        plot(X(1,:),Puv)
        xlabel('x (m)')
        ylabel('<U''^2(x)>,<V''^2(x)>,<U''V''(x)> (m^2/s^2)')
        legend('<U''^2(x)>','<V''^2(x)>','<U''V''(x)>') 
    case 10,
        
         data.nfig=PostPlotType;
         data.figname='UpyVpy';
        Pu=nanmedian(cellfun(@var,U),2);
        Pv=nanmedian(cellfun(@var,V),2);
        Puv=nanmedian(cellfun(@(x,y) mean((x-mean(x)).*(y-mean(y))),U,V),2);
             figure(PostPlotType)
        clf
        plot(Pu,Y(:,1))
        hold on
        plot(Pv,Y(:,1))
        plot(Puv,Y(:,1))
        ylabel('y (m)')
        xlabel('<U''^2(y)>,<V''^2(y)>,<U''V''(y)> (m^2/s^2)')
        legend('<U''^2(y)>','<V''^2(y)>','<U''V''(y)> (m^2/s^2)') 
        axis ij
     case 11,
         
         data.nfig=PostPlotType;
         data.figname='100traj';
         % Plot 500 traj
        ntraj=100;
        filename=[data.path '/' data.fname '_track.mat'];
        load(filename);
        idtraj=unique(round(rand(ntraj,1)*max(Pts(:,2))));
        figure(PostPlotType)
        clf
        axis ij
        hold on
        for i=1:length(idtraj)
        id=find(Pts(:,2)==idtraj(i));
        xt=Pts(id,3);
        yt=Pts(id,4);
        plot(xt,yt)
        pause(0.001)
        end
     case 12,
         data.nfig=PostPlotType;
         data.figname='500traj';
         % Plot 500 traj
        ntraj=500;
        filename=[data.path '/' data.fname '_track.mat'];
        load(filename);
        idtraj=unique(round(rand(ntraj,1)*max(Pts(:,2))));
             figure(PostPlotType)
        clf
        axis ij
        hold on
        for i=1:length(idtraj)
        id=find(Pts(:,2)==idtraj(i));
        xt=Pts(id,3);
        yt=Pts(id,4);
        plot(xt,yt)
        pause(0.001)
        end
     case 13,
         data.nfig=PostPlotType;
         data.figname='1000traj';
         % Plot 500 traj
        ntraj=2000;
        filename=[data.path '/' data.fname '_track.mat'];
        load(filename);
        idtraj=unique(round(rand(ntraj,1)*max(Pts(:,2))));
        figure(PostPlotType)
        clf
        axis ij
        hold on
        for i=1:length(idtraj)
        id=find(Pts(:,2)==idtraj(i));
        xt=Pts(id,3);
        yt=Pts(id,4);
        plot(xt,yt)
        pause(0.001)
        end
    end

        
set(handles.Info,'String',data.infos_all);
        drawnow
end
guidata(hObject,data);
 else
 
        set(handles.Info,'String',sprintf('No PostProcessing file found ! Press Post-Processing and then Save !'));
end

% --- Executes on selection change in PostPlotType.
function PostPlotType_Callback(hObject, eventdata, handles)
% hObject    handle to PostPlotType (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns PostPlotType contents as cell array
%        contents{get(hObject,'Value')} returns selected item from PostPlotType

% --- Executes during object creation, after setting all properties.
function PostPlotType_CreateFcn(hObject, eventdata, handles)
% hObject    handle to PostPlotType (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function PostFPS_Callback(hObject, eventdata, handles)
% hObject    handle to PostFPS (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% 
nf=max(0,(str2double(get(handles.PostFPS,'String'))));
set(handles.PostFPS,'String',num2str(nf));

% Hints: get(hObject,'String') returns contents of PostFPS as text
%        str2double(get(hObject,'String')) returns contents of PostFPS as a double

% --- Executes during object creation, after setting all properties.
function PostFPS_CreateFcn(hObject, eventdata, handles)
% hObject    handle to PostFPS (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function PostRes_Callback(hObject, eventdata, handles)
% hObject    handle to PostRes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% 
nf=max(0,(str2double(get(handles.PostRes,'String'))));
set(handles.PostRes,'String',num2str(nf));

% Hints: get(hObject,'String') returns contents of PostRes as text
%        str2double(get(hObject,'String')) returns contents of PostRes as a double

% --- Executes during object creation, after setting all properties.
function PostRes_CreateFcn(hObject, eventdata, handles)
% hObject    handle to PostRes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function PostBinX_Callback(hObject, eventdata, handles)
% hObject    handle to PostBinX (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% 
% 
nf=max(1,(str2double(get(handles.PostBinX,'String'))));
set(handles.PostBinX,'String',num2str(nf));

% Hints: get(hObject,'String') returns contents of PostBinX as text
%        str2double(get(hObject,'String')) returns contents of PostBinX as a double

% --- Executes during object creation, after setting all properties.
function PostBinX_CreateFcn(hObject, eventdata, handles)
% hObject    handle to PostBinX (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function PostBinY_Callback(hObject, eventdata, handles)
% hObject    handle to PostBinY (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% 
nf=max(1,(str2double(get(handles.PostBinY,'String'))));
set(handles.PostBinY,'String',num2str(nf));
% Hints: get(hObject,'String') returns contents of PostBinY as text
%        str2double(get(hObject,'String')) returns contents of PostBinY as a doubl

% --- Executes during object creation, after setting all properties.
function PostBinY_CreateFcn(hObject, eventdata, handles)
% hObject    handle to PostBinY (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in Filelist.
function Filelist_Callback(hObject, eventdata, handles)

% [file, path] = uigetfile({'*.txt;','Filelist in txt format'});
% filename=[path file];
% [path, name, ext] = fileparts(filename);
% filename=data.fname;
% 
% if exist(filename)
% [pathstr,name,ext] = fileparts(filename);
% else
% pathstr=[];
% end

[file, path] = uigetfile({'*.txt;','Filelist in txt format'});
filelist=[path file];

if ~isempty(filelist)
if exist(filelist)
fid=fopen(filelist,'r');
T=textscan(fid,'%s');

for t=1:length(T{1,:})
filename=T{1}{t};
set(handles.Load_vid,'String',filename);
drawnow
if exist(filename)
% Load video filemenu
load_video(hObject, eventdata, handles)
% start treatment
Start_Callback(hObject, eventdata, handles)
% save treatment
%Save_Callback(hObject, eventdata, handles)
SaveASCII_Callback(hObject, eventdata, handles)
% Post Processing
PostProc_Callback(hObject, eventdata, handles)
% save Post Processing
PostSave_Callback(hObject, eventdata, handles)
%clear existing data
guidata(hObject,struct());
end
end
fclose(fid)
else
set(handles.Info,'String','File List not found');
end
end
    

% hObject    handle to Filelist (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% NAN functions
function y = nanmean(x,dim)
% FORMAT: Y = NANMEAN(X,DIM)
% 
%    Average or mean value ignoring NaNs
%
%    This function enhances the functionality of NANMEAN as distributed in
%    the MATLAB Statistics Toolbox and is meant as a replacement (hence the
%    identical name).  
%
%    NANMEAN(X,DIM) calculates the mean along any dimension of the N-D
%    array X ignoring NaNs.  If DIM is omitted NANMEAN averages along the
%    first non-singleton dimension of X.
%
%    Similar replacements exist for NANSTD, NANMEDIAN, NANMIN, NANMAX, and
%    NANSUM which are all part of the NaN-suite.
%
%    See also MEAN

% -------------------------------------------------------------------------
%    author:      Jan Gl�scher
%    affiliation: Neuroimage Nord, University of Hamburg, Germany
%    email:       glaescher@uke.uni-hamburg.de
%    
%    $Revision: 1.1 $ $Date: 2004/07/15 22:42:13 $

if isempty(x)
	y = NaN;
	return
end

if nargin < 2
	dim = min(find(size(x)~=1));
	if isempty(dim)
		dim = 1;
	end
end

% Replace NaNs with zeros.
nans = isnan(x);
x(isnan(x)) = 0; 

% denominator
count = size(x,dim) - sum(nans,dim);

% Protect against a  all NaNs in one dimension
i = find(count==0);
count(i) = ones(size(i));

y = sum(x,dim)./count;
y(i) = i + NaN;

function y = nansum(x,dim)
% FORMAT: Y = NANSUM(X,DIM)
% 
%    Sum of values ignoring NaNs
%
%    This function enhances the functionality of NANSUM as distributed in
%    the MATLAB Statistics Toolbox and is meant as a replacement (hence the
%    identical name).  
%
%    NANSUM(X,DIM) calculates the mean along any dimension of the N-D array
%    X ignoring NaNs.  If DIM is omitted NANSUM averages along the first
%    non-singleton dimension of X.
%
%    Similar replacements exist for NANMEAN, NANSTD, NANMEDIAN, NANMIN, and
%    NANMAX which are all part of the NaN-suite.
%
%    See also SUM

% -------------------------------------------------------------------------
%    author:      Jan Gl�scher
%    affiliation: Neuroimage Nord, University of Hamburg, Germany
%    email:       glaescher@uke.uni-hamburg.de
%    
%    $Revision: 1.2 $ $Date: 2005/06/13 12:14:38 $

if isempty(x)
	y = [];
	return
end

if nargin < 2
	dim = min(find(size(x)~=1));
	if isempty(dim)
		dim = 1;
	end
end

% Replace NaNs with zeros.
nans = isnan(x);
x(isnan(x)) = 0; 

% Protect against all NaNs in one dimension
count = size(x,dim) - sum(nans,dim);
i = find(count==0);

y = sum(x,dim);
y(i) = NaN;

function y = nanstd(x,dim,flag)
% FORMAT: Y = NANSTD(X,DIM,FLAG)
% 
%    Standard deviation ignoring NaNs
%
%    This function enhances the functionality of NANSTD as distributed in
%    the MATLAB Statistics Toolbox and is meant as a replacement (hence the
%    identical name).  
%
%    NANSTD(X,DIM) calculates the standard deviation along any dimension of
%    the N-D array X ignoring NaNs.  
%
%    NANSTD(X,DIM,0) normalizes by (N-1) where N is SIZE(X,DIM).  This make
%    NANSTD(X,DIM).^2 the best unbiased estimate of the variance if X is
%    a sample of a normal distribution. If omitted FLAG is set to zero.
%    
%    NANSTD(X,DIM,1) normalizes by N and produces the square root of the
%    second moment of the sample about the mean.
%
%    If DIM is omitted NANSTD calculates the standard deviation along first
%    non-singleton dimension of X.
%
%    Similar replacements exist for NANMEAN, NANMEDIAN, NANMIN, NANMAX, and
%    NANSUM which are all part of the NaN-suite.
%
%    See also STD

% -------------------------------------------------------------------------
%    author:      Jan Gl�scher
%    affiliation: Neuroimage Nord, University of Hamburg, Germany
%    email:       glaescher@uke.uni-hamburg.de
%    
%    $Revision: 1.1 $ $Date: 2004/07/15 22:42:15 $

if isempty(x)
	y = NaN;
	return
end

if nargin < 3
	flag = 0;
end

if nargin < 2
	dim = min(find(size(x)~=1));
	if isempty(dim)
		dim = 1; 
	end	  
end


% Find NaNs in x and nanmean(x)
nans = isnan(x);
avg = nanmean(x,dim);

% create array indicating number of element 
% of x in dimension DIM (needed for subtraction of mean)
tile = ones(1,max(ndims(x),dim));
tile(dim) = size(x,dim);

% remove mean
x = x - repmat(avg,tile);

count = size(x,dim) - sum(nans,dim);

% Replace NaNs with zeros.
x(isnan(x)) = 0; 


% Protect against a  all NaNs in one dimension
i = find(count==0);

if flag == 0
	y = sqrt(sum(x.*x,dim)./max(count-1,1));
else
	y = sqrt(sum(x.*x,dim)./max(count,1));
end
y(i) = i + NaN;

function y = nanmedian(x,dim)
% FORMAT: Y = NANMEDIAN(X,DIM)
% 
%    Median ignoring NaNs
%
%    This function enhances the functionality of NANMEDIAN as distributed
%    in the MATLAB Statistics Toolbox and is meant as a replacement (hence
%    the identical name).  
%
%    NANMEDIAN(X,DIM) calculates the mean along any dimension of the N-D
%    array X ignoring NaNs.  If DIM is omitted NANMEDIAN averages along the
%    first non-singleton dimension of X.
%
%    Similar replacements exist for NANMEAN, NANSTD, NANMIN, NANMAX, and
%    NANSUM which are all part of the NaN-suite.
%
%    See also MEDIAN

% -------------------------------------------------------------------------
%    author:      Jan Gl�scher
%    affiliation: Neuroimage Nord, University of Hamburg, Germany
%    email:       glaescher@uke.uni-hamburg.de
%    
%    $Revision: 1.2 $ $Date: 2007/07/30 17:19:19 $

if isempty(x)
	y = [];
	return
end

if nargin < 2
	dim = min(find(size(x)~=1));
	if isempty(dim)
		dim = 1;
	end
end

siz  = size(x);
n    = size(x,dim);

% Permute and reshape so that DIM becomes the row dimension of a 2-D array
perm = [dim:max(length(size(x)),dim) 1:dim-1];
x = reshape(permute(x,perm),n,prod(siz)/n);


% force NaNs to bottom of each column
x = sort(x,1);

% identify and replace NaNs
nans = isnan(x);
x(isnan(x)) = 0;

% new dimension of x
[n m] = size(x);

% number of non-NaN element in each column
s = size(x,1) - sum(nans);
y = zeros(size(s));

% now calculate median for every element in y
% (does anybody know a more eefficient way than with a 'for'-loop?)
for i = 1:length(s)
	if rem(s(i),2) & s(i) > 0
		y(i) = x((s(i)+1)/2,i);
	elseif rem(s(i),2)==0 & s(i) > 0
		y(i) = (x(s(i)/2,i) + x((s(i)/2)+1,i))/2;
	end
end

% Protect against a column of NaNs
i = find(y==0);
y(i) = i + nan;

% permute and reshape back
siz(dim) = 1;
y = ipermute(reshape(y,siz(perm)),perm);

% $Id: nanmedian.m,v 1.2 2007/07/30 17:19:19 glaescher Exp glaescher $


% --- Executes on button press in PostPrint.
function PostPrint_Callback(hObject, eventdata, handles)
data=guidata(hObject);

if isfield(data,'nfig')
nfig=data.nfig;
figname=data.figname;
filename=data.fname;
filename=filename;
fname=[data.path '/' filename '_' figname '.png'];

        set(handles.Info,'String',sprintf('Printing to %s ...',fname));
drawnow
nfig
figure(nfig)
set(gcf,'PaperPositionMode','auto')
set(gcf,'InvertHardcopy','off')
print(nfig,[data.path '/' filename '_' figname '.png'],'-dpng','-r300')

        set(handles.Info,'String',sprintf('Printing to %s Ok !',fname));
drawnow
else

        set(handles.Info,'String',sprintf('No plot found !'));end

% hObject    handle to PostPrint (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function PostRota_Callback(hObject, eventdata, handles)
% hObject    handle to textRot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Load_vid=handles.Load_vid.String;
%handles

%if isfield(handles,'PostRota')
rot=str2double(get(handles.PostRota,'String'));
data=guidata(hObject);
if isfield(data,'vid1')
I00=read(data.vid1,1);
ImW=size(I00,2);
ImH=size(I00,1);
x1=[1;ImW];
y1=ImH/2+(ImW/2)*tand(rot)*[-1;1];
x2=ImW/2+(ImH/2)*tand(rot)*[1;-1];
y2=[1;ImH];

if isfield(data,'Prot')
delete(data.Prot); rmfield(data,'Prot');
end

axes(handles.axes1);
hold on
X=[x1 x2];
Y=[y1 y2];
data.Prot=plot(X,Y,'r-','Parent',handles.axes1);
drawnow
guidata(hObject,data);
end
%end

% if exist(Load_vid)
% vid1 = VideoReader(Load_vid);
% nFrames=vid1.NumberOfFrames;
% I00=read(vid1,1);
% x1=[1;size(I00,2)];
% y1=floor([size(I00,1)/2;size(I00,1)/2]);
% x2=floor([size(I00,2)/2;size(I00,2)/2]);
% y2=floor([1;size(I00,1)]);
% 
% x1rot=
% y1rot=
% plot()

%end
% Hints: get(hObject,'String') returns contents of textRot as text
%        str2double(get(hObject,'String')) returns contents of textRot as a double


% --- Executes during object creation, after setting all properties.
function textRot_CreateFcn(hObject, eventdata, handles)
% hObject    handle to textRot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function PostRota_CreateFcn(hObject, eventdata, handles)
% hObject    handle to PostRota (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in MinimaOn.
function MinimaOn_Callback(hObject, eventdata, handles)
% hObject    handle to MinimaOn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.MinimaOff,'Value',0);
% Hint: get(hObject,'Value') returns toggle state of MinimaOn


% --- Executes on button press in Help.
function Help_Callback(hObject, eventdata, handles)
% Display Help image
fname='ReadMe.png';
if exist(fname)
I=imread(fname);
h=figure('Position',[100 100 1100 656]);
imshow(I)
else
set(handles.Info,'String',sprintf('No help file found :( '));
end


% --- Executes on button press in PlotError.
function PlotError_Callback(hObject, eventdata, handles)
% hObject    handle to PlotError (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if get(handles.PlotError,'Value')==1;
   set(handles.PlotPoints,'Value',0);
   set(handles.PlotID,'Value',0);
end
% Hint: get(hObject,'Value') returns toggle state of PlotError


% --- Executes on selection change in PlotImageType.
function PlotImageType_Callback(hObject, eventdata, handles)
% hObject    handle to PlotImageType (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns PlotImageType contents as cell array
%        contents{get(hObject,'Value')} returns selected item from PlotImageType


% --- Executes during object creation, after setting all properties.
function PlotImageType_CreateFcn(hObject, eventdata, handles)
% hObject    handle to PlotImageType (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in BWAuto.
function BWAuto_Callback(hObject, eventdata, handles)
% hObject    handle to BWAuto (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
A=get(handles.BWAuto,'Value');
if A
    set(handles.bw,'Visible','Off')
else
    set(handles.bw,'Visible','On')
end
% Hint: get(hObject,'Value') returns toggle state of BWAuto


% --- Executes on button press in RecOn.
function RecOn_Callback(hObject, eventdata, handles)
% hObject    handle to RecOn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of RecOn


% --------------------------------------------------------------------
function Load_vid_ClickedCallback(hObject, eventdata, handles)
Load_vid_Callback(hObject, eventdata, handles)
% hObject    handle to Load_vid (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Stop_ClickedCallback(hObject, eventdata, handles)
Stop_Callback(hObject, eventdata, handles)


% --------------------------------------------------------------------
function Start_ClickedCallback(hObject, eventdata, handles)
Start_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function Save_ClickedCallback(hObject, eventdata, handles)
Save_Callback(hObject, eventdata, handles)
% hObject    handle to Save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function  figure1_WindowButtonUpFcn(hObject, eventdata, handles)


% --------------------------------------------------------------------
function traj10_Callback(hObject, eventdata, handles)
PostPlot_Callback(hObject, eventdata, handles,11)
% hObject    handle to traj2000 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function traj500_Callback(hObject, eventdata, handles)
PostPlot_Callback(hObject, eventdata, handles,12)
% hObject    handle to traj500 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function traj2000_Callback(hObject, eventdata, handles)
PostPlot_Callback(hObject, eventdata, handles,13)
% hObject    handle to traj2000 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function upup_Callback(hObject, eventdata, handles)
PostPlot_Callback(hObject, eventdata, handles,4)
% hObject    handle to upup (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



% --------------------------------------------------------------------
function vpvp_Callback(hObject, eventdata, handles)
PostPlot_Callback(hObject, eventdata, handles,5)
% hObject    handle to vpvp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% --------------------------------------------------------------------
function upvp_Callback(hObject, eventdata, handles)
PostPlot_Callback(hObject, eventdata, handles,6)
% hObject    handle to upvp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function upx_Callback(hObject, eventdata, handles)
PostPlot_Callback(hObject, eventdata, handles,9)
% hObject    handle to upx (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function upy_Callback(hObject, eventdata, handles)
PostPlot_Callback(hObject, eventdata, handles,10)
% hObject    handle to upy (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Uxy_Callback(hObject, eventdata, handles)
% hObject    handle to Uxy (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
PostPlot_Callback(hObject, eventdata, handles,2)

% --------------------------------------------------------------------
function Vxy_Callback(hObject, eventdata, handles)
% hObject    handle to Vxy (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
PostPlot_Callback(hObject, eventdata, handles,3)

% --------------------------------------------------------------------
function UVx_Callback(hObject, eventdata, handles)
PostPlot_Callback(hObject, eventdata, handles,7)
% hObject    handle to UVx (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function UVy_Callback(hObject, eventdata, handles)
PostPlot_Callback(hObject, eventdata, handles,8)
% hObject    handle to UVy (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function TrackMenu_Callback(hObject, eventdata, handles)
% hObject    handle to TrackMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function PostMenu_Callback(hObject, eventdata, handles)
% hObject    handle to PostMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function FileMenu_Callback(hObject, eventdata, handles)
% hObject    handle to FileMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function PlotMenu_Callback(hObject, eventdata, handles)
% hObject    handle to PlotMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function AvMenu_Callback(hObject, eventdata, handles)
% hObject    handle to AvMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function FlucMenu_Callback(hObject, eventdata, handles)
% hObject    handle to FlucMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function TrajMenu_Callback(hObject, eventdata, handles)
% hObject    handle to TrajMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function SetParam_Callback(hObject, eventdata, handles)
    data=guidata(hObject);
    prompt = {'Frames per second:','Resolution (m/pix):','Grid size X (pix):','Grid size Y (pix):','Rotation (°):'};
    dlg_title = 'Post-Processing Parameters';
    num_lines = 1;
    fps=1;res=1;binX=1;binY=1;rot=0;
    if isfield(data,'par')
        if isfield(data.par,'fps'); fps=data.par.fps; end
        if isfield(data.par,'res'); res=data.par.res; end
        if isfield(data.par,'binX'); binX=data.par.binX; end
        if isfield(data.par,'binY'); binY=data.par.binY; end
        if isfield(data.par,'rot'); rot=data.par.rot; end
    end
    defaultans = {num2str(fps),num2str(res),...
            num2str(binX),num2str(binY),num2str(rot)};

    answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
    data.par.fps=str2num(answer{1,1});
    data.par.res=str2num(answer{2,1});
    data.par.binX=str2num(answer{3,1});
    data.par.binY=str2num(answer{4,1});
    data.par.rot=str2num(answer{5,1});
    guidata(hObject,data);
    
% hObject    handle to SetParam (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in MinimaOff.
function MinimaOff_Callback(hObject, eventdata, handles)
% hObject    handle to MinimaOff (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.MinimaOn,'Value',0);
% Hint: get(hObject,'Value') returns toggle state of MinimaOff


function [U_filt,E_filt,Xm_old,Um_old,Em_old]=Propagate_MotionModel_KdTree(C,Xm,Um,Em,BND,Xm_old,Um_old,Em_old,th)
%% propagation of MOTION MODEL with nn neighboors of k d trees
% KdTree
Xref=[Xm;Xm_old;BND];
Uref=[Um;Um_old;BND];
Eref=[Em;Em_old;zeros(size(BND,1),1)];

% Get firsts nn'th neighboors of query points
nn=max(1,min(int8(th.motion_av),size(Xref,1)));

%Make Tree
tXref = KDTreeSearcher(Xref);
[neighboors,distance] = knnsearch(tXref,C,'k',nn);

% Make matrices and compute local average
values_U=reshape(Uref(neighboors(:),:),[],nn,2);
values_E=reshape(Eref(neighboors(:)),[],nn);

% Averaging over neighboors values

% Option 1 : simple average of neighbooring values
%	U_filt=np.nanmean(values_U,axis=1).reshape(-1,2)
%	E_filt=np.nanmean(values_E,axis=1).reshape(-1)

% Option 2 : weighted average with weight on previous model
W=ones(size(Xref,1),1);
%W[:Xm.shape[0]]=W[:Xm.shape[0]]*np.maximum(0,-Em)
W(end-size(Xm_old,1)+1:end)=W(end-size(Xm_old,1)+1:end)*th.filter_time; %np.maximum(0,-Em_old)
Wneighboors=reshape(W(neighboors(:)),[],nn);
U_filt=reshape(sum(values_U.*repmat(Wneighboors,1,1,2),2)./sum(repmat(Wneighboors,1,1,2),2),[],2);
E_filt=sum(values_E.*Wneighboors,2)./sum(Wneighboors,2);

% Option 2 : weighted average with weight on previous model + distance + error


% Find nan and replace by default 0 value
idgood=isfinite(U_filt(:,1));

% Save non nan to ponderate next iteration
Xm_old=C(idgood,:);
Um_old=U_filt(idgood,:);
Em_old=E_filt(idgood);

% Replace nan if no model points where given
if length(U_filt)>0,
idnan=isnan(U_filt(:,1));
U_filt(idnan,:)=0;
E_filt(isnan(E_filt))=2;
end



function motion_it_Callback(hObject, eventdata, handles)
% hObject    handle to motion_it (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of motion_it as text
%        str2double(get(hObject,'String')) returns contents of motion_it as a double


% --- Executes during object creation, after setting all properties.
function motion_it_CreateFcn(hObject, eventdata, handles)
% hObject    handle to motion_it (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in radiobutton17.
function radiobutton17_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton17


% --- Executes on button press in motion_steady.
function motion_steady_Callback(hObject, eventdata, handles)
% hObject    handle to motion_steady (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(handles.motion_unsteady,'Value',0)
% Hint: get(hObject,'Value') returns toggle state of motion_steady


% --- Executes on button press in motion_unsteady.
function motion_unsteady_Callback(hObject, eventdata, handles)
% hObject    handle to motion_unsteady (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.motion_steady,'Value',0)
% Hint: get(hObject,'Value') returns toggle state of motion_unsteady



function vid_loop_Callback(hObject, eventdata, handles)
% hObject    handle to vid_loop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of vid_loop as text
%        str2double(get(hObject,'String')) returns contents of vid_loop as a double


% --- Executes during object creation, after setting all properties.
function vid_loop_CreateFcn(hObject, eventdata, handles)
% hObject    handle to vid_loop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function load_webcam_Callback(hObject, eventdata, handles)
% hObject    handle to load_webcam (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
data=guidata(hObject);
if isfield(data,'cam');
    
delete(data.cam); 
data=rmfield(data,'cam');
    
else
    
data.cam = webcam;
I00=snapshot(data.cam);
imshow(I00,'Parent',handles.axes1);
data.isvid=0;
data.path='./';
data.fname='0';% name of the web cam 
data.BND='';
    set(handles.TrackMenu,'Enable','On');
    set(handles.Start,'Enable','On');
    set(handles.Stop,'Enable','On');
% Defaut values
    set(handles.ROIxmin,'String','1');
    set(handles.ROIxmax,'String',num2str(size(I00,2),'%i'));
    set(handles.ROIymin,'String','1');
    set(handles.ROIymax,'String',num2str(size(I00,1),'%i'));
    drawnow
% Load Parameter filemenu if exist
par=struct();
filename_par=['./0_par.txt'];
if exist(filename_par)
    par=load_parfile(filename_par);
    % Update handles with par data
    if isfield(par,'vid_loop'), set(handles.vid_loop,'String',num2str(par.vid_loop)); end
    if isfield(par,'ROIxmin'), set(handles.ROIxmin,'String',num2str(par.ROIxmin)); end
    if isfield(par,'ROIxmax'), set(handles.ROIxmax,'String',num2str(par.ROIxmax)); end
    if isfield(par,'ROIymin'), set(handles.ROIymin,'String',num2str(par.ROIymin)); end
    if isfield(par,'ROIymax'), set(handles.ROIymax,'String',num2str(par.ROIymax)); end
    if isfield(par,'peak_th'), set(handles.bw,'String',num2str(par.peak_th)); end
    if isfield(par,'peak_th_auto'), set(handles.BWAuto,'Value',par.peak_th_auto==1); end
    if isfield(par,'peak_neigh'), set(handles.neigh,'String',num2str(par.peak_neigh)); end
    if isfield(par,'peak_conv_size'), set(handles.KerSize,'String',num2str(par.peak_conv_size)); end
    if isfield(par,'peak_conv'), set(handles.ConvKern,'Value',par.peak_conv); end
    if isfield(par,'peak_subpix'), set(handles.subpix,'Value',par.peak_subpix); end
    if isfield(par,'peak_minima'), set(handles.MinimaOn,'Value',par.peak_minima==1); set(handles.MinimaOff,'Value',par.peak_minima==0); end
    if isfield(par,'motion'), set(handles.MotionOn,'Value',par.motion==1); end
    if isfield(par,'motion_av'), set(handles.MotionFilt,'String',num2str(par.motion_av)); end
    if isfield(par,'motion_it'), set(handles.motion_it,'String',num2str(par.motion_it)); end
    if isfield(par,'motion_steady'), set(handles.motion_steady,'Value',par.motion_steady);  set(handles.motion_unsteady,'Value',~par.motion_steady); end
    if isfield(par,'noise_size'), set(handles.noisefilt,'String',num2str(par.noise_size)); end
    if isfield(par,'noise'), set(handles.NoiseOn,'Value',par.noise==1); end
    if isfield(par,'BG_speed'), set(handles.BGspeed,'String',num2str(par.BG_speed)); end
    if isfield(par,'BG'), set(handles.BgOn,'Value',par.BG==1); end
    if isfield(par,'filter_time'), set(handles.udt,'String',num2str(par.filter_time)); end
    if isfield(par,'filter_std'), set(handles.nstd,'String',num2str(par.filter_std)); end
    if isfield(par,'filter'), set(handles.OutliersOn,'Value',par.filter==1); end

    set(handles.Info,'String',sprintf('Parameter file read OK !'));
    drawnow
    
else
set(handles.Info,'String',sprintf('No parameter file found, using defaut values!'));
end
end
guidata(hObject,data);


% --- Executes on button press in histOn.
function histOn_Callback(hObject, eventdata, handles)
% hObject    handle to histOn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of histOn


% --- Executes on button press in PlotID.
function PlotID_Callback(hObject, eventdata, handles)
% hObject    handle to PlotID (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if get(handles.PlotID,'Value')
    set(handles.PlotPoints,'Value',0);
   set(handles.PlotError,'Value',0);
end
% Hint: get(hObject,'Value') returns toggle state of PlotID



function PlotTail_Callback(hObject, eventdata, handles)
% hObject    handle to PlotTail (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of PlotTail as text
%        str2double(get(hObject,'String')) returns contents of PlotTail as a double

set(handles.PlotTail,'String',num2str(min(100,max(2,ceil(str2double(get(handles.PlotTail,'String')))))));

% --- Executes during object creation, after setting all properties.
function PlotTail_CreateFcn(hObject, eventdata, handles)
% hObject    handle to PlotTail (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
