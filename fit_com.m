clc
clear all
close all
%load track_bead_all
%track_bead_all=unnamed;
filename=input('What is the file name');
low_thresh=input('threshold');
min_area=input('min_area');
frame_number_for_test=input('last_frame');

xyloObj = VideoReader(filename);             %imput the video
nFrames = xyloObj.NumberOfFrames;           % No of frames
fps=xyloObj.FrameRate;
height_vid = xyloObj.Height;
width_vid = xyloObj.Width;
video_frame=read(xyloObj,10);% size(video_frame)
tipanglrot = rgb2gray(video_frame) ;
% size(A)

% figure(1)
% imshow(video_frame)

BWvideo=zeros(height_vid,width_vid,1);
size(BWvideo);
%% plots cell in all frames, re-orients the cell and asks for the position of the kink
counter=0;
yum=1:frame_number_for_test;
tipanglrot=zeros(yum,1);
tip_posx=zeros(yum,1);
 tip_posy=zeros(yum,1);
 Medx1=zeros(yum,1);
 Medy1=zeros(yum,1);
 Mex1=zeros(yum,1);
 Mey1=zeros(yum,1);
 diff_x2=zeros(yum,1);
 diff_y2=zeros(yum,1);
 COM_x_coord5=zeros(yum,1);
 COM_y_coord5=zeros(yum,1);
 COM_x_coord7=zeros(yum,1);
 COM_y_coord7=zeros(yum,1);
 COM_bumpy_before_rot_x1=zeros(yum,1);
 COM_bumpy_before_rot_y1=zeros(yum,1);
 COM_bumpy_x2=zeros(yum,1);
 COM_bumpy_y2=zeros(yum,1);
 diff_bumpy_x3=zeros(yum,1);
 diff_bumpy_y3=zeros(yum,1);
 COM_b4bump_b4rot_x1=zeros(yum,1);
 COM_b4bump_b4rot_x1=zeros(yum,1);
 Theta_Rad_4_1=zeros(yum,1);
 Theta_Degrees_4_1=zeros(yum,1);
 COM_4_x2=zeros(yum,1);
 COM_4_y2=zeros(yum,1);
 
counter4=0
%%
 for  ii=1:frame_number_for_test;
    BWvideoframe_ii_lowthresh=im2bw(rgb2gray(read(xyloObj,ii)),low_thresh); %ii=frame_number_for_test:frame_number_for_test % this was thib version
    
    BWvideoframe_ii_lowthresh_minus_small1 = bwareaopen(BWvideoframe_ii_lowthresh, min_area); % removes objects smaller than min_area (ie not cells)
    
    BWvideoframe_ii_lowthresh_minus_small = bwareaopen(BWvideoframe_ii_lowthresh_minus_small1, min_area); 
    
    % filtering original image
    myfilter = fspecial('gaussian',[3 3], 4.5);
    BWvideoframe_ii_lowthresh_minus_small = imfilter(BWvideoframe_ii_lowthresh_minus_small, myfilter, 'replicate');
    
    clearvars myfilter
    clearvars BWvideoframe_ii_lowthresh
    clearvars BWvideoframe_ii_lowthresh_minus_small1
    [L, num] = bwlabel(BWvideoframe_ii_lowthresh_minus_small, 4);
    STATS = regionprops(BWvideoframe_ii_lowthresh_minus_small, 'Area','Centroid','Perimeter');
    BWvideoframe_ii_lowthresh_minus_small_perimeter = bwperim(BWvideoframe_ii_lowthresh_minus_small,8);
 %% align cell the 1st -90 rotation: beacuse of the movie analyzed in matlab and track from imagej
     [x_coord1,y_coord1]= find(BWvideoframe_ii_lowthresh_minus_small==1);
     
     clearvars BWvideoframe_ii_lowthresh_minus_small
     
 % Rotating the cell by -90 degrees so that it aligns with bead track that is plotted as (x, -y)    
     thetan=-pi/2;
x_coord = x_coord1.*cos(thetan) - y_coord1.*sin(thetan);
y_coord = x_coord1.*sin(thetan) + y_coord1.*cos(thetan);
 
clearvars x_coord1
clearvars y_coord1
%% Mean of the scatter after the -90 rotation
    Mex=mean(x_coord);
    Mex1(ii,1)=Mex;
    
    Mey=mean(y_coord);
    Mey1(ii,1)=Mey;
    COM=[Mex1,Mey1];
   
    %% Limagine
    Limagine=[200,-200];
    Limagine1=repmat(Limagine,ii,1);
     


%% Number of points before com and creating variable with points before com
    number_points_before_COM=0;
    for jj=1:length(y_coord)
        if y_coord(jj) <= round(Mey1)
            number_points_before_COM=number_points_before_COM+1;
        end
    end
    
 
    
 % loads data before com in new variable   
 before_COM_x=zeros(number_points_before_COM,1);
 before_COM_y=zeros(number_points_before_COM,1);
 %after_min_x=zeros(number_points_after_ymin,1);
 %after_min_y=zeros(number_points_after_ymin,1);
 
 counter_points=0;
    for jj=1:length(y_coord)
         if y_coord(jj) <= round(Mey1)
             counter_points=counter_points+1;
             before_COMX_x(counter_points)=x_coord(jj);
             before_COMY_y(counter_points)=y_coord(jj);
         end
    end
    
    counter_points
    ppx=(before_COMX_x)';
    ppy=(before_COMY_y)';
    
   
 %% fit line
    fit_before_COM=fit(ppx,ppy,'poly1');
    %plot(fit_before_COM)
    paramm_fit_before_COM=double(coeffvalues(fit_before_COM));
    %%
%     paramm_fit_before_COM=double(coeffvalues(fit_before_COM));
%     
%    % getting real numbers for the line fit
%     qq=0
%     for ql=y_min_spline:0.05:Mey1;
% qq=qq+1;
% x_fit_line(qq)=ql;
% y_fit_line(qq)=fit_before_COM(1)*ql + fit_before_COM(2);
%     end
    %scatter(ppx,ppy,'k')
    %plot(x_fit_line,y_fit_line)
    
%%
% gets angle of fitted line
ThetaInRads_before_com = atan(paramm_fit_before_COM(1))*(-1);
ThetaInDegrees_before_com = atan(paramm_fit_before_COM(1))*(180/pi)*(-1);
Theta_fit_rad(ii,1)=ThetaInRads_before_com;
Theta_fit_deg(ii,1)=ThetaInDegrees_before_com;


       
    
    %% rotate every point with fit theta 
%     
% Theta_fit_rad_rep=repmat((Theta_fit_rad(end,1)),(length(x_coord)),1);
% x_coord4 = x_coord.*cos(Theta_fit_rad_rep) -  y_coord.*sin(Theta_fit_rad_rep);
% y_coord4 = x_coord.*sin(Theta_fit_rad_rep) +  y_coord.*cos(Theta_fit_rad_rep);

Theta_fit_rad_rep=repmat((Theta_fit_rad(end,1)),(length(x_coord)),1);
x_coord4 = x_coord.*cos(Theta_fit_rad_rep) -  y_coord.*sin(Theta_fit_rad_rep);
y_coord4 = x_coord.*sin(Theta_fit_rad_rep) +  y_coord.*cos(Theta_fit_rad_rep);

clearvars x_coord
clearvars y_coord

%% COM after rotation of cell
COM_x_coord4 = mean(x_coord4);
COM_y_coord4 = mean(y_coord4);
COM_x_coord5(ii,1)=COM_x_coord4;
COM_y_coord5(ii,1)=COM_y_coord4;
COM_5=[COM_x_coord5,COM_y_coord5];

%% finding diff between COM of the rotated scatter and 200,-200
    
diff=Limagine1-COM_5;

diff_x1 = diff(ii,1);
diff_y1 = diff(ii,2);
%     
hh=length(x_coord4);
diff_x_rep = repmat(diff_x1,hh,1);
diff_y_rep = repmat(diff_y1,hh,1);
% New x and y coord after position correction    
x_coord6 = x_coord4 + diff_x_rep;
y_coord6 = y_coord4 + diff_y_rep;

clearvars x_coord4
clearvars y_coord4
clearvars diff_x_rep
clearvars diff_y_rep
%     
diff_x2(ii,1)=diff_x1;
diff_y2(ii,1)=diff_y1;
diff_cell_eachframe =[diff_x2,diff_y2];

% Calculating COM of the position corrected (rotation was done on it earlier)
COM_x_coord6 = mean(x_coord6);
COM_y_coord6 = mean(y_coord6);
COM_x_coord7(ii,1)=COM_x_coord6;
COM_y_coord7(ii,1)=COM_y_coord6;
COM_7=[COM_x_coord7,COM_y_coord7];
   close 
if ii == 1;
 close
%figure
scatter(x_coord6,y_coord6,'g')
hold on
axis equal
plot(COM_x_coord7,COM_y_coord7,'r*')
%close
end

%% plotting for kink selection
%close
%figure
%scatter(x_coord6,y_coord6,'g')
%hold on
%plot(COM_x_coord7,COM_y_coord7,'r*')
%plot(bead_rot_pos(:,1),bead_rot_pos(:,2),'r')
%axis equal
% 
%% To correct for motion of the right kink of the cell, pick a point near the right kink of cell
% % Manual selection of kink position
if ii == 1;
   [after_kink_position_x,after_kink_position_y] = ginput(1);
   
end

%% selecting points x, y after the selection

number_points_after_kink=0;
count_if = 0;
    for jo=1:length(y_coord6)
        if x_coord6(jo) >= round(after_kink_position_x)
            number_points_after_kink=number_points_after_kink+1;
        end
    end
    
region_after_kink_x=zeros(number_points_after_kink,1);
region_after_kink_y=zeros(number_points_after_kink,1);

counter_kink=0;
    for jl=1:length(y_coord6)
         if x_coord6(jl) >= round(after_kink_position_x)
             counter_kink=counter_kink+1;
             region_after_kink_x(counter_kink)=x_coord6(jl);
             region_after_kink_y(counter_kink)=y_coord6(jl);
         end
         counter_kink;
    end
    
    
    if length(region_after_kink_x) > 0
    
    bumpy_x=(region_after_kink_x);
    bumpy_y=(region_after_kink_y);
    COM_bumpy_before_rot_x = mean(bumpy_x);
    COM_bumpy_before_rot_y = mean(bumpy_y);
     COM_bumpy_before_rot_x1(ii,1) = COM_bumpy_before_rot_x;
     COM_bumpy_before_rot_y1(ii,1) = COM_bumpy_before_rot_y;
     COM_bumpy_before_rot = [COM_bumpy_before_rot_x1,COM_bumpy_before_rot_y1];
    
    
    
    %% fit line on the after kink region
   % fit line
    fit_kink=fit(bumpy_x,bumpy_y,'poly1');
    paramm_fit_kink=double(coeffvalues(fit_kink));
%close
%figure
  %scatter(bumpy_x,bumpy_y,'b')
  
  
%% angle of the fit kink
  
ThetaInRads_kink = atan(paramm_fit_kink(1))*(-1);
ThetaInDegrees_kink = atan(paramm_fit_kink(1))*(180/pi)*(-1);
Theta_fitkink_rad(ii,1) = ThetaInRads_kink;
Theta_fitkink_deg(ii,1) = ThetaInDegrees_kink;

%% rotate 2nd time
Theta_fitkink_rad_rep=repmat((Theta_fitkink_rad(end,1)),(length(bumpy_x)),1);

bumpy_x1 = bumpy_x.*cos(Theta_fitkink_rad_rep) -  bumpy_y.*sin(Theta_fitkink_rad_rep);
bumpy_y1 = bumpy_x.*sin(Theta_fitkink_rad_rep) +  bumpy_y.*cos(Theta_fitkink_rad_rep);

%% COM after rotation of bumpy region
COM_bumpy_x1 = mean(bumpy_x1);
COM_bumpy_y1 = mean(bumpy_y1);
COM_bumpy1=[COM_bumpy_x1,COM_bumpy_y1];

COM_bumpy_x2(ii,1)=COM_bumpy_x1;
COM_bumpy_y2(ii,1)=COM_bumpy_y1;
COM_bumpy2_after_rot = [COM_bumpy_x2,COM_bumpy_y2]; % diff from selected point, frame by frame

diff_COM_bumpy =  COM_bumpy_before_rot - COM_bumpy2_after_rot;


% %% Creating a new variable with bumpy rep 
%  L_im_at_bumpy_x = bumpy_x(ii);  
%  L_im_at_bumpy_y = bumpy_y(ii);
%  
%   %L_im_at_bumpy_x_rep =  repmat(L_im_at_bumpy_x,length(bumpy_x1),1);
%   %L_im_at_bumpy_y_rep =  repmat(L_im_at_bumpy_y,length(bumpy_y1),1);
%   
%   L_im_at_bumpy = [L_im_at_bumpy_x,L_im_at_bumpy_y];
%  
% diff_bumpy = L_im_at_bumpy - COM_bumpy1;

%% adding diff b/w com b4 rot and after rot to each point of the scatter
diff_bumpy_x2 = diff_COM_bumpy(ii,1);
diff_bumpy_y2 = diff_COM_bumpy(ii,2);
%     
hh=length(bumpy_x);
diff_x_bumpy_rep = repmat(diff_bumpy_x2,hh,1);
diff_y_bumpy_rep = repmat(diff_bumpy_y2,hh,1);

% New x and y coord after bumpy correction    
bumpy_x3 = bumpy_x1 + diff_x_bumpy_rep;
bumpy_y3 = bumpy_y1 + diff_y_bumpy_rep;
%     
% diff_bumpy_x3(ii,1)=diff_bumpy_x2;
% diff_bumpy_y3(ii,1)=diff_bumpy_y2;
% diff_bumpy_eachframe =[diff_bumpy_x3,diff_bumpy_y3];

%% 

%   hold on
%     plot(fit_kink,'r')
%     hold on
% plot(bumpy_x3,bumpy_y3,'m')
% plot(x_coord6,y_coord6,'r')
%     axis equal
%    
    
 count_if = count_if+1 
    end
    %
    %
    %
    %% 4th rotation before the selected point (kink)
     number_points_4=0;
count_if5=0;
    for jo=1:length(y_coord6)
        if x_coord6(jo) <= 175
            number_points_4= number_points_4 + 1;
        end
    end
    
region_4_x=zeros(number_points_4,1);
region_4_y=zeros(number_points_4,1);

counter_kink5=0;
    for jl=1:length(x_coord6)
         if x_coord6(jl) <= 175
             counter_kink5=counter_kink5+1;
             region_4_x(counter_kink5)=x_coord6(jl);
             region_4_y(counter_kink5)=y_coord6(jl);
         end
    end
  %%   
    counter_kink5;
     
      %length(region_4_x) > 0
         
    x_4=(region_4_x);
    y_4=(region_4_y);
    COM_4_b4rot_x = mean(x_4);
    COM_4_b4rot_y= mean(y_4);
    COM_4_b4rot_x1(ii,1) = COM_4_b4rot_x;
    COM_4_b4rot_y1(ii,1) = COM_4_b4rot_y;
    COM_4_b4rot= [COM_4_b4rot_x1,COM_4_b4rot_y1];
%% fit line on the after kink region
   % fit line
    fit_4=fit(x_4, y_4,'poly1');
    paramm_fit_4=double(coeffvalues(fit_4));

%figure
  %scatter(before_bump_x,before_bump_y,'b')
  
  %plot(fit_4, 'r') 
%% gets angle of fitted line, NOTICE: THE *(-1) IS (NOT?) THERE
ThetaInRads_4 = atan(paramm_fit_4(1))*(-1);
ThetaInDegrees_4 = atan(paramm_fit_4(1))*(180/pi)*(-1);
Theta_Rad_4_1(ii,1)=ThetaInRads_4;
Theta_Degrees_4_1(ii,1)= ThetaInDegrees_4;   
           
%% rotate 3rd time X, Y with fit theta 
%     
 Theta_Rad_4_rep = repmat((Theta_Rad_4_1(end,1)),(length(x_4)),1);

x1_4 = x_4.*cos(Theta_Rad_4_rep) -  y_4.*sin(Theta_Rad_4_rep);
y1_4 = x_4.*sin(Theta_Rad_4_rep) +  y_4.*cos(Theta_Rad_4_rep);
     
     
%% COM after rotation of bumpy region
COM_x1_4 = mean(x1_4);
COM_y1_4 = mean(y1_4);
COM_4=[COM_x1_4,COM_y1_4];

COM_4_x2(ii,1)=COM_x1_4;
COM_4_y2(ii,1)=COM_y1_4;
COM_4_after_rot = [COM_4_x2,COM_4_y2]; % diff from selected point, frame by frame

diff_COM_4 =  COM_4_b4rot - COM_4_after_rot;

%% adding diff b/w com b4 rot and after rot to each point of the scatter
diff_4_x = diff_COM_4(ii,1);
diff_4_y = diff_COM_4(ii,2);
%     
hh=length(diff_4_x);
diff_4_x_rep = repmat(diff_4_x,hh,1);
diff_4_y_rep = repmat(diff_4_y,hh,1);

% New x and y coord after correction    
x1_4_2_add = x1_4 + diff_4_x;
%x1_4_2_sub = x1_4 - diff_4_x;
%y1_4_2_sub = y1_4 - diff_4_y;
y1_4_2_add = y1_4 + diff_4_y;
counter4=counter4+1
 end 
     
 count_if
  'aap_mahaan_ho' 
  'ab_individual_frame_correct_karen?'
 %scatter(before_bump_x1,before_bump_y1,'k')
    %hold on
    %plot(fit_before_COM)
    
    
    %plot(x_coord4,y_coord4)
%hold on
%plot(x_coord4,y_coord4,'r')

%% COORD of the bead
 




%% select all points after the kink


