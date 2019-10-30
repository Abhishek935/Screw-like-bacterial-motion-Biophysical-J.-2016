function test_fit_spline(filename,low_thresh,min_area,frame_number_for_test);

% Amov = dir ('*.avi');
% a=numel(Amov);
% kewl=100;
close all
% filename = getfield (Amov,{i,1},'name')
xyloObj = VideoReader(filename);             %imput the video
nFrames = xyloObj.NumberOfFrames;           % No of frames
fps=xyloObj.FrameRate;
height_vid = xyloObj.Height;
width_vid = xyloObj.Width;

video_frame=read(xyloObj,10);
% size(video_frame)

A = rgb2gray(video_frame) ;
% size(A)

% figure(1)
% imshow(video_frame)

BWvideo=zeros(height_vid,width_vid,1);
size(BWvideo);




%% plots cell in frame 25, re-orients the cell and asks for the position of the kink
for ii=frame_number_for_test:frame_number_for_test
	BWvideoframe_ii_lowthresh=im2bw(rgb2gray(read(xyloObj,ii)),low_thresh);
    BWvideoframe_ii_lowthresh_minus_small = bwareaopen(BWvideoframe_ii_lowthresh, min_area); % removes objects smaller than min_area (ie not cells)
    % filtering original image
    myfilter = fspecial('gaussian',[3 3], 4.5);
    BWvideoframe_ii_lowthresh_minus_small = imfilter(BWvideoframe_ii_lowthresh_minus_small, myfilter, 'replicate');
    
    [L, num] = bwlabel(BWvideoframe_ii_lowthresh_minus_small, 4);
    STATS = regionprops(BWvideoframe_ii_lowthresh_minus_small, 'Area','Centroid','Perimeter');
    BWvideoframe_ii_lowthresh_minus_small_perimeter = bwperim(BWvideoframe_ii_lowthresh_minus_small,8);
    
    %% align cell
%     figure(222)
%     hold on
    [x_coord,y_coord]= find(BWvideoframe_ii_lowthresh_minus_small==1);
    Lmagine=[1,0];

%     % find first and last point position
%     dist_to_zero=10000;%in pixels
%     for jj=1:length(x_coord)
%         norm_cell(jj)=norm(y_coord(jj),x_coord(jj));
%         if norm_cell(jj) < dist_to_zero
%             dist_to_zero=norm_cell(jj);
%             index_min_norm=jj;
%         end
% 
%     end
%     tail_cell=[y_coord(index_min_norm),x_coord(index_min_norm)];
%     dist_to_zero=0;%in pixels
%     for jj=1:length(x_coord)
%         norm_cell(jj)=norm(y_coord(jj),x_coord(jj));
%         if norm_cell(jj) > dist_to_zero
%             dist_to_zero=norm_cell(jj);
%             index_max_norm=jj;
%         end
% 
%     end
%     tip_cell=[y_coord(index_max_norm),x_coord(index_max_norm)];
%     CosTheta = dot(Lmagine,tip_cell)/(norm(Lmagine)*norm(tip_cell));
%     ThetaInRads = acos(CosTheta);
%     ThetaInDegrees = acos(CosTheta)*180/pi;
% 
%     x_prime = x_coord.*cos(ThetaInRads) - y_coord.*sin(ThetaInRads);
%     y_prime = x_coord.*sin(ThetaInRads) + y_coord.*cos(ThetaInRads);
% 
%     % find first and last point position
%     dist_to_zero=10000;%in pixels
%     for jj=1:length(x_coord)
%         norm_cell(jj)=norm(y_prime(jj),x_prime(jj));
%         if norm_cell(jj) < dist_to_zero
%             dist_to_zero=norm_cell(jj);
%             index_min_norm=jj;
%         end
%     end
%     dist_to_zero=0;%in pixels
%     for jj=1:length(x_prime)
%         norm_cell(jj)=norm(y_prime(jj),x_prime(jj));
%         if norm_cell(jj) > dist_to_zero
%             dist_to_zero=norm_cell(jj);
%             index_max_norm=jj;
%         end
%     end
%     tip_cell=[y_prime(index_max_norm),x_prime(index_max_norm)];
% 
% 
% %     x_prime=x_prime-tip_cell(2);
% %     y_prime=y_prime-tip_cell(1);
% 
% %     scatter(y_prime,x_prime,'b')
% %     scatter(y_prime(index_max_norm),x_prime(index_max_norm),'r')
% %     scatter(y_prime(index_min_norm),x_prime(index_min_norm),'g')
    
    
%     figure(333)
%     hold on
%     scatter(y_coord,x_coord,'r')
    
    
%     break
    
    
    fit_intermediaire = fit(y_coord,x_coord,'poly1');
    paramm_intermediaire=double(coeffvalues(fit_intermediaire));
    
    
%     figure(111)
%     hold on
    
%     scatter(limited_region_after_kink_y,limited_region_after_kink_x,'b')
%     plot(fit_after_kink,'r')

    % plots fit for data after kink
    y_min_fit_intermedaire=min(y_coord);
    y_max_fit_intermedaire=max(y_coord);
    kk=0;
    for ll=y_min_fit_intermedaire:0.05:y_max_fit_intermedaire
        kk=kk+1;
        x_fit_intermedaire(kk)=ll;
        y_fit_intermedaire(kk)=paramm_intermediaire(1)*ll + paramm_intermediaire(2); % order 1
    end
%     plot(x_fit_intermedaire,y_fit_intermedaire,'k')
    
    x_min_fit_intermedaire=paramm_intermediaire(1)*y_min_fit_intermedaire + paramm_intermediaire(2); %Limits the two extremes of the cell
    x_max_fit_intermedaire=paramm_intermediaire(1)*y_max_fit_intermedaire + paramm_intermediaire(2);
%     scatter(y_min_fit_intermedaire,x_min_fit_intermedaire,'b')
%     scatter(y_max_fit_intermedaire,x_max_fit_intermedaire,'g')
    negative_angle=1;
    if (x_max_fit_intermedaire-x_min_fit_intermedaire) <= 0
        negative_angle=-1;
    end
    
    y_max_fit_intermedaire=y_max_fit_intermedaire-y_min_fit_intermedaire;
    x_max_fit_intermedaire=x_max_fit_intermedaire-x_min_fit_intermedaire;
    
%     scatter(y_max_fit_intermedaire,x_max_fit_intermedaire,'k')
    
    tip_cell_intermediaire=[y_max_fit_intermedaire,x_max_fit_intermedaire];
    
    CosTheta_intermediaire = dot(Lmagine,tip_cell_intermediaire)/(norm(Lmagine)*norm(tip_cell_intermediaire));
    ThetaInRads_intermediaire = negative_angle*acos(CosTheta_intermediaire);
    ThetaInDegrees_intermediaire = negative_angle*acos(CosTheta_intermediaire)*180/pi;

    x_prime_intermediaire2 = x_coord.*cos(ThetaInRads_intermediaire) - y_coord.*sin(ThetaInRads_intermediaire); % the 1st crude rotation
    y_prime_intermediaire2 = x_coord.*sin(ThetaInRads_intermediaire) + y_coord.*cos(ThetaInRads_intermediaire);
    
    y_prime_intermediaire2=y_prime_intermediaire2-max(y_prime_intermediaire2);
    
    
    figure(15)
    scatter(y_prime_intermediaire2,x_prime_intermediaire2,'g')
    
    axis equal % changes the scale so it's the same on both axes
%
%     figure(5)
%     scatter(y_prime,x_prime,'b')
%     hold on
%     scatter(y_prime(index_max_norm),x_prime(index_max_norm),'r')
%     scatter(y_prime(index_min_norm),x_prime(index_min_norm),'g')

    %% manual selection of kink region
    [before_kink_position_y,before_kink_position_x] = ginput(1);
    [kink_position_y,kink_position_x] = ginput(1);
    close all
end



% return




time_frame=zeros(nFrames,1);
position_kink_y=zeros(nFrames,1);

MAT_file_name = filename(1:end-4)
savefilename=strcat(MAT_file_name,'_data.mat');


%% re-orients and aligns cell for each frame, fits entire cell
for ii=1:nFrames
    
%     
    %% opens frame
    BWvideoframe_ii_lowthresh=im2bw(rgb2gray(read(xyloObj,ii)),low_thresh);
    BWvideoframe_ii_lowthresh_minus_small = bwareaopen(BWvideoframe_ii_lowthresh, min_area); % removes objects smaller than min_area (ie not cells)
    % filtering original image
    myfilter = fspecial('gaussian',[3 3], 4.5);
    BWvideoframe_ii_lowthresh_minus_small = imfilter(BWvideoframe_ii_lowthresh_minus_small, myfilter, 'replicate');
    
    [L, num] = bwlabel(BWvideoframe_ii_lowthresh_minus_small, 4);
    STATS = regionprops(BWvideoframe_ii_lowthresh_minus_small, 'Area','Centroid','Perimeter');
    BWvideoframe_ii_lowthresh_minus_small_perimeter = bwperim(BWvideoframe_ii_lowthresh_minus_small,8);
    
    
    %% align cell
%     figure(222)
%     hold on
    [x_coord,y_coord]= find(BWvideoframe_ii_lowthresh_minus_small==1);
    Lmagine=[1,0];
    
    
    
    
    
    
    
    
    
    
    
    
    
    fit_intermediaire = fit(y_coord,x_coord,'poly1');
    paramm_intermediaire=double(coeffvalues(fit_intermediaire)); % another rough fit on each frame
    
    
%     figure(111)
%     hold on
    
%     scatter(limited_region_after_kink_y,limited_region_after_kink_x,'b')
%     plot(fit_after_kink,'r')

    % plots fit for data after kink
    y_min_fit_intermedaire=min(y_coord);
    y_max_fit_intermedaire=max(y_coord);
    kk=0;
    for ll=y_min_fit_intermedaire:0.05:y_max_fit_intermedaire
        kk=kk+1;
        x_fit_intermedaire(kk)=ll;
%        y_fit(kk)=paramm(1)*jj^6 + paramm(2)*jj^5 + paramm(3)*jj^4 + paramm(4)*jj^3 + paramm(5)*jj^2 + paramm(6)*jj + paramm(7); % order 6
%         y_fit(kk)=paramm(1)*jj^5 + paramm(2)*jj^4 + paramm(3)*jj^3 + paramm(4)*jj^2 + paramm(5)*jj + paramm(6); % order 5
        y_fit_intermedaire(kk)=paramm_intermediaire(1)*ll + paramm_intermediaire(2); % order 4
%        y_fit(kk)=paramm(1)*jj^3 + paramm(2)*jj^2 + paramm(3)*jj + paramm(4); % order 3
    
    end
%     plot(x_fit_intermedaire,y_fit_intermedaire,'k')
    
    x_min_fit_intermedaire=paramm_intermediaire(1)*y_min_fit_intermedaire + paramm_intermediaire(2);
    x_max_fit_intermedaire=paramm_intermediaire(1)*y_max_fit_intermedaire + paramm_intermediaire(2);
%     scatter(y_min_fit_intermedaire,x_min_fit_intermedaire,'b')
%     scatter(y_max_fit_intermedaire,x_max_fit_intermedaire,'g')
    negative_angle=1;
    if (x_max_fit_intermedaire-x_min_fit_intermedaire) <= 0
        negative_angle=-1;
    end
    
    y_max_fit_intermedaire=y_max_fit_intermedaire-y_min_fit_intermedaire;
    x_max_fit_intermedaire=x_max_fit_intermedaire-x_min_fit_intermedaire;
    
%     scatter(y_max_fit_intermedaire,x_max_fit_intermedaire,'k')
    
    tip_cell_intermediaire=[y_max_fit_intermedaire,x_max_fit_intermedaire];
    
    CosTheta_intermediaire = dot(Lmagine,tip_cell_intermediaire)/(norm(Lmagine)*norm(tip_cell_intermediaire));
    ThetaInRads_intermediaire = negative_angle*acos(CosTheta_intermediaire);
    ThetaInDegrees_intermediaire = negative_angle*acos(CosTheta_intermediaire)*180/pi;

    x_prime_intermediaire2 = x_coord.*cos(ThetaInRads_intermediaire) - y_coord.*sin(ThetaInRads_intermediaire);
    y_prime_intermediaire2 = x_coord.*sin(ThetaInRads_intermediaire) + y_coord.*cos(ThetaInRads_intermediaire);
    
    
%     close
    
    
    % aligns cell and tip and tail points so that tip is around (0,0)
    before_kink_position_x=0;
    kink_position_x=0;
    
    figure(9)
    hold on
%     scatter(y_prime_intermediaire2,x_prime_intermediaire2,'b')
% 	scatter(before_kink_position_y,before_kink_position_x,'r')
% 	scatter(kink_position_y,kink_position_x,'g')
    
    x_prime_intermediaire2 = x_prime_intermediaire2 - mean(x_prime_intermediaire2);
    y_prime_intermediaire2=y_prime_intermediaire2-max(y_prime_intermediaire2);
    
%     figure(9)
%     hold on
%     scatter(y_prime_intermediaire2,x_prime_intermediaire2,'b')
% 	scatter(before_kink_position_y,before_kink_position_x,'r')
% 	scatter(kink_position_y,kink_position_x,'g')
    

    % find first and last point position
    dist_to_zero=10000;%in pixels
    for jj=1:length(x_prime_intermediaire2)
        dist_point=sqrt((x_prime_intermediaire2(jj))^2+(y_prime_intermediaire2(jj))^2);
        if dist_point < dist_to_zero
            dist_to_zero=dist_point;
            index_min_norm=jj;
        end

    end
    tail_cell=[y_prime_intermediaire2(index_min_norm),x_prime_intermediaire2(index_min_norm)];
    dist_to_zero=0;%in pixels
    for jj=1:length(x_prime_intermediaire2)
        dist_point=sqrt((x_prime_intermediaire2(jj))^2+(y_prime_intermediaire2(jj))^2);
        if dist_point > dist_to_zero
            dist_to_zero=dist_point;
            index_max_norm=jj;
        end

    end
    tip_cell=[y_prime_intermediaire2(index_max_norm),x_prime_intermediaire2(index_max_norm)];

    
    % number of points before kink
    number_points_before_kink=0;
    for jj=1:length(y_prime_intermediaire2)
        if y_prime_intermediaire2(jj) <= round(before_kink_position_y)
            number_points_before_kink=number_points_before_kink+1;
        end
    end
    
    % number of points after kink
    number_points_after_kink=0;
    for jj=1:length(y_prime_intermediaire2)
        if y_prime_intermediaire2(jj) >= round(kink_position_y)
            number_points_after_kink=number_points_after_kink+1;
        end
    end
    
    
    % loads data before and after kink in new variable
    limited_region_before_kink_x=zeros(number_points_before_kink,1);
    limited_region_before_kink_y=zeros(number_points_before_kink,1);
    limited_region_after_kink_x=zeros(number_points_after_kink,1);
    limited_region_after_kink_y=zeros(number_points_after_kink,1);
    
    counter_points=0;
    for jj=1:length(y_prime_intermediaire2)
        if y_prime_intermediaire2(jj) <= round(before_kink_position_y)
            counter_points=counter_points+1;
            limited_region_before_kink_x(counter_points)=x_prime_intermediaire2(jj);
            limited_region_before_kink_y(counter_points)=y_prime_intermediaire2(jj);
        end
    end
    counter_points
    figure(1)
%     hold on

    scatter(y_prime_intermediaire2,x_prime_intermediaire2,'b')
    round(before_kink_position_y)
    counter_points=0;
    for jj=1:length(y_prime_intermediaire2)
        if y_prime_intermediaire2(jj) >= round(kink_position_y)
            counter_points=counter_points+1;
            limited_region_after_kink_x(counter_points)=x_prime_intermediaire2(jj);
            limited_region_after_kink_y(counter_points)=y_prime_intermediaire2(jj);
        end
    end
    
%     break
    
    
    
    
% 	scatter(limited_region_before_kink_y,limited_region_before_kink_x,'r')
% 	scatter(limited_region_after_kink_y,limited_region_after_kink_x,'g')
    
%     scatter(tail_cell(1),tail_cell(2),'r')
%     scatter(tip_cell(1),tip_cell(2),'g')
    
    
    % fit part after kink
    fit_after_kink = fit(limited_region_after_kink_y,limited_region_after_kink_x,'poly1');
    paramm_after_kink=double(coeffvalues(fit_after_kink)); 
    % plots fit for data after kink
    y_min_fit_after_kink=min(limited_region_after_kink_y);
    y_max_fit_after_kink=max(limited_region_after_kink_y);
    kk=0;
    for ll=y_min_fit_after_kink:((y_max_fit_after_kink-y_min_fit_after_kink)/50):y_max_fit_after_kink %?
        kk=kk+1;
        x_fit_after_kink(kk)=ll; %?
        y_fit_after_kink(kk)=paramm_after_kink(1)*ll + paramm_after_kink(2); % order 1?
    end
%     plot(fit_after_kink,'r')
%     plot(x_fit_after_kink,y_fit_after_kink,'k')
    
    
    % gets angle of fitted line
    point_tip_minus_tail(1)=y_min_fit_after_kink-y_max_fit_after_kink;
    point_tip_minus_tail(2)=paramm_after_kink(1)*y_min_fit_after_kink + paramm_after_kink(2)-(paramm_after_kink(1)*y_max_fit_after_kink + paramm_after_kink(2));
%     scatter(point_tip_minus_tail(1),point_tip_minus_tail(2),'k')
%     ThetaInRads_after_kink = acos(dot(Lmagine,point_tip_minus_tail)/(norm(Lmagine)*norm(point_tip_minus_tail)));
%     ThetaInRads_after_kink = atan(paramm_after_kink(1));
    ThetaInRads_after_kink = atan(point_tip_minus_tail(2)/point_tip_minus_tail(1));
    ThetaInDegrees_after_kink = ThetaInRads_after_kink*180/pi;
    
    % rotates fitted line
    x_fit_after_kink_rot = x_fit_after_kink.*cos(ThetaInRads_after_kink) + y_fit_after_kink.*sin(ThetaInRads_after_kink);
    y_fit_after_kink_rot = -x_fit_after_kink.*sin(ThetaInRads_after_kink) + y_fit_after_kink.*cos(ThetaInRads_after_kink);
    
    x_fit_after_kink=x_fit_after_kink_rot;
    y_fit_after_kink=y_fit_after_kink_rot-mean(y_fit_after_kink_rot);
%     clear x_fit_after_kink_rot
%     clear y_fit_after_kink_rot
    figure(9)
    plot(x_fit_after_kink,y_fit_after_kink,'k')
    
    
    % rotates cell shape and aligns it
    x_prime_rot = x_prime_intermediaire2.*cos(ThetaInRads_after_kink) - y_prime_intermediaire2.*sin(ThetaInRads_after_kink);
    y_prime_rot = x_prime_intermediaire2.*sin(ThetaInRads_after_kink) + y_prime_intermediaire2.*cos(ThetaInRads_after_kink);
    x_prime_intermediaire2=x_prime_rot-mean(y_fit_after_kink_rot); % aligns rotated cells around Y=0
    y_prime_intermediaire2=y_prime_rot;
    clear x_prime_rot
    clear y_prime_rot
%     scatter(y_prime_intermediaire2,x_prime_intermediaire2,'b')
    
    
    % rotates cell shape BEFORE KINK and aligns it
    limited_region_before_kink_rot_x = limited_region_before_kink_x.*cos(ThetaInRads_after_kink) - limited_region_before_kink_y.*sin(ThetaInRads_after_kink);
    limited_region_before_kink_rot_y = limited_region_before_kink_x.*sin(ThetaInRads_after_kink) + limited_region_before_kink_y.*cos(ThetaInRads_after_kink);
    limited_region_before_kink_x=limited_region_before_kink_rot_x-mean(y_fit_after_kink_rot); % aligns rotated cells before kink around Y=0
    limited_region_before_kink_y=limited_region_before_kink_rot_y;
    clear limited_region_before_kink_rot_x
    clear limited_region_before_kink_rot_y
%     scatter(limited_region_before_kink_y,limited_region_before_kink_x,'r')
    
    
    % rotates cell shape BEFORE KINK and aligns it
    limited_region_after_kink_rot_x = limited_region_after_kink_x.*cos(ThetaInRads_after_kink) - limited_region_after_kink_y.*sin(ThetaInRads_after_kink);
    limited_region_after_kink_rot_y = limited_region_after_kink_x.*sin(ThetaInRads_after_kink) + limited_region_after_kink_y.*cos(ThetaInRads_after_kink);
    limited_region_after_kink_x=limited_region_after_kink_rot_x-mean(y_fit_after_kink_rot); % aligns rotated cells before kink around Y=0
    limited_region_after_kink_y=limited_region_after_kink_rot_y;
    clear limited_region_after_kink_rot_x
    clear limited_region_after_kink_rot_y
%     scatter(limited_region_after_kink_y,limited_region_after_kink_x,'g')
    clear x_fit_after_kink_rot
    clear y_fit_after_kink_rot
    
    
    
    
    % fit part before kink
    fit_before_kink = fit(limited_region_before_kink_y,limited_region_before_kink_x,'poly1');
    paramm_before_kink=double(coeffvalues(fit_before_kink));
    % plots fit for data before kink
    y_min_fit_before_kink=min(limited_region_before_kink_y);
    y_max_fit_before_kink=max(limited_region_before_kink_y);
    kk=0;
    for ll=y_min_fit_before_kink:((y_max_fit_before_kink-y_min_fit_before_kink)/50):y_max_fit_before_kink
        kk=kk+1;
        x_fit_before_kink(kk)=ll;
        y_fit_before_kink(kk)=paramm_before_kink(1)*ll + paramm_before_kink(2); % order 1
    end
%     plot(fit_before_kink,'r')
    plot(x_fit_before_kink,y_fit_before_kink,'r')
    
    

    time_frame_ii(ii)=ii/fps;
    saved_data.time_frame(ii)=ii/fps;
    
    saved_data.first_point_kink_x(ii)=x_fit_before_kink(1);
    saved_data.first_point_kink_y(ii)=y_fit_before_kink(1);
    saved_data.last_point_kink_interm_x(ii)=x_fit_before_kink(length(x_fit_before_kink));
    saved_data.last_point_kink_interm_y(ii)=y_fit_before_kink(length(y_fit_before_kink));
    
    position_kink_y(ii)=y_fit_before_kink(1);
    
    
    
    
%     if ii == 25
%         break
%     end





%     
%     CosTheta = dot(Lmagine,tip_cell)/(norm(Lmagine)*norm(tip_cell));
%     ThetaInRads = acos(CosTheta);
%     ThetaInDegrees = acos(CosTheta)*180/pi;

%     x_prime = x_coord.*cos(ThetaInRads) - y_coord.*sin(ThetaInRads);
%     y_prime = x_coord.*sin(ThetaInRads) + y_coord.*cos(ThetaInRads);
% 
%     % find first and last point position
%     dist_to_zero=10000;%in pixels
%     for jj=1:length(x_coord)
%         norm_cell(jj)=norm(y_prime(jj),x_prime(jj));
%         if norm_cell(jj) < dist_to_zero
%             dist_to_zero=norm_cell(jj);
%             index_min_norm=jj;
%         end
%     end
%     dist_to_zero=0;%in pixels
%     for jj=1:length(x_prime)
%         norm_cell(jj)=norm(y_prime(jj),x_prime(jj));
%         if norm_cell(jj) > dist_to_zero
%             dist_to_zero=norm_cell(jj);
%             index_max_norm=jj;
%         end
%     end
%     tip_cell=[y_prime(index_max_norm),x_prime(index_max_norm)];
% 
% 
%     x_prime=x_prime-tip_cell(2);
%     y_prime=y_prime-tip_cell(1);
% 
% %     scatter(y_prime(index_max_norm),x_prime(index_max_norm),'r')
% %     scatter(y_prime(index_min_norm),x_prime(index_min_norm),'g')
% %     scatter(y_prime,x_prime,'b')
% % 
% %     figure(5)
% %     scatter(y_prime,x_prime,'b')
% %     hold on
% %     scatter(y_prime(index_max_norm),x_prime(index_max_norm),'r')
% %     scatter(y_prime(index_min_norm),x_prime(index_min_norm),'g')
%     
%     
%     
%     
%     
%     
%     
%     
%     % number of points after kink
%     number_points=0;
%     for jj=1:length(x_prime)
%         if y_prime(jj) >= round(kink_position_y)
%             number_points=number_points+1;
%         end
%     end
%     
%     
%     % loads data after kink in new variable
%     limited_region_after_kink_x=zeros(number_points,1);
%     limited_region_after_kink_y=zeros(number_points,1);
%     
%     number_points=0;
%     for jj=1:length(x_prime)
%         if y_prime(jj) >= round(kink_position_y)
%             number_points=number_points+1;
%             limited_region_after_kink_x(number_points,1)=x_prime(jj);
%             limited_region_after_kink_y(number_points,1)=y_prime(jj);
%         end
%     end
%     
%     
%     
%     % fit of data after kink in order to find angle (fit of the first order)
%     fit_after_kink = fit(limited_region_after_kink_y,limited_region_after_kink_x,'poly1')
%     paramm_after_kink=double(coeffvalues(fit_after_kink));
%     
%     
%     figure(111)
%     hold on
%     
% %     scatter(limited_region_after_kink_y,limited_region_after_kink_x,'b')
% %     plot(fit_after_kink,'r')
% 
%     % plots fit for data after kink
%     y_min_fit_kink=limited_region_after_kink_y(1);
%     y_max_fit_kink=limited_region_after_kink_y(length(limited_region_after_kink_y));
%     kk=0;
%     for ll=y_min_fit_kink:0.05:y_max_fit_kink
%         kk=kk+1;
%         x_fit_kink(kk)=ll;
% %        y_fit(kk)=paramm(1)*jj^6 + paramm(2)*jj^5 + paramm(3)*jj^4 + paramm(4)*jj^3 + paramm(5)*jj^2 + paramm(6)*jj + paramm(7); % order 6
% %         y_fit(kk)=paramm(1)*jj^5 + paramm(2)*jj^4 + paramm(3)*jj^3 + paramm(4)*jj^2 + paramm(5)*jj + paramm(6); % order 5
%         y_fit_kink(kk)=paramm_after_kink(1)*ll + paramm_after_kink(2); % order 4
% %        y_fit(kk)=paramm(1)*jj^3 + paramm(2)*jj^2 + paramm(3)*jj + paramm(4); % order 3
%     
%     end
% % plot(fitobject,'r')
% %     plot(x_fit_kink,y_fit_kink,'k')
% 
%     % gets angle of fitted line
%     ThetaInRads_after_kink = atan(paramm_after_kink(1));
%     ThetaInDegrees_after_kink = atan(paramm_after_kink(1))*180/pi;
%     
%     % rotates data to align along Y axis
%     x_prime_fit_adjusted = x_fit_kink.*cos(ThetaInRads_after_kink) - y_fit_kink.*sin(ThetaInRads_after_kink); % fit
%     y_prime_fit_adjusted = -x_fit_kink.*sin(ThetaInRads_after_kink) + y_fit_kink.*cos(ThetaInRads_after_kink); % fit
%     
%     x_prime_adjusted = x_prime.*cos(ThetaInRads_after_kink) - y_prime.*sin(ThetaInRads_after_kink); % data
%     y_prime_adjusted = -x_prime.*sin(ThetaInRads_after_kink) + y_prime.*cos(ThetaInRads_after_kink); % data
%     
%     x_prime_adjusted = x_prime_adjusted - mean(y_prime_fit_adjusted); % data
%     limited_region_after_kink_x = limited_region_after_kink_x - mean(y_prime_fit_adjusted); % data after kink
%     y_prime_fit_adjusted = y_prime_fit_adjusted - mean(y_prime_fit_adjusted); % fit
%     
%  
%     
%     
%     % number of points BEFORE kink
%     number_points_before=0;
%     for jj=1:length(x_prime)
%         if y_prime(jj) <= round(before_kink_position_y)
%             number_points_before=number_points_before+1;
%         end
%     end
%     
%     
%     % loads data BEFORE kink in new variable
%     limited_region_before_kink_x=zeros(number_points_before,1);
%     limited_region_before_kink_y=zeros(number_points_before,1);
%     
%     number_points_before=0;
%     for jj=1:length(x_prime_adjusted)
%         if y_prime_adjusted(jj) <= round(before_kink_position_y)
%             number_points_before=number_points_before+1;
%             limited_region_before_kink_x(number_points_before,1)=x_prime_adjusted(jj);
%             limited_region_before_kink_y(number_points_before,1)=y_prime_adjusted(jj);
%         end
%     end
%     
%     limited_region_before_kink_x(limited_region_before_kink_x==0)=[]; % removes last zero
%     limited_region_before_kink_y(limited_region_before_kink_y==0)=[]; % removes last zero
%     
%     
%     
%     max_y_prime_adjusted=0;
%     if mean(y_prime_adjusted)>0
%         max_y_prime_adjusted=max(y_prime_adjusted);
%         y_prime_adjusted=y_prime_adjusted-max_y_prime_adjusted;
%         limited_region_after_kink_x=limited_region_after_kink_x-max_y_prime_adjusted;;
%         limited_region_before_kink_y=limited_region_before_kink_y-max_y_prime_adjusted;
%         
%     end
%     
%     
%     
%          
%     
%     
% %     scatter(limited_region_after_kink_x,limited_region_after_kink_y,'r')
% %     scatter(y_prime_adjusted,x_prime_adjusted,'g')
%     
%     
%     
%     
% %     scatter(limited_region_after_kink_y,limited_region_after_kink_x,'r')
%     
%     
%     
%     % fitobject = fit(x_coord,y_coord,'poly6');
%     % fitobject = fit(x_coord,y_coord,'poly5')
% %     scatter(y_prime_adjusted,x_prime_adjusted,'b') % data
%     fit_final = fit(y_prime_adjusted,x_prime_adjusted,'poly2');
% %     fit_final = fit(y_prime_adjusted,x_prime_adjusted,'poly4');
%     % fitobject = fit(x_prime,y_prime,'poly3')
%     paramm=double(coeffvalues(fit_final));
% %     plot(fit_final,'r') % data
% 
%     y_min_fit=min(y_prime);
%     y_max_fit=max(y_prime);
%     kk=0;
% 
%     for jj=y_min_fit:0.05:y_max_fit
%         kk=kk+1;
%         x_fit(kk)=jj;
% %     y_fit(kk)=paramm(1)*jj^6 + paramm(2)*jj^5 + paramm(3)*jj^4 + paramm(4)*jj^3 + paramm(5)*jj^2 + paramm(6)*jj + paramm(7); % order 6
% %     y_fit(kk)=paramm(1)*jj^5 + paramm(2)*jj^4 + paramm(3)*jj^3 + paramm(4)*jj^2 + paramm(5)*jj + paramm(6); % order 5
% %         y_fit(kk)=paramm(1)*jj^4 + paramm(2)*jj^3 + paramm(3)*jj^2 + paramm(4)*jj + paramm(5); % order 4
% %     y_fit(kk)=paramm(1)*jj^3 + paramm(2)*jj^2 + paramm(3)*jj + paramm(4); % order 3
%     y_fit(kk)=paramm(1)*jj^2 + paramm(2)*jj + paramm(3); % order 2
%     end
%     
% %     plot(x_fit,y_fit,'r') % data
%     
%     
% 	% plots fit after kink and all data
%     scatter(x_prime_fit_adjusted,y_prime_fit_adjusted,'b') % fit
% %     scatter(y_prime_adjusted,x_prime_adjusted,'b') % data
%     
%     
%     
% %     scatter(limited_region_before_kink_y,limited_region_before_kink_x,'b') % data
% 
% 
% 
%     
%     fit_final_before_kink = fit(limited_region_before_kink_y,limited_region_before_kink_x,'poly1');
%     paramm_before=double(coeffvalues(fit_final_before_kink));
%     
%     
% %     length(limited_region_before_kink_y)
% %     length(y_prime_adjusted)
%     
%     
%     kk=0;
%     for jj=min(limited_region_before_kink_y):0.05:limited_region_before_kink_y(number_points_before)
%         kk=kk+1;
%         x_fit_before(kk)=jj;
% %     y_fit(kk)=paramm(1)*jj^6 + paramm(2)*jj^5 + paramm(3)*jj^4 + paramm(4)*jj^3 + paramm(5)*jj^2 + paramm(6)*jj + paramm(7); % order 6
% %     y_fit(kk)=paramm(1)*jj^5 + paramm(2)*jj^4 + paramm(3)*jj^3 + paramm(4)*jj^2 + paramm(5)*jj + paramm(6); % order 5
% %         y_fit(kk)=paramm(1)*jj^4 + paramm(2)*jj^3 + paramm(3)*jj^2 + paramm(4)*jj + paramm(5); % order 4
% %     y_fit(kk)=paramm(1)*jj^3 + paramm(2)*jj^2 + paramm(3)*jj + paramm(4); % order 3
%     y_fit_before(kk)=paramm_before(1)*jj + paramm_before(2); % order 2
%     end
%     
% %     plot(fit_final_before_kink,'r') % data    
%     plot(x_fit_before,y_fit_before,'r') % data   
% %     plot(fit_final_before_kink,'g') % data   
% 
%     
%     
%     
%     
%     


nFrames
ii


    

end




figure(9999)
    
    plot(time_frame_ii,position_kink_y,'r','MarkerSize',10)





save(savefilename, '-struct', 'saved_data')

