%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%
% cvpr_visualsearch.m
% Skeleton code provided as part of the coursework assessment
%
% This code will load in all descriptors pre-computed (by the
% function cvpr_computedescriptors) from the images in the MSRCv2 dataset.
%
% It will pick a descriptor at random and compare all other descriptors to
% it - by calling cvpr_compare.  In doing so it will rank the images by
% similarity to the randomly picked descriptor.  Note that initially the
% function cvpr_compare returns a random number - you need to code it
% so that it returns the Euclidean distance or some other distance metric
% between the two descriptors it is passed.
%
% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
% Centre for Vision Speech and Signal Processing (CVSSP)
% University of Surrey, United Kingdom
%%
clc;
close all;
clear all;

%%
% Edit the following line to the folder you unzipped the MSRCv2 dataset to
DATASET_FOLDER = 'D:\Github\ClassNotes\COMPUTER VISION AND PATTERN RECOGNITION (EEE3032)\Assignment\cwork_basecode_2012\msrc_objcategimagedatabase_v2\MSRC_ObjCategImageDatabase_v2';

% Folder that holds the results...
DESCRIPTOR_FOLDER = 'D:\Github\ClassNotes\COMPUTER VISION AND PATTERN RECOGNITION (EEE3032)\Assignment\cwork_basecode_2012\descriptors';

% and within that folder, another folder to hold the descriptors
% options: globalRGBhisto, spatialGridEdgeOnly, spatialGridMeanColor,
% spatialGridColorHistogram
% spatialGridEdge+MeanColor
% spatialGridGaborWavelt+MeanColor
DESCRIPTOR_SUBFOLDER='globalRGBhisto';


%% 1) Load all the descriptors into "ALLFEAT"
% each row of ALLFEAT is a descriptor (is an image)

ALLFEAT=[];
ALLFILES=cell(1,0);
ctr=1;
allfiles=dir (fullfile([DATASET_FOLDER,'/Images/*.bmp']));
class = zeros(length(allfiles),1);
for filenum=1:length(allfiles)
    fname=allfiles(filenum).name;
    if fname(2) == '_'
        class(filenum,1) = str2num(fname(1));
    else
        class(filenum,1) = str2num(fname(1:2));
    end
    imgfname_full=([DATASET_FOLDER,'/Images/',fname]);
    img=double(imread(imgfname_full));
    thesefeat=[];
    featfile=[DESCRIPTOR_FOLDER,'/',DESCRIPTOR_SUBFOLDER,'/',fname(1:end-4),'.mat'];%replace .bmp with .mat
    load(featfile, 'F');
    ALLFILES{ctr}=imgfname_full;
    ALLFEAT=[ALLFEAT ; F];
    ctr=ctr+1;
end

%% 2) Pick an image to be the query

QUERY_CLASS = 2; % specify a class
NIMG=size(ALLFEAT,1);           % number of images in collection
%queryimg=floor(rand()*NIMG);    % index of a random image
% from class extract all imgs
queryimg = find(class == QUERY_CLASS);

%% 3) Compute the distance of image to the query & Visualization
method = 'l1';
SHOW = 15;
load([DESCRIPTOR_FOLDER,'/',DESCRIPTOR_SUBFOLDER,'/covariance.mat'])
dst= zeros(NIMG,2);
for j = 1:size(queryimg)
    for i=1:NIMG
        candidate=ALLFEAT(i,:);
        query=ALLFEAT(queryimg(j),:);
        thedst=compare(query,candidate,method,covariance);
        dst(i,:)= [thedst i];
    end
    dst = sortrows(dst,1);% sort the results
    [precision,recall] = prCurve(class,dst(:,2),j);%pr-curve
    %  Visualise the results
    for i = 1:15
        figure(j)
        hold on
        subplot(3,5,i)
        img=imread(ALLFILES{dst(i,2)});
        imshow(img)
    end
end