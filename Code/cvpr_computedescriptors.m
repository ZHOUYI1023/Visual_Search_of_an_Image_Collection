%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
% cvpr_computedescriptors.m
% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
% Centre for Vision Speech and Signal Processing (CVSSP)
% University of Surrey, United Kingdom
close all;
clear all;
%% PCA flag
FLAG_PCA = true;
%% Path of MSRCv2 dataset\
DATASET_FOLDER = 'D:\Github\ClassNotes\COMPUTER VISION AND PATTERN RECOGNITION (EEE3032)\Assignment\cwork_basecode_2012\msrc_objcategimagedatabase_v2\MSRC_ObjCategImageDatabase_v2';
%% Create a folder to hold the results...
OUT_FOLDER = 'D:\Github\ClassNotes\COMPUTER VISION AND PATTERN RECOGNITION (EEE3032)\Assignment\cwork_basecode_2012/descriptors';
%% the folder specified as 'OUT_FOLDER'.
% options: globalRGBhisto, spatialGridEdgeOnly, spatialGridMeanColor,
% spatialGridColorHistogram
% spatialGridEdge+MeanColor
% spatialGridGaborWavelt+MeanColor
OUT_SUBFOLDER='spatialGridGaborWavelt+MeanColor';
%% calculate descriptors
Method = OUT_SUBFOLDER;
F_PCA = [];
allfiles=dir (fullfile([DATASET_FOLDER,'/Images/*.bmp']));
for filenum=1:length(allfiles)
    fname=allfiles(filenum).name;
    fprintf('Processing file %d/%d - %s\n',filenum,length(allfiles),fname);
    tic;
    imgfname_full=([DATASET_FOLDER,'/Images/',fname]);
    img=double(imread(imgfname_full));
    fout=[OUT_FOLDER,'/',OUT_SUBFOLDER,'/',fname(1:end-4),'.mat'];%replace .bmp with .mat
    F = extractDescripter(img, Method);
    toc
    save(fout,'F');
    F_PCA = [F_PCA,F'];
end
% Mahalanobis
feature = F_PCA - mean(F_PCA,2);
covariance = feature * feature';
save([OUT_FOLDER,'/',OUT_SUBFOLDER,'/covariance.mat'], 'covariance');
%% PCA
if FLAG_PCA == true
    FReduced = pca(F_PCA);
    F_PCA = [];
    OUT_SUBFOLDER = [OUT_SUBFOLDER, 'PCA'];
    for filenum = 1:length(allfiles)
        fname=allfiles(filenum).name;
        fout=[OUT_FOLDER,'/',OUT_SUBFOLDER,'/',fname(1:end-4),'.mat'];
        F = FReduced(:,filenum)';
        F_PCA = [F_PCA,F'];
        save(fout,'F');
    end
end
% Mahalanobis
feature = F_PCA - mean(F_PCA,2);
covariance = feature * feature';
save([OUT_FOLDER,'/',OUT_SUBFOLDER,'/covariance.mat'], 'covariance');