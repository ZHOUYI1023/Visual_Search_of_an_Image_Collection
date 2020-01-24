close all;
clear all;
%% Edit the following line to the folder you unzipped the MSRCv2 dataset to
DATASET_FOLDER = 'D:\Github\ClassNotes\COMPUTER VISION AND PATTERN RECOGNITION (EEE3032)\Assignment\cwork_basecode_2012\msrc_objcategimagedatabase_v2\MSRC_ObjCategImageDatabase_v2';

%% Create a folder to hold the results...
OUT_FOLDER = 'D:\Github\ClassNotes\COMPUTER VISION AND PATTERN RECOGNITION (EEE3032)\Assignment\cwork_basecode_2012/descriptors';
%% and within that folder, create another folder to hold these descriptors
%% the idea is all your descriptors are in individual folders - within
%% the folder specified as 'OUT_FOLDER'.
OUT_SUBFOLDER= 'BOG';
allfiles=dir (fullfile([DATASET_FOLDER,'/Images/*.bmp']));
featuresAll = [];
index = zeros(length(allfiles)+1,1); 
%% Built-in Method
BIN = false;
if BIN == true
    try
        load("bag.mat");
    catch
        warning("Recalculating BoW");
        imds = imageDatastore(fullfile([DATASET_FOLDER,'/Images/*.bmp']));
        bag = bagOfFeatures(imds, 'PointSelection', 'Detector');
    end

    for filenum = 1:length(allfiles)
        fname=allfiles(filenum).name;
        imgfname_full=([DATASET_FOLDER,'/Images/',fname]);
        img=double(imread(imgfname_full))./255;
        img = rgb2gray(img);
        F = encode(bag,img);
        fout=[OUT_FOLDER,'/',OUT_SUBFOLDER,'_BUILT_IN/',fname(1:end-4),'.mat'];
        save bag bag
        save(fout,'F');
    end
end
%% My Implementation
Method = ["harris", "sift"];
for filenum=1:length(allfiles)
    fname=allfiles(filenum).name;
    fprintf('Processing file %d/%d - %s\n',filenum,length(allfiles),fname);
    tic;
    imgfname_full=([DATASET_FOLDER,'/Images/',fname]);
    img=double(imread(imgfname_full))./255;
    img = rgb2gray(img);
    features = extractDescripter(img, Method(1), Method(2));
    index(filenum+1) = size(features,1); 
    featuresAll = [featuresAll;features];
    toc
end
[C, bestAssignments, bestDists, bestCompactness] = approximateKMeans(featuresAll, 500);
index = cumsum(index);


for i = 1:length(index)-1
    points = featuresAll(index(i)+1:index(i+1),:);
    [~,idx_test] = pdist2(C,points,'euclidean','Smallest',1);
    binEdge = (0:500);
    F = histcounts(idx_test, binEdge);
    fname=allfiles(i).name;
    fout=[OUT_FOLDER,'/',OUT_SUBFOLDER,'/',fname(1:end-4),'.mat'];
    save(fout,'F');
end
