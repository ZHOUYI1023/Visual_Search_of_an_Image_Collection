function F = extractDescripter(img, varargin)
%EXTRACTDESCRIPTER Summary of this function goes here
%   Detailed explanation goes here
%img = varargin{1};
if nargin == 1
    Method = 'globalRGBhisto';
elseif nargin == 2
    Method = char(varargin);
    Detector = 'sift';
elseif nargin == 3
    Method = char(varargin{2});
    Detector = char(varargin{1});
end

if exist('Detector') == 1
    points = extractDetector(img, Detector);
if string(Method(1:11)) == "spatialGrid"
    Method2 = Method(12:end);
    Method = Method(1:11);
end
switch Method
    case 'globalRGBhisto' 
        q = 8;
        F = globalColorHistogram(img, q);
    case 'spatialGrid'
        q = 2;
        gridSize = [5, 5];
        F = spatialGrid(img, gridSize, q, Method2);
    case 'sift'
        [F, ~] =extractFeatures(img,points,'Method','SURF');
    case 'surf'
        [F, ~] =extractFeatures(img,points,'Method','SURF');
    case 'hog'
        [F, ~] = extractHOGFeatures(I,points);
    case 'orb'
        [F, ~] =extractFeatures(I,points,'ORB');
    otherwise
        error("Unsupportted Descripter")
end

end