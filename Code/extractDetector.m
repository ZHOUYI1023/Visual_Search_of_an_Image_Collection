function points = extractDetector(I, Method)
%DETECTOR Summary of this function goes here
%   Detailed explanation goes here
if nargin < 2
    Method = "harris";
end
I = rgb2gray(I);
switch Method
    case "harris"
        points = detectHarrisFeatures(I);
    case "orb"
        points = detectORBFeatures(I);
    case "fast"
        points = detectFASTFeatures(I);       
    case "surf"
        points = detectSURFFeatures(I); 
    case "sift"
        points = detectSURFFeatures(I,'MetricThreshold',1); 
end

