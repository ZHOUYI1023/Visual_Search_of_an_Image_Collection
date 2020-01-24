% N = 128;
% n = -(N-1)/2:(N-1)/2;
% alpha = 8;
% alpha2 = 16;
% w = gausswin(N,alpha);
% w2 = gausswin(N,alpha2);
% % stdev = (N-1)/(2*alpha);
% % y = exp(-1/2*(n/stdev).^2);
% figure(1)
% plot(n,w-w2)
% hold on
% plot(n,y,'.')
% hold off

% xlabel('Samples')
% title('Gaussian Window, N = 64')













% 
% img = imread('9_30_s.bmp');
% img = imread('19_8_s.bmp');
% figure
% subplot(1,2,1)
% imshow(rgb2gray(img))
% hold on
% F = detectSURFFeatures(rgb2gray(img), 'MetricThreshold', 800);
% plot(F.Location(1:30,1), F.Location(1:30,2),'rx','LineWidth', 2, 'MarkerSize', 12)
% subplot(1,2,2)
% img1 = impyramid(img, 'reduce');
% imshow(rgb2gray(img1))
% hold on
% F = detectSURFFeatures(rgb2gray(img1),'MetricThreshold', 800);
% plot(F.Location(1:30,1), F.Location(1:30,2),'rx','LineWidth', 2, 'MarkerSize', 12)

% grayImg = rgb2gray(img);
% 
% [nRows, nCols] = size(grayImg);
% STEP = 30;
% colInd = (STEP/2 : STEP : nCols)';
% rowInd = (STEP/2 : STEP : nRows)';
% [A, B] = meshgrid(colInd, rowInd);
% densePoints = [A(:) B(:)];
% 
% [featuresDense, validPointsDense] = extractFeatures(grayImg, densePoints, 'Method', 'SURF');
% figure, imshow(grayImg)
% hold on
% plot(validPointsDense,'showOrientation',true);



img2 = rgb2gray(imread('9_30_s.bmp'));
% img3 = imread('19_8_s.bmp');
% img4 = imread('9_6_s.bmp');
% method = 'l2';
% F2 = globalColorHistogram(double(img2), 8);
% F3 = globalColorHistogram(double(img3), 8);
% F4 = globalColorHistogram(double(img4), 8);
% dst1=compare(F2, F3, method)
% dst2=compare(F2, F4, method)

% figure
% subplot(1,2,1)
% BW1 = edge(rgb2gray(img2),'Sobel');
% imshow(BW1)
% title("Sobel Operator")
% ax = gca;
% ax.FontSize = 16;
% subplot(1,2,2)
% BW1 = edge(rgb2gray(img2),'Canny', [0.05 0.4]);
% imshow(BW1)
% title('Canny Edge Detector')
% ax = gca;
% ax.FontSize = 16;
% F2 = spatialGrid(double(img2) , [5, 5], 8, 'GaborWavelt+MeanColor');
% F3 = spatialGrid(double(img3) , [5, 5], 8, 'GaborWavelt+MeanColor');
% F4 = spatialGrid(double(img4) , [5, 5], 8, 'GaborWavelt+MeanColor');
% 
% dst1=compare(F2, F3, method,covariance)
% dst2=compare(F2, F4,method ,covariance)
[featureVector,hogVisualization] = extractHOGFeatures(img2);
figure;
imshow(img2); 
hold on;
plot(hogVisualization);
