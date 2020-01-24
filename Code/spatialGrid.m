function F = spatialGrid(img ,gridSize, q, method)
%SPATIALGRID Summary of this function goes here
%   Detailed explanation goes here
if nargin == 3
    method = 'ColorHistogram';
end
step = floor([size(img,1)/gridSize(1), size(img,2)/gridSize(2)]);
F = [];
 for i = 1:gridSize(1)
     for j = 1:gridSize(2)
         img_temp = img((i-1)*step(1)+1:i*step(1), (j-1)*step(2)+1:j*step(2),:);
         %% Mean Color
         switch method 
             case {'Edge+MeanColor', 'GaborWavelt+MeanColor'}
                [R,G,B] = imsplit(img_temp);
                RMean = mean(mean(R));
                GMean = mean(mean(G));
                BMean = mean(mean(B));
                color = [RMean,GMean,BMean]./255;
         %% Color Histogram
             case {'ColorHistogram', 'Edge+ColorHistogram'}
                color = globalColorHistogram(img_temp, 8);
             case 'Edge'
                 color = [];
             otherwise
                 error("Unindentified Method");
         end
         %% edge
         switch method
             case {'Edge+ColorHistogram', 'Edge+MeanColor', 'Edge'}
                [~,~,Gv,Gh] = edge(rgb2gray(uint8(img_temp)));
                Gv = reshape(Gv, [], 1);
                Gh = reshape(Gh, [], 1);
                theta = atan2(Gh, Gv);
                binEdge = (0:pi/4:2*pi);
                edgeOrient = histcounts(theta, binEdge);
                edgeFeature = edgeOrient./sum(edgeOrient);
             otherwise
                 edgeFeature = [];
         end
         %% Concatenation
         F = [F,color,edgeFeature];
     end
 end
 if string(method) == "GaborWavelt+MeanColor"
     [v, m] = gaborWavelet(rgb2gray(uint8(img_temp))/255,4, 6);
     F = [v, m, F];
figure(1)
subplot(2,3,3)
imshow(uint8(img))
subplot(2,3,6)
a = stem(F,'Marker', 'none');
xlabel('Bins')
ylabel('Normalized Counts')
ax = gca;
ax.FontSize = 16;

end