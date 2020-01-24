function F = globalColorHistogram(img, q)
%GLOBALCOLORHISTOGRAM Summary of this function goes here
%   Detailed explanation goes here
[R,G,B] = imsplit(img);
R = reshape(R, [], 1);
G = reshape(G, [], 1);
B = reshape(B, [], 1);
R1 = floor(R*q/256);
G1 =  floor(G*q/256);
B1 =  floor(B*q/256);
i = R1 .* (q^2) + G1 .* q + B1; 
binEdge = (0: q^3);
F = histcounts(i, binEdge);
% figure(6)
% hold on
% subplot(1,4,3)
% imshow(uint8(img))
% subplot(1,4,4)
% a = histogram(i,'BinEdges',binEdge,'Normalization','pdf');
% xlabel('Bins')
% ylabel('Normalized Counts')
% ax = gca;
% ax.FontSize = 16;
end

