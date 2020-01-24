function featureReduced = pca(feature)
%PCA Summary of this function goes here
%   Detailed explanation goes here
feature = feature - mean(feature,2);
covariance = feature * feature';
[V,D] = eig(covariance);
[d,ind] = sort(diag(D),'descend');
sumD = cumsum(d);
sumD = sumD/sumD(end);
indD = find(sumD < 0.97);
eigenVector = V(:,ind(1:indD(end)));
featureReduced = eigenVector'*feature;
end

