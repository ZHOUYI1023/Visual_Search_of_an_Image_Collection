function dst=compare(F1, F2, varargin)
if nargin == 3
    Method = char(varargin);
elseif nargin == 4
    Method = char(varargin{1});
    covariance = varargin{2};
end

% This function should compare F1 to F2 - i.e. compute the distance
% between the two descriptors

% For now it just returns a random number
switch Method 
    case 'l2'
        dst = sqrt(sum((F1 - F2).^2));
    case 'l1'
        dst = sum(abs(F1 - F2));
    case 'cosine'
        dst = abs(sum(F1.*F2))/sqrt(sum(F1.^2)*sum(F2.^2));
    case 'pearson'
        F1 = F1 - mean(F1);
        F2 = F2 - mean(F2);
        dst = sum(F1.*F2)/sqrt(sum(F1.^2)*sum(F2.^2));
    case 'mahalanobis'
        dst = sqrt((F1-F2)*pinv(covariance)*(F1-F2)');
    otherwise
        error('Unsupported Distance Metric')
end
return;
