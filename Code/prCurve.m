function [precision,recall]= prCurve(class,dst,ind)
truth_class = class(dst(1));
p = 0;
n = 0;
t = length(find(class == truth_class));
precision = zeros(length(dst),1);
recall = zeros(length(dst),1);
for i = 1:length(dst)
    if class(dst(i)) == truth_class
        p = p + 1;
    else
        n = n + 1;
    end
    precision(i) = p/i; 
    recall(i) = p/t;
end
t = string(['image ', num2str(ind)]);
figure(15+ind)
plot(recall, precision, 'r-', 'LineWidth', 2, 'MarkerSize', 12)
xlabel('Recall')
ylabel('Precision')
title(t)
axis([0,1,0,1])
ax = gca;
ax.FontSize = 16;
end
    