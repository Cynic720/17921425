clc
clear

ptCloud = pcread('data.pcd');


figure;
pcshow(ptCloud);
ptCloudFiltered = ptCloud;

figure;
pcshow(ptCloudFiltered);

pts = ptCloudFiltered.Location;
[coeff, score, latent] = pca(pts);

axisLine = coeff(:,1); 

hold on;
translation = [-14, 0, 0];
meanPoint = mean(pts, 1)+translation; 
quiver3(meanPoint(1), meanPoint(2), meanPoint(3), axisLine(1), axisLine(2), axisLine(3), 32, 'r', 'LineWidth', 2);
hold off;

disp(axisLine);


