%compute the centrods
function centroids = findCentroidss(X, idx, K)
%find the centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);



% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%

	for i=1:K
		sel = find(idx == i); %find returns a column-vector, with 0s and 1s. 1 will be present wherever the "idx[] = i". Ex: "idx[6] == i", then sel[6] = 1
		centroids(i,:) = mean(X(sel,:));
	end





end

