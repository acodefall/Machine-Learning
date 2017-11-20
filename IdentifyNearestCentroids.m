function idx = IdentifyNearestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);
%fprintf('X  %f  %f\n', rows(X), columns(X)); %300 x 2
%fprintf('centroids  %f  %f\n', rows(centroids), columns(centroids)); %3 x 2

% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%

%
	
	m = rows(X);
	
	
	for i=1:m
		dist = zeros(K,1);
		%fprintf('sample  %f  \n', i); %300 x 2
		for j=1:K
			
			%fprintf('1      %f  %f \n', X(i,1), X(i,2));
			%fprintf('2      %f  %f \n', centroids(j,1), centroids(j,2));
			%x1 = X(i,1) - centroids (j,1);
			%y1 = X(i,2) - centroids (j,2);
			
			%dist(j,1) = (x1^2 + y1^2);
			%fprintf('3      %f  %f = %f\n', x1^2, y1^2, (x1^2 + y1^2));
			
			tmp = zeros(1,columns(centroids));
			%fprintf('4     \n');
			%X(i)
			
			%fprintf('5     \n');
			%centroids(j)
			
			%we have to subtract the row-wise so : is required
			tmp = (X(i,:) - centroids(j,:));
			%fprintf('7     \n');
			%tmp
			
			tmp = tmp .^2;
			%fprintf('10     \n');
			%tmp
			
			
			dist(j,1) = sum(tmp);
			%fprintf('61      %f \n', dist(j,1));
		
		end
		%dist
		%fprintf('%f %f %f %f % f\n', dist(1,1), dist(2,1), dist(3,1), dist(4,1), dist(5,1)); %300 x 2
		
		[f,k] = min(dist);
	
		%fprintf('min %f %f\n', k, f); %300 x 2
		idx(i) = k;
	end
	
	

	%fprintf('done \n'); %300 x 2

% =============================================================

end

