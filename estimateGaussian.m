%Estimate the Gaussian using X
function [mu sigma2] = estimateGaussian(X)


	% Useful variables
	[m, n] = size(X);

	% You should return these values correctly
	mu = zeros(n, 1);
	sigma2 = zeros(n, 1);

	fprintf('X  %f  %f\n', rows(X), columns(X)); %4 x 3


	% ====================== YOUR CODE HERE ======================
	% Instructions: Compute the mean of the data and the variances
	%               In particular, mu(i) should contain the mean of
	%               the data for the i-th feature and sigma2(i)
	%               should contain variance of the i-th feature.
	%



	%compute the MU for each feature
	j = 1;
	for i=1:n
		mu(i) = sum(X(:,i))/m;
	end
	
	%compute Sigma
	j = 1;
	
	for i=1:n
		tmpF1 = zeros(m,1);
		tmpF1 = X(:,i); % create column vector for Feature 1;
		
		s = 0;
		for j=1:m
			s = s + ((tmpF1(j) - mu(i)) ^ 2);
		end
		sigma2(i) = s/m;
	end

end
