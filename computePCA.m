%compute PCA
function [U, S] = computepca(X)
	%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
	%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
	%

	% Useful values
	[m, n] = size(X);

	% You need to return the following variables correctly.
	U = zeros(n);
	S = zeros(n);

	% Instructions: You should first compute the covariance matrix. Then, you
	%               should use the "svd" function to compute the eigenvectors
	%               and eigenvalues of the covariance matrix. 
	%

	
	sigma = X' * X;
	sigma = sigma ./m;
	
	[U, S, V] = svd(sigma);


end
