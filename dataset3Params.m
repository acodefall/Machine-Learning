function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

%following are the values to be used for C and sigma
val = [0.01;0.03;0.1;0.3;1;3;10;30];
% You need to return the following variables correctly.
C = 1; 
sigma = 0.3; 
%fprintf('X  %f  %f\n', rows(X), columns(X)); % 211 x 2
%fprintf('y  %f  %f\n', rows(y), columns(y)); % 211 x 1
%fprintf('Xval  %f  %f\n', rows(Xval), columns(Xval)); %200 x 2
%fprintf('yval  %f  %f\n', rows(yval), columns(yval)); %200 x 1
%fprintf('sigma  %f \n', sigma); %0.300
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
	
	x1 = [1 2 1];  %got this from ex6.m
	x2 = [0 4 -1]; %got this from ex6.m
	len = rows(val);
	
	%these store the result of error, C, Sigma combo
	rC = zeros(64,1); 
	rSig = zeros(64,1);
	rE = zeros(64,1); 
	res = 1;
	for i = 1:len  %loop around C
		for j = 1:len %loop around sigma
			C = val(i);
			sigma = val(j);
			
			%Train the alg for the given "Sigma and C" combo
			%Use Training Data set.
			%Output is a model
			model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
			
			%Use the model for predicting the values for 'Cross Validation Set'
			predValue(:, 1) = svmPredict(model, Xval); 
			
			%compute the error between predicated-value and actual-value
			%store the resulting "error, C, sigma". There will be 64 results
			rE(res) = mean(double(predValue ~= yval));
			rC(res) = C;
			rSig(res) = sigma;
			
			res = res+1;
		end
	end


	%find "error, C, sigma". combo with least error. Check inside rE.
	[f,k] = min(rE); 
	
	%k points "least error", so get the corresponding Sigma and C
	C = rC(k);
	sigma=rSig(k);
% =========================================================================
end
