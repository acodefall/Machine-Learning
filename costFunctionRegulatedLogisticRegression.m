%compute the cost for Logostic Regression WITH Regulation 
function [costJ, gradient] = costFunctionReg(theta, X, y, lambda)
	%X training-set data has just one FEATURE
	%theta Theta(min)
	

	% Initialize some useful values
	m = length(y); % number of training examples

	% You need to return the following variables correctly 
	costJ = 0;
	gradient = zeros(size(theta));



	%Use the Hypothesis to compute Predicted value
	%We will get a Column vctor  of Predicted value
	PV = sigmoidFunc(X*theta);
	
	%Apply the equation cost function to know the cost of using PV
	errors0 = (-y)'*log(PV);
	errors1 = (1 - y)'*log(1 - PV);



	%Compute the regularization. regularization does not apply to Theta0,
	% make theta0 as ZERO.
	theta(1) = 0;
	RegFactor = (lambda/(2*m)) * sum(theta .^2);


	%Compute the Cost function taking in to account Regularization factor
	costJ = ((errors0 - errors1)/m) + RegFactor;
	
	RegFactor2 = (lambda/m) .* theta;
	
	%compute gradiant taking in to account Regularization factor
	gradient = ((X' * (PV-y))/m) + RegFactor2;


end
