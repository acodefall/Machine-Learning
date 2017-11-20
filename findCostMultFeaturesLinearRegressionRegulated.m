%find error and gradient for Regulated Linear Regression(there are multiple FEATURES)
function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
	m = length(y); % number of training examples

	
	J = 0;
	grad = zeros(size(theta));

	%fprintf('X  %f  %f\n', rows(X), columns(X));
	%fprintf('theta  %f  %f\n', rows(theta), columns(theta));
	%fprintf('y  %f  %f\n', rows(y), columns(y));
	%fprintf('lambda  %f \n', lambda);


	%1
	predValue = X*theta; %compute predictedValue
	
	%2
	errors = predValue - y; %compute J() cost error
	
	%3
	%Compute the total error by adding the Error for every entry in Training set
	%Square the error 
	SqErrors = errors .^2;
	totalError = sum(SqErrors);
	
	%4
	%Average the totalError
	avgError = totalError/(2*m);
	
	%5	
	theta(1) = 0;
	RegFactor = (lambda/(2*m)) * sum(theta .^2); %compute Reg Factor

	
	J=avgError + RegFactor;
	
	%fprintf('J  %f  avgError %f\n', J, avgError);
	
	%compute gradient
	x2 = (X' * errors)/m;
	
	%compute regularization for Grdainet
	Reg2 = (lambda/m) .* theta;
	
	grad = x2 + Reg2;



	grad = grad(:);

end
