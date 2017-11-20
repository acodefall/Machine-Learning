%compute the cost from Linear Regression
function costJ = findCost(X, y, theta)

	%X training-set data has just one FEATURE
	%y training-set data has OUTPUT Y
	%theta Theta(min)
	
	m = length(y); 

	
	costJ = 0;


	
	%1
	%compute predicatedValue using (Theta' * X)
	%output goes to an array predValue 
	predValue = X*theta; 
	
	%2
	%Compute error in every predicatedValue. That is (predValue - actualValue) for every row of Training set
	%Store these errors in array errors
	errors = predValue - y;
	
	%3 
	%Compute the total error by adding the Error for every entry in Training set
	%Square the error 
	SqErrors = errors .^2;
	totalError = sum(SqErrors);
	
	%4
	%Average the totalError
	avgError = totalError/(2*m);
		
	costJ=avgError;


end
