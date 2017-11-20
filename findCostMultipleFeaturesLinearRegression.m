function costJ = findCostMultipleFeatures(X, y, theta)
	%X training-set data has just one FEATURE
	%y training-set data has OUTPUT Y
	%theta Theta(min)
	
	m = length(y); % number of training examples
	Xc = columns(X);
	t = length(theta); % number of training examples
	fprintf('X len %f \n', length(X));
	fprintf('Y len %f \n', length(y));
	fprintf('Theta len %f \n', length(theta));


	costJ = 0;



   
   %Extend the X by one column. Extra column should have value 1
   Xnew = ones(t,m); %allocate a larger matrix to hold X.
   for i = 1: Xc     
		for j = 1: m
			Xnew(i+1, j) = X(i,j); %Transfer the content from X to Xnew, leave the 1st column empty
		end
	end
		
   %compute Hypothesis using MATRIX muplication (theta' * X)
   errors = (theta' * X) - y;

   %square the errors
	SqErrors = errors .^2;
	totalError = sum(SqErrors);
	
	%Average the totalError
	avgError = totalError/(2*m);
		
	costJ=avgError;



end
