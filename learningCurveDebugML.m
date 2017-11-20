function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);
cvLen = rows(Xval);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------


	%to plot learning curve repeat these steps using different subset of data
	%	-Train the system using x TrainingSet.  This gives the THETA
	%   -Compute JtrainError, using THETA using TrainingSet data
	%   -Compute JcvError, using THETA using crossValidSet data
	for i = 1:m
		
		%Train the system using x TrainingSet.  This gives the THETA
			thetaT = trainLinearReg(X(1:i,:), y(1:i,:), lambda);
		
		%compute the J(trainError) using TrainingSet (that is "1 to i")
			%1
			predValue = X(1:i,:)*thetaT; %compute predictedValue
			
			%2
			errors = predValue - y(1:i,:); %compute J() cost error
			
			%3
			%Compute the total error by adding the Error for every entry in Training set
			%Square the error 
			SqErrors = errors .^2;
			totalError = sum(SqErrors);
			
			%4
			%Average the totalError
			error_train(i) = totalError/(2*i);
		
		
		%compute the J(cvError) using crossValidSet Xval, yval
			%1
			predValue = Xval*thetaT; %compute predictedValue
			
			%2
			errors = predValue - yval; %compute J() cost error
			
			%3
			%Compute the total error by adding the Error for every entry in Training set
			%Square the error 
			SqErrors = errors .^2;
			totalError = sum(SqErrors);
			
			%4
			%Average the totalError
			error_val(i) = totalError/(2*cvLen);
				
		% Compute train/cross validation errors using training examples 
           % X(1:i, :) and y(1:i), storing the result in 
           % error_train(i) and error_val(i)
           ....
           
    end




% -------------------------------------------------------------

% =========================================================================

end
