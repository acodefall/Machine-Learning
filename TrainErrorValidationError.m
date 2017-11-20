%compute the TrainError and Validation error.
%Modify the LAMBDA 
%Generate cost error for Validation set and plot it againt THETA
%Output is train  and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) 
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);
cvLen = rows(Xval);
m = rows(X);


	   for i = 1:length(lambda_vec)
		
		%Train the system using x TrainingSet.  This gives the THETA
			thetaT = trainLinearReg(X, y, lambda_vec(i));
		
		%compute the J(trainError) using TrainingSet (that is "1 to i")
			%1
			predValue = X*thetaT; %compute predictedValue
			
			%2
			errors = predValue - y; %compute J() cost error
			
			%3
			%Compute the total error by adding the Error for every entry in Training set
			%Square the error 
			SqErrors = errors .^2;
			totalError = sum(SqErrors);
			
			%4
			%Average the totalError
			error_train(i) = totalError/(2*m);
		
		
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
%



end
