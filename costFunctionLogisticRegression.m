%compute the cost for Logistic Regression
function [costJ, gradient] = costFunctionLogisticRegression(theta, X, y)
	%X training-set data has just one FEATURE
	%y training-set data has OUTPUT Y
	%theta Theta(min) 
	m = length(y); % number of training examples

	%fprintf('X rows: %f  cols: %f \n', rows(X), columns(X));
	%fprintf('y rows: %f  cols: %f \n', rows(y), columns(y));
	%fprintf('theta rows: %f  cols: %f \n', rows(theta), columns(theta));
	% You need to return the following variables correctly 
	costJ = 0;
	grad = zeros(size(theta));



	h = sigmoidFunc(X * theta);
	%fprintf('h() rows: %f  cols: %f \n', rows(h), columns(h));
    
	%print the predicted value and actual value
	%fprintf(' print the predicted value and actual value \n');
	%for i = 1: m 
	%	fprintf(' : %f  cols: %f \n', h(i,1), y(i,1));
	%end
	
	%pause;
	
	errors0 = (-y)'*log(h);
	%fprintf('errors0 rows: %f  cols: %f \n', rows(errors0), columns(errors0));
	
	errors1 = (1 - y)'*log(1 - h);
	%fprintf('errors1 rows: %f  cols: %f \n', rows(errors1), columns(errors1));
	costJ = (errors0 - errors1)/m;

	%fprintf('j rows: %f  cols: %f \n', rows(j), columns(j));

	%compute gradiant
	gradient = (X' * (h-y))/m;



end
