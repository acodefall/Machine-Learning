%Predict the value in terms of 0 or 1
function predictedValue = predictValueLogisticRegression(theta, X)
	%X training-set data
	%theta Theta(min)
	

	m = size(X, 1); % Number of training examples

	% You need to return the following variables correctly
	predictedValue = zeros(m, 1);

	fprintf('Inside Predict \n');
	fprintf('X rows: %f  cols: %f \n', rows(X), columns(X));
	fprintf('theta rows: %f  cols: %f \n', rows(theta), columns(theta));



	h = sigmoidFunc(X * theta);
	fprintf('h() rows: %f  cols: %f \n', rows(h), columns(h));
    
	
	%print the predicted value and actual value
	%fprintf(' print the predicted value and actual value \n');
	for i = 1: m 
			%actual = X(i,:) * theta;
			if(h(i) >= 0.5) %use 0.5 as Threshold
				predictedValue(i) = 1;
			end
			%fprintf('h() a: %f  p: %f \n', actual, h(i));
	end
end
