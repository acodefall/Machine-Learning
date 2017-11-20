function findGD(X, y, theta, alpha)
	%X has training set
	%y is the actual output for training set
	%alpha Learning rate
	%find the gradient and 


	m = length(y); % number of training examples
	costJ_history = zeros(num_iters, 1);

	fprintf('Alpha %f \n', alpha);
	fprintf('X len %f \n', length(X));
	fprintf('Y len %f \n', length(y));
	fprintf('Theta len %f \n', length(theta));
	for iter = 1:1000
		%predicated value for Traingset
		PredictedValue = X*theta;
		
		%costError for every row J()
		Errors = (PredictedValue - y);
		
		%AvgError
		ErrorsAvg = (X' * Errors)/m;
		theta = theta - ErrorsAvg.*alpha;
		
		% Save the cost J in every iteration    
		costJ_history(iter) = findCost(X, y, theta);

	end
	fprintf('Done\n');
end
