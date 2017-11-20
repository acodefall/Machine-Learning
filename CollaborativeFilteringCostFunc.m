%implement Collaborative Filtering Cost Function that returns CostError and Gradient
function [costJ, grad] = CollaborativeFilteringCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)


% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
costJ = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

	fprintf('num_users  %f \n',num_users); %3
	fprintf('num_features  %f \n',num_features); %2
	fprintf('Theta  %f  %f\n', rows(Theta), columns(Theta)); %3 x 2
	fprintf('R  %f  %f	\n', rows(R), columns(R)); %4 x 3
	fprintf('X  %f  %f	\n', rows(X), columns(X)); %4 x 2
	fprintf('Y  %f  %f	\n', rows(Y), columns(Y)); %4 x 3
	
	
	%compute the Predicted Value
	predvalue = X * Theta';
	
	%	predvalue
	%	u1M1  u2M1
	%	u1M2  u2M2
    %	u1M3  u2M3	

	%Y
	%	u1M1 	u2M1	u3M1
	%	u1M2	u2M2	u3M2
	%	u1M3	u2M3	u3M3
	%	u1M4	u2M4	u3M4

	%compute the Error
	Error = predvalue - Y; 

	%Inside 'Error' we have to SQUARE only those cell that have 1 in [R]
	
	%R
	%	u1M1 	u2M1	u3M1
	%	u1M2	u2M2	u3M2
	%	u1M3	u2M3	u3M3
	%	u1M4	u2M4	u3M4
	Error = R .* Error;

	%Square the error 
	Error = Error .^2;
	
	costJ = sum(sum(Error));
	
	costJ = costJ/2;
% =============================================================
	%compute Gradient for X and Theta
	predvalue = X * Theta';
	Error = predvalue - Y; 
	Error = R .* Error;
	X_grad = Error * Theta;
	Theta_grad = Error' * X;
	
	%compute regularization factor
	ThetaTemp = Theta .^ 2;
	ThetaTemp = sum(sum(ThetaTemp));
	RegFactor1 = ThetaTemp * (lambda/2);
	
	xTemo = X .^ 2;
	xTemo = sum(sum(xTemo));
	RegFactor2 = xTemo * (lambda/2);
	
	costJ = costJ + RegFactor2 + RegFactor1;
	
	%compute reg factor for grads
	xgradReg = lambda .* X;
	X_grad = X_grad .+ xgradReg;
	
	
	
	
	%tgradReg = sum(Theta,2);
	tgradReg = lambda .* Theta;
	Theta_grad = Theta_grad .+ tgradReg;
%compute the regularization factor
grad = [X_grad(:); Theta_grad(:)];

end
