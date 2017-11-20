%This is two layer Nueral Network(input layer, hidden layer).
function [costJ grad] = NeuralNetworkCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%fprintf('m  %f  num_labels %f\n', m, num_labels); %num_labels = 10
%fprintf('y  %f  %f\n', rows(y), columns(y)); %5000 x 1 
%fprintf('X  %f  %f\n', rows(X), columns(X)); %5000 x 400
%fprintf('Theta1  %f  %f\n', rows(Theta1), columns(Theta1));  %25 x 401
%fprintf('Theta2  %f  %f\n', rows(Theta2), columns(Theta2));  %10 x 26


%Theta1 shows 401 features where as X has 400 column
%so add one extra column to X to accomadate Theta0 (do this after TRANSPOSE).
X = [ones(rows(X), 1) X];; %add a row to top
%fprintf('X  %f  %f\n', rows(X), columns(X)); %5000 x 400
X = X';

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
%   
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
	
	%Cost = PredictedValue - ActualCost
	%Computing cost requires to find PredictedValue or h(x). We do that using Forward Prop.
	%Then use the Actual Cost given in Y to compute the COST.
	D1 = zeros(rows(Theta1), columns(Theta1));
	D2 = zeros(rows(Theta2), columns(Theta2));
	hAll = zeros(1,num_labels); %TrainingSet x ClassIndex
	JNoReg = 0;
	RegFactor = 0;
	errors0 =0;
	errors1 = 0;
	%fprintf('Y data \n', i, rows(a2), columns(a2));
	%read the value from Y and enable it in h(x) 
	
	for i=1:m
		%%%  FF Prop
			a1 = X(:, i); %X training set as column-vector so read one column at a time
						  %set input as Input layer
			%fprintf('a1  %f  %f\n', rows(a1), columns(a1));
			
			z2 = Theta1 * a1;
			a2 = sigmoid(z2); 	%compute the values for Layer-2
										%h will give out values as [column-vector]
			%fprintf('a2[%f]  %f  %f\n', i, rows(a2), columns(a2));
			
			
			a2 = [ones(1,columns(a2)) ; a2]; %add a row to top to Layer-2
			%fprintf('a2[%f]  %f  %f\n', i, rows(h), columns(h));
			
			z3 = Theta2 * a2;
			a3 = sigmoid(Theta2 * a2); %compute the values for Layer-3 (Output layer)
									  %h will give out values as [column-vector]
									  %this is also the PredicatedValues or h(x)		
										%sigmoidGradient
			%store the Predicated value or a3, as a ROW(row-vector) within hAll
			hAll = [hAll; a3'];
		%%%%%%%%
		
		%%%  Compute Cost (PredValue - ACtualCost)
			% Y identifies the output by number 1, 2, 3. We need to convert this in to 0 or 1
			% "y[] = 5" should become hx2 = [0; 0; 0; 0; 1;]
			% Instead of saying output is 5, we make the 5 element as 1, in 10 element array
			actualValue = zeros(num_labels,1); %create 10 element column-vector to hold h(x)
			actualValue(y(i),1) = 1; %read the value from Y and enable it in h(x) 
		
			%we have to compute error for every node of a3.
			%if the a[x] = 0, then use [0; 0; 0; 0;]
			%if the a[3] = , then use [0;0;1;0;]
			%There will be K nodes in Output layer. Among K nodes, 
			%consider only one vector whos value is TRUE, ignore the others. 
			%This means SUMMATION will be using only node that gave TRUE; all other 
			%nodes will be ignored.
			%
			errors0 = (actualValue') * log(a3);
			errors1 = (1 - actualValue)' *log(1 - a3);
			JNoReg = JNoReg + (errors0 + errors1);
		%%%%%%%%
		
		%%% BackwardProp
			actualValue = zeros(num_labels,1); %create 10 element column-vector to hold h(x)
			
			actualValue(y(i),1) = 1; %read the value from Y and enable it in h(x) 
		
			%printf('Done J %f %f  %f \n', actualValue(y(i)), rows(actualValue), columns(actualValue));
		
			%Understand that small-delta of layerX is used for recomputing the values from layerX-1
			%Though we do not generate for small-delta of layer1, we do alter the Theta of Layer1, by using small-delta of layer2
			%compute small-delta for last layer
			delta3 = a3 - actualValue; %this is for individual entry in training set
			
			
			%compute small-delta for 2nd layer
			%delta2 = (delta3 * Theta2) .* sigmoidGradient (a2 .*(1-a2)); %this is for individual entry in training set
			delta2 = Theta2(:,2:end)' * delta3 .* sigmoidGradient(z2);
			
			
			%do not compute the small-delta for layer-1
			%delta1 = (Theta1' * delta2) .*a1 .*(1-a1); %this is for individual entry in training set
			
			
		
			%accumate delta values in to D
			%accumulate partial derivatives
			D2 = D2 + (delta3 * a2');
			D1 = D1 + (delta2 * a1');
			
			
		%%%%
	end
	
	
	%%%  Compute Regularization factor This is NOT done for every items in training set
			%so put outside for loop
			%Square every only a specific cell except the Top row, that is the bias row
			RegFactor = RegFactor + sum(sum(Theta1(:,2:end) .^2)); 
			RegFactor = RegFactor + sum(sum(Theta2(:,2:end) .^2));
	%%%%%
		
	%%finalize J
		RegFactor 	= (lambda/(2*m)) * RegFactor;
		JNoReg		= JNoReg/m;
		costJ = -JNoReg + RegFactor;
	%%%%
	
	%find the MAX value's index in each row. 
	%[f, k] = max(hAll, [], 2);
	%J = k;
	for g1=1:rows(D1)
		for g2=1:columns(D1)
			if(g2 > 1)
				D1(g1,g2) =  D1(g1,g2) + (D1(g1,g2)*lambda);
			end
		end
	end

	for s1=1:rows(D2)
		for s2=1:columns(D2)
			if(s2 > 1)
				D2(s1,s2) =  D2(s1,s2) + (D2(s1,s2)*lambda);
			end
		end
	end
	
	
	%Regularization of BPP is not correct
	Theta1_grad = D1 ./m;
	Theta2_grad = D2 ./m;
	

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
