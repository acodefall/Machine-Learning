%Find the predictedValue for NN
function predictedValue = predict(Theta1, Theta2, X)
	%Theta1 theta for input-layer
	%Theta2 theta for 2nd-layer
	%X training set
		
	% Useful values
	m = size(X, 1);
	num_labels = size(Theta2, 1);

	% You need to return the following variables correctly 
	predictedValue = zeros(size(X, 1), 1);

	%fprintf('Theta1  %f  %f\n', rows(Theta1), columns(Theta1));
	%fprintf('Theta2  %f  %f\n', rows(Theta2), columns(Theta2));
	%fprintf('X  %f  %f\n', rows(X), columns(X));

	%         trg  src
	%Theta1 = 4 x 3
	%Theta2 = 4 x 5
	%X      = 16 x 2

	%Theta1' = 3 X 4

	%   X = 16 * 2 //add one column to X
	   
	X = [ones(m, 1) X];   

	aFinalAll = zeros(m,num_labels);  
	for i=1:m
		%fprintf('i  %f\n', i);
		%take one row from X and apply the WEIGHT to it
		a2 = Theta1 * X(i,:)';
		%fprintf('a2  %f  %f\n', rows(a2), columns(a2));
		%Apply the logistic regression. a3 will act as FEATURE for next round of Logistic regression
		a3 = sigmoidFunc(a2);
		%fprintf('a3  %f  %f\n', rows(a3), columns(a3));
		
	
		%add one row to top a3
		a4 =  ones(rows(a3) +1, columns(a3));   
		
		%now copy a3 to a4
		for c1=1:rows(a3) 
			for c2=1:columns(a3)
				a4(c1+1, c2) = a3(c1,c2); 
			end
		end
		
		a5 = Theta2 * a4;
		aFinal = sigmoidFunc(a5);
		%fprintf('aFinalx  %f  %f\n', rows(aFinal), columns(aFinal));
		
		%copy aFinal to a ROW in aFinalAll
		for c3=1:rows(aFinal)
			aFinalAll(i,c3) = aFinal(c3);
		end
	end

	[f,c] = max(aFinalAll, [], 2);
	predictedValue = c;
end
