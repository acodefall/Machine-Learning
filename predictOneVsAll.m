%predit the output using one-v/s-all method
%This is for Nueral Network. Number of YTheta gives the number Lables.
%LABELS consists of number starting from 1. 
%Assume 1, 2,....Num_Labels as LABELS
%Compute theata'T, then apply Sigmoid, then assign a label
function vectOfPredictions = predictOneVsAll(all_theta, X)
	m = size(X, 1);
	num_labels = size(all_theta, 1);

	% You need to return the following variables correctly 
	vectOfPredictions = zeros(size(X, 1), 1);

	% Add ones to the X data matrix
	X = [ones(m, 1) X];

	fprintf('M  %f  num_labels %f\n', m, num_labels);
	fprintf('K  %f \n', size(all_theta, 1));
	fprintf('X  %f  %f\n', rows(X), columns(X));
	fprintf('p  %f  %f\n', rows(vectOfPredictions), columns(vectOfPredictions));
	fprintf('all_theta  %f  %f\n', rows(all_theta), columns(all_theta));



	%create matrix of m X m for storing the output of sigmod() for every class.
	%Apply S() for whole training-set, to test for class-1, and store the output in hAll[1]
	%Apply S() for whole training-set, to test for class-2, and store the output in hAll[2]
	%Apply S() for whole training-set, to test for class-3, and store the output in hAll[3]
	hAll = zeros(rows(X),num_labels);
	
	for i = 1: num_labels 
		%Fetch ROW from all_theta, and turn it in to Column-Matrix
		thetTemp  = all_theta(i,:); %Fetch ROW for class1 from [all_theta]
		thetTemp  = thetTemp';      %costruct column-Matrix out of it.
		
		fprintf('thetTemp  %f  %f\n', rows(thetTemp), columns(thetTemp));
		
		%h is column-matrix. store this as COLUMN in hAll
		h = sigmoidFunc(X * thetTemp);
		fprintf('h[%f]  %f  %f\n', i, rows(h), columns(h));
		
		for j = 1: m 
			hAll(j,i) = h(j);
		end
	end

	for i = 1: num_labels 
		%Fetch ROW from all_theta, and turn it in to Column-Matrix
		thetTemp  = all_theta(i,:); %Fetch ROW for class1 from [all_theta]
		thetTemp  = thetTemp';      %costruct column-Matrix out of it.
		
		fprintf('thetTemp  %f  %f\n', rows(thetTemp), columns(thetTemp));
		
		%h is column-matrix. store this as COLUMN in hAll
		h = sigmoidFunc(X * thetTemp);
		fprintf('h[%f]  %f  %f\n', i, rows(h), columns(h));
		
		for j = 1: m 
			hAll(j,i) = h(j);
		end
	end
	
	%find the MAX value's index in each row
	[f, k] = max(hAll, [], 2);
	vectOfPredictions = k;
	fprintf('p %f  %f\n',  rows(vectOfPredictions), columns(vectOfPredictions));
	



end
