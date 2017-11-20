%return Gaussian distance between X1 and X2
function sim = gaussianKernel(x1, x2, sigma)

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;
%fprintf('x1  %f  %f\n', rows(x1), columns(x1)); 3 x 1
%fprintf('x2  %f  %f\n', rows(x2), columns(x2)); 3 x 1
%fprintf('sigma  %f \n', sigma);




	sim = sum((x1 - x2) .^2);
	sim = sim /(2*(sigma^2));
	sim = exp(-sim);


end
