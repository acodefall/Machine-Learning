%implement sigmoid functions
function sig = sigmoid(z)
	%z has theta'*X
	%output the SIGMOID function output


	%Z is matrix
	%fprintf('z r: %f  c: %f \r', rows(z), columns(z));
	
	%Sigmoid implementation using STEPS
	%Sigmoid = (1/(1+e^(-z)) 
	% a) "-z" means element-wise multiplicatoion using -1
		Zi = -z;
		
		% b) Apply EXP element-wise on "-z"
		Ze =  e.^Zi;
		
		% c) Add 1 to every element (addition operation is ELEMENt-wise by default
		Za = 1 + Ze;
		
		% e) Do inverse element-wise
		sig = 1 / Za;
	
	%Sigmoid implementation in one line
	%All in one line
		sig = 1 ./ (1 + e.^ (-z));
	
	%fprintf('G r: %f  c: %f \r', rows(g), columns(g));

end
