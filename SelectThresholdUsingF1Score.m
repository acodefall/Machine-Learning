%Computes Precision and Rrecall, and use them to compute F1 Score
%select the threshold with highest F1 score
function [bestEpsilon bestF1] = SelectThresholdUsingF1Score(yval, pval)

	bestEpsilon = 0;
	bestF1 = 0;
	F1 = 0;

	stepsize = (max(pval) - min(pval)) / 1000;
	for epsilon = min(pval):stepsize:max(pval)
		% Instructions: Compute the F1 score of choosing epsilon as the
		%               threshold and place the value in F1. The code at the
		%               end of the loop will compare the F1 score for this
		%               choice of epsilon and set it to be the best epsilon if
		%               it is better than the current choice of epsilon.
		%               
		% Note: You can use predictions = (pval < epsilon) to get a binary vector
		%       of 0's and 1's of the outlier predictions
		m = rows(yval);
		tp = 0;
		fp = 0;
		fn = 0;
		
		for i=1:m
			if(pval(i) < epsilon) %anamoly
				if(yval(i) == 1) %label says anamoly
					tp = tp + 1;
				else
					fp = fp + 1;
				endif
			else  %Not anamoly
				if(yval(i) == 0) %label says not-anamoly
					; 
				else
					fn = fn + 1;
				endif
			endif
		end
		%now compute recall and precision
		prec = tp/(tp+fp);
		rec = tp/(tp+fn);
		
		%compute f1score
		F1 = (2*(rec*prec))/(prec+rec);
		
		%F1 best when it is 1
		if(F1 > bestF1)
			bestEpsilon = epsilon;
			bestF1 = F1;
		endif		
	end

end
