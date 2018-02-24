function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hx = sigmoid(X*theta);
hx_y = hx - y;

J = (1/m*sum(-y.*log(hx)-(1-y).*log(1-hx))) ...
		+ (lambda/(2*m)*sum(theta(2:size(theta)) .^ 2));

%grad(1) = 1/m * sum(hx_y .* X(:, 1));
%for j = 2: size(theta)
%	grad(j) = 1/m * sum(hx_y .* X(:, j)) + lambda/m * theta(j);
%end

grad = 1/m * X'*hx_y;

reg_temp = lambda/m * theta;
reg_temp(1) = 0; 									% for Theta0, not apply regularization;
grad = grad + reg_temp;

%     				OR
%reg_temp = lambda/m * theta(2:end, :);
%grad(2:end, :) = grad(2:end, :) + reg_temp;

% =============================================================

end
