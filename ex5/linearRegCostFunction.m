function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% ------------------------------------------------------------
% Unregularized cost function and gradient
% ------------------------------------------------------------

% compute the h function
h = X * theta;

% compute the difference with y
error = h - y;

% square it element by element
error_sqr = error.^2;

% calculate J
J = (1 / (2*m)) * sum(error_sqr);

% calculate gradient
grad = (1 / m) * (X' * error);

% ------------------------------------------------------------
% Regularization
% ------------------------------------------------------------

% set theta(1) to 0
theta(1) = 0;

% calculate regularized J
J = J + (lambda / (2*m)) * sum(theta.^2);

% calculate regularized gradient
grad = grad + (lambda / m) * theta;

% =========================================================================

grad = grad(:);

end
