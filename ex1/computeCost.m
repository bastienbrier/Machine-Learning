function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% compute the h function
h = X * theta;
% compute the difference with y
error = h - y;
% square it element by element
error_sqr = error.^2;

% calculate J
J = 1/(2*m) * sum(error_sqr);

% =========================================================================

end
