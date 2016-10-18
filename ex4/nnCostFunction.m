function [J grad] = nnCostFunction(nn_params, ...
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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
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

% -------------------------------------------------------------
% Part 1.1 : Forward Propagation
% -------------------------------------------------------------

% implement the y_matrix
y_matrix = eye(num_labels)(y,:);

% add a first column to X to create a1
a1 = [ones(m, 1) X];

% multiply by Theta1 and it becomes 'z2'
z2 = a1 * Theta1';

% take the sigmoid, add a column of 1's, and it becomes 'a2'
a2 = [ones(m, 1) sigmoid(z2)];

% multiply by Theta2, take the sigmoid() and it becomes 'a3'
a3 = sigmoid(a2 * Theta2');

% cost function
cost = y_matrix .* log(a3) + (1-y_matrix) .* log(1-a3);

% cost
J = -(1/m) * sum(sum(cost));

% -------------------------------------------------------------
% Part 1.2 : Regularization
% -------------------------------------------------------------

% select Theta without the first column
Theta1noBias = Theta1(:, 2:end);
Theta2noBias = Theta2(:, 2:end);

% calculate regularization term
reg = (lambda / (2 * m)) * (sum(sum(Theta1noBias .^ 2)) + sum(sum(Theta2noBias .^ 2)));

% final regularize cost function
J = J + reg;

% -------------------------------------------------------------
% Part 2 : Back Propagation
% -------------------------------------------------------------

% calculate delta3
d3 = a3 - y_matrix;

% calculate delta2
d2 = d3 * Theta2noBias .* sigmoidGradient(z2);

% calculate Delta1
Delta1 = d2' * a1;

% calculate Delta2
Delta2 = d3' * a2;

% finally, calculate the Thetas
Theta1_grad = (1/m) .* Delta1;
Theta2_grad = (1/m) .* Delta2;

% -------------------------------------------------------------
% Part 3 : Regularization
% -------------------------------------------------------------

% set the first column of Theta1 and Theta2 to all-zeros
Theta1(:, 1) = 0;
Theta2(:, 1) = 0;

% scale each Theta matrix by λ/m
Theta1 = (lambda/m) .* Theta1;
Theta2 = (lambda/m) .* Theta2;

% add each of these modified-and-scaled Theta matrices to the un-regularized Theta gradients
Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
