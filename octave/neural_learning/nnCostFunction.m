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

% perform foward propagation
% a1 = X; a2 = sigmoid(z2)
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = [ones(m,1) sigmoid(z2)];
% h_theta is output = a3 = sigmoid(z3)
h_theta = sigmoid(a2 * Theta2');

% y(k) - recode the labels as vectors of 0 or 1
yk = zeros(num_labels, m);
for i=1:m,
  yk(y(i),i)=1;
end

J = (1/m) * sum(sum((-yk) .* log(h_theta') - (1-yk) .* log(1 - h_theta')));

% calculate regularized cost without the bias term
% get rid of the bias
theta1 = Theta1(:,2:(size(Theta1,2)));
theta2 = Theta2(:,2:(size(Theta2,2)));

% lambda/2m * Sum(Sum(Theta1)^2) + Sum(Sum(Theta2)^2)
regularized_sum = (lambda/(2*m)) * (sum(sum(theta1 .^ 2)) + sum(sum(theta2 .^ 2)));

% regularized cost
J = J + regularized_sum;

%
% vectorized backpropagation algorithm
%
% first calculate delta for output unit
delta3 = h_theta - yk';
delta2 = (delta3 * theta2) .* sigmoidGradient(z2);

% now calculate the partial derivatives
Theta1_grad = delta2' * a1;
Theta2_grad = delta3' * a2;

% need to zero out bias term
Theta1_grad = Theta1_grad / m + (lambda / m) * [zeros(hidden_layer_size,1) Theta1(:,2:end)];
Theta2_grad = Theta2_grad / m + (lambda / m) * [zeros(num_labels,1) Theta2(:,2:end)];

%%
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
