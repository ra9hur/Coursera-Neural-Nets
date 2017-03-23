function ret = d_loss_by_d_model(model, data, wd_coefficient)
  % model.input_to_hid is a matrix of size <number of hidden units> by <number of inputs i.e. 256>
  % model.hid_to_class is a matrix of size <number of classes i.e. 10> by <number of hidden units>
  % data.inputs is a matrix of size <number of inputs i.e. 256> by <number of data cases>
  % data.targets is a matrix of size <number of classes i.e. 10> by <number of data cases>

  % The returned object is supposed to be exactly like parameter <model>, i.e. it has fields ret.input_to_hid and ret.hid_to_class. However, the contents of those matrices are gradients (d loss by d model parameter), instead of model parameters.
	 
  % This is the only function that you're expected to change. Right now, it just returns a lot of zeros, which is obviously not the correct output. Your job is to change that.

  M = size(data.inputs,2);

%  [hid_input, hid_output, class_input, log_class_prob, class_prob] = forward_pass(model, data);
  hid_input = model.input_to_hid * data.inputs; 
  % output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
  hid_output = logistic(hid_input); 
  % input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>
  class_input = model.hid_to_class * hid_output; 
  
  class_normalizer = log_sum_exp_over_rows(class_input); % log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
  log_class_prob = class_input - repmat(class_normalizer, [size(class_input, 1), 1]); % log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
  class_prob = exp(log_class_prob); % probability of each class. Each column (i.e. each case) sums to 1. size: <number of classes, i.e. 10> by <number of data cases>

  diff = data.targets - class_prob;

  % Output Layer
%  outderv = diff .* class_prob .* (1 - class_prob);
  outderv = -diff;

  % Hidden Layer
  hidderv = model.hid_to_class' * outderv;
  hidderv = hidderv .* hid_output .* (1 - hid_output);

% Gradient and Weight Regularization  
  W2grad = zeros(size(model.hid_to_class));
  W2grad = W2grad + outderv * hid_output';
  W2grad = 1/M * (W2grad);
  W2grad = W2grad + wd_coefficient * model.hid_to_class;
  
  W1grad = zeros(size(model.input_to_hid));
  W1grad = W1grad + hidderv * data.inputs';
  W1grad = 1/M * (W1grad);
  W1grad = W1grad + wd_coefficient * model.input_to_hid;
  
  ret.input_to_hid = W1grad;
  ret.hid_to_class = W2grad;  

  % ret.input_to_hid = model.input_to_hid * 0;
  % ret.hid_to_class = model.hid_to_class * 0;
end

%{
L2
W1grad = W1grad + wd_coefficient * model.input_to_hid;
wd_loss = sum(model.input_to_hid.^2)/2*wd_coefficient;

L1
W1grad = W1grad + wd_coefficient;
wd_loss = mod(w)*wd_coefficient;

Elastic net regularization (combine the L1 regularization with the L2 regularization)
W1grad = W1grad + L1deri + L2deri;
wd_loss = L1 + L2;

Bayesian Regularization
initially have a random lambda and train for a few rounds, then reset lambda
lambda = varianceD / varianceW;     % variance of the residual errors / variance of actually learned weight

Langevin Monte Carlo method

Drop-out

%}