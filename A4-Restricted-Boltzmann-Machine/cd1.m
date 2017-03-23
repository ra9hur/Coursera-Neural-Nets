function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.

# Including for Q8
# visible_data = sample_bernoulli(visible_data);

% get conditional probabilities of hidden states
% calculate h0_prob from visible_data
  h0_prob = logistic(rbm_w*visible_data);

% sample hidden states
% sample h0 from h0_prob
  h0 = sample_bernoulli(h0_prob);

% calculate gradient0 from visible_data and h0
  gradient0 = configuration_goodness_gradient(visible_data, h0);

% get visible state probabilities
% calculate v1_prob from h0
  v1_prob = logistic(rbm_w'*h0);
% sample visible states
% sample v1 from v1_prob
  v1 = sample_bernoulli(v1_prob);

% get hidden state probabilities
% calculate h1_prob from v1
  h1_prob = logistic(rbm_w*v1);

% sample hidden states
% sample h1 from h1_prob
  h1 = sample_bernoulli(h1_prob);                          # commenting for Q7
% calculate grandient1 from v1 and h1
  gradient1 = configuration_goodness_gradient(v1, h1);     # commenting for Q7
#  gradient1 = configuration_goodness_gradient(v1, h1_prob);

% update weight matrix with gradient1 and gradient2 and return it.
% The returned value is the gradient approximation produced by CD-1
% return the difference between gradient0 and gradient1
% I had the same except the last step (returning the difference between gradients)
% As Aliaksandr points out slides 23 and 24 of lecture 12 give the equation for it
  ret = gradient0 - gradient1;    # visible_data*h0' - v1*h1'

    %    error('not yet implemented');
end

