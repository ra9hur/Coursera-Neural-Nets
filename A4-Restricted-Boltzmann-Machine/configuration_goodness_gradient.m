function d_G_by_rbm_w = configuration_goodness_gradient(visible_state, hidden_state)
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% <hidden_state> is a (possibly but not necessarily binary) matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% You don't need the model parameters for this computation.
% This returns the gradient of the mean configuration goodness (negative energy, as computed by function <configuration_goodness>) with respect to the model parameters. 
% Thus, the returned value is of the same shape as the model parameters, which by the way are not provided to this function. 
% Notice that we're talking about the mean over data cases (as opposed to the sum over data cases).
% 100 * 256 <rbm_w>
% 256 * 1,10,37 <visible_state>
% 100 * 1,10,37 <hidden_state>

% If, on average, the two units take on the same value the weight will be positive
% If the units take opposite values we get negative weights predominating
    d_G_by_rbm_w = hidden_state * visible_state' / size(visible_state,2);

%    error('not yet implemented');
end
