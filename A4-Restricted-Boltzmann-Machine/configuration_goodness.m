function G = configuration_goodness(rbm_w, visible_state, hidden_state)
% 100 * 256 <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% 256 * 1,10,37 <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% 100 * 1,10,37 <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.

    size(rbm_w)
    size(visible_state)
    size(hidden_state)
    
    G =  sum(sum((rbm_w*visible_state).*hidden_state))/size(visible_state,2);

    % error('not yet implemented');
end
