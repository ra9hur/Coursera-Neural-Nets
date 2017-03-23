function ret = initial_model(n_hid)
  n_params = (256+10) * n_hid;
  as_row_vector = cos(0:(n_params-1));
  ret = theta_to_model(as_row_vector(:) * 0.1); % We don't use random initialization, for this assignment. This way, everybody will get the same results.
end