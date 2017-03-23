function ret = classification_performance(model, data)
  % This returns the fraction of data cases that is incorrectly classified by the model.
  hid_input = model.input_to_hid * data.inputs; % input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
  hid_output = logistic(hid_input); % output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
  class_input = model.hid_to_class * hid_output; % input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>
  
  [dump, choices] = max(class_input); % choices is integer: the chosen class, plus 1.
  [dump, targets] = max(data.targets); % targets is integer: the target class, plus 1.
  ret = mean(double(choices ~= targets));
end