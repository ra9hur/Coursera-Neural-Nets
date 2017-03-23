function [testloss] = run_experiment(n_hid, pretrain_iters, train_iters, train_learnrate)

rand('twister',8);
randn('seed',8);

from_data_file = load('a4data.mat');
a4data = from_data_file.a4data;

if pretrain_iters > 0
	[hidbiases, vishid] = rbmfun(a4data.training.inputs_unlabelled, n_hid, pretrain_iters);
	model = struct('input_to_hid', vishid, 'hid_to_class', .01*randn(10, n_hid));
	testloss = a4(a4data, 0, model, train_iters, train_learnrate, 0.9, false, 100);
else
	model = struct('input_to_hid', .01*randn(n_hid, 256), 'hid_to_class', .01*randn(10, n_hid));
	testloss = a4(a4data, 0, model, train_iters, train_learnrate, 0.9, false, 100);
end

end
