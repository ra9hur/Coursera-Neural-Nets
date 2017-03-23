load a4data

rand('twister',8);
randn('seed',8);

n_hid=50;
pretrain_iters=50;
[hidbiases, vishid] = rbmfun(a4data.training.inputs_unlabelled, n_hid, pretrain_iters);
	
clear