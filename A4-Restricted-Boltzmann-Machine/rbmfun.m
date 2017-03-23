function [hidbiases, vishid] = rbmfun(data,numhid,maxepoch)

epsilonw      = 0.05;
epsilonvb     = 0.05;
epsilonhb     = 0.05;

initialmomentum  = 0.5;
finalmomentum    = 0.9;


  [numdims numcases]=size(data);

  epoch=1;

  vishid     = 0.03*randn(numhid, numdims);
  hidbiases  = 0*ones(numhid, 1);
  visbiases  = zeros(numdims, 1);
  vishidinc  = zeros(numhid, numdims);
  hidbiasinc = zeros(numhid, 1);
  visbiasinc = zeros(numdims, 1);

for epoch = epoch:maxepoch,
  poshidprobs = 1./(1 + exp(-vishid*data - repmat(hidbiases,1,numcases)));    
  posprods    = poshidprobs * data';
  poshidact   = sum(poshidprobs, 2);
  posvisact = sum(data, 2);

%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%

poshidstates = poshidprobs > rand(numhid, numcases);

%%%%%%%%  START NEGATIVE PHASE  %%%%%%%%%

  negdata = 1./(1 + exp(-vishid'*poshidstates - repmat(visbiases,1, numcases)));
  neghidprobs = 1./(1 + exp(-vishid*negdata - repmat(hidbiases,1, numcases)));    
  negprods  = neghidprobs*negdata';
  neghidact = sum(neghidprobs, 2);
  negvisact = sum(negdata, 2); 

%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%

  errsum= sum(sum( (data-negdata).^2 ));
   if epoch>5,
     momentum=finalmomentum;
   else
     momentum=initialmomentum;
   end;

   %%%%% THIS IS WHERE YOU COMPUTE THE GRADIENTS, %%%%%
	   
   %%%%% REMEMBER TO DIVIDE BY THE NUMBER OF CASES. %%%%%
	
	vishidinc = momentum*vishidinc + epsilonw*vishidgrad;
    visbiasinc = momentum*visbiasinc + epsilonvb*visbiasgrad;
    hidbiasinc = momentum*hidbiasinc + epsilonhb*hidbiasgrad;
	
    vishid = vishid + vishidinc;
    visbiases = visbiases + visbiasinc;
    hidbiases = hidbiases + hidbiasinc;


    if rem(epoch,10) ==0,
      fprintf(1, 'numhid %4.0i epoch %4.0i  reconstruction error %6.1f  \n', numhid, epoch, errsum); 
    end;
   
end;