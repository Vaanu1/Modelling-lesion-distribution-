% Optimization step for finding the coefficient matrix :)

tic

sp1 = spmak(1:2:15,[1 1 1]);
val = fnval(sp1,1:15);
val = val./sum(val);

load('MNI152_T1_2mm_brain_mask.mat');
load('NR_details.mat');

% compensating for convolution error
normalizing_image = convnsep({val,val,val,val},ones(size(Rmat)),'same');

% getting the brain mask, R and N matrices ready
mask = mask_image > 0;
mask = repmat(mask,[1,1,1,27]);
Rmat(~mask) = 0;
Nmat(~mask) = 0;

c00 = 0.5+zeros(size(Nmat));%Rmat./Nmat;
c00(~mask) = 0;
c0 = zeros(size(c00));
c0(1:2:end,1:2:end,1:2:end,1:2:end) = c00(1:2:end,1:2:end,1:2:end,1:2:end);
c01 = convnsep({val,val,val,val},c0,'same');
%c01 = (c01./normalizing_image).*(max(normalizing_image(:)));% smoothed average lesion map (initialization)
c01 = normalise((c01./normalizing_image));

lc01 = c01;
lc02 = (convnsep({val,val,val,val},lc01,'same')./normalizing_image).*(max(normalizing_image(:)));

lc01(c01 == 0) = 0;
lc01_norm = normalise(lc01);
lc01_norm(c01 == 0) = 0;
lc01_norm = normalise(lc01_norm);

tmp = 2*lc01 - 1;
% tmp = normalise(convnsep({val1,val1,val1,val1},c00,'same')./convnsep({val1,val1,val1,val1},ones(size(Rmat)),'same'));
tmp(tmp<-0.99) = -0.99;
tmp(tmp>0.99) = 0.99;
lambda = atanh(tmp);
% lambda = convnsep({val,val,val,val},lambda,'same')./convnsep({val,val,val,val},ones(size(Rmat)),'same');

func = @(C) derivative_function_opt(C,Nmat,Rmat,normalizing_image,mask);

%[xc,histout,costdata,itc] = cgtrust(lc01_norm(:),func,[10,0.1,60,20],1e-8);

%[x,histout,costdata] = steep(lambda(:),func,0,50);
[x,histout,costdata] = stoch_grad_descent(lambda(:),func);
xcimage = reshape((1+tanh(x))./2,size(c01));
xc_theta = convnsep({val,val,val,val},xcimage,'same');

xc_theta1 = xc_theta./normalizing_image;

xc_theta1(~mask) = 0;
figure(66),imagesc(xc_theta1(:,:,45,13));


% % % options = optimset('Gradobj','on','Algorithm','trust-region-reflective','Hessian','off','Display','iter','MaxIter',5,'MaxFunEvals',1,...
% % % 'FinDiffRelStep',10*ones(size(Nmat)),'TolX',1e-12,'TolFun',1e-12);%,'TolCon',1e-13,'TolGradCon',1e-13,'TolProjCGAbs',1e-13);
% % % [cout,fval] = fmincon(func,c01,[],[],[],[],-1*ones(size(Nmat)),Inf*ones(size(Nmat)),[],options);
% options.MaxIter = 10;
% [cout,fval] = fminunc(func,c01,options);
toc



