load('MNI152_T1_2mm_brain_mask.mat');
load('weights.mat');

% Randomly defining theta and getting brain mask

theta = zeros(91,109,91,10);
theta_in = theta(:);
mask_in = repmat(mask_image,[1,1,1,size(theta,4)]);
mask = mask_in(:);

% weights for specifying that more samples has to selected from center of
% the brain. The ewights hav high values in the center and low values in
% the periphery
weights = weights(:);
weights_in = weights(mask>0);

% specifying number of voxels to sample and randomly sampling the voxels
% within the brain extracted image
no_brainvoxels = numel(mask(mask>0));
fraction_voxels = round(0.001*no_brainvoxels);
 
rand_voxels = randsample(1:no_brainvoxels,fraction_voxels,true,weights_in);

theta_randvoxels = theta_in(mask>0);
theta_randvoxels(rand_voxels) = theta_randvoxels(rand_voxels) + 25;

theta_in(mask>0) = theta_randvoxels;
theta_in = reshape(theta_in,size(theta));

% Forming meaningful lesionlike regions by clumping the voxels
theta_in = imdilate(theta_in,strel('square',2));



% % mask4 = zeros(size(theta_in));
% % mask4(45:55,:,:,:) = 1;
% % mask4(:,55:65,:,:) = 1;
% % theta_in(mask4>0) = 0.05;
kernel1D = fspecial('gaussian',[1,3],3);
% kernel1D = kernel1D./5;
theta_in = convnsep({kernel1D,kernel1D,kernel1D,kernel1D},theta_in,'same');

ni = convnsep({kernel1D,kernel1D,kernel1D,kernel1D},ones(size(theta_in)),'same');
ni3 = convnsep({kernel1D,kernel1D,kernel1D},ones(size(theta_in(:,:,:,1))),'same');
theta_in=theta_in./ni;
% theta_in = normalise(theta_in);
% theta_in(mask == 0) = 0.05;
theta_in(theta_in<0.4) = 0;
theta_in=max(theta_in,0.01);
theta_in=min(theta_in,0.99);

mu0=norminv(theta_in,0,1);
mu0(isinf(mu0)) = 1;

data_simulated = [];
numbers= randsample(400:500,10);
Rmat = zeros(size(mu0));
for n=1:10 
    pim=zeros(size(mu0(:,:,:,n)));
  for nn = 1:numbers(n)  
      f0=randn(size(mu0(:,:,:,n)))+mu0(:,:,:,n);
      f1=convnsep({kernel1D,kernel1D,kernel1D},f0,'same');  
       f1 = f1./ni3 ;
       f1 = f1>0;
      pim=pim+f1;
  end
  Rmat(:,:,:,n) = pim;
  
end
pim=pim/10; % range of pim = [0,1]

Nmat = zeros(size(mu0));% Instead of varying n above, I ahve forming N matrix with random number of subjects in each age group
for i = 1:size(mu0,4)
Nmat(:,:,:,i) = repmat(numbers(i),[91,109,91]);
end


[x,histout,costdata] = maximizing_function(Nmat, Rmat, mask_in>0); % optimizing function (to get lambda_cap)
xcimage = reshape((1+tanh(x))./2,size(Nmat)); % getting theta_cap

% sp1 = spmak(1:6:37,[1 1 1]);
% val = fnval(sp1,1:37);
sp1 = spmak(1:2:11,[1 1 1]);
val = fnval(sp1,1:11);
sp11 = spmak(1:1:7,[1 1 1]);
val1 = fnval(sp11,1:7);
% sp1 = spmak(1:4:23,[1 1 1]);
% val = fnval(sp1,1:23);
val = val./(sum(val));
val1 = val1./(sum(val1));

normalizing_image = convnsep({val,val,val,val1},ones(size(Rmat)),'same');

xc_theta = convnsep({val,val,val,val1},xcimage,'same');

xc_theta1 = xc_theta./normalizing_image;

xc_theta1(~mask) = 0;

figure(66),imagesc(xc_theta1(:,:,45,5));colorbar

kernel1D = fspecial('gaussian',[1,11],3);
kernel1D = kernel1D./sum(kernel1D);

kernel1Ds = fspecial('gaussian',[1,5],3);
kernel1Ds = kernel1Ds./sum(kernel1Ds);
ni = convnsep({kernel1D,kernel1D,kernel1D,kernel1Ds},ones(size(theta_in)),'same');
mu1=convnsep({kernel1D,kernel1D,kernel1D,kernel1Ds},mu0,'same');
mu1 = mu1./ni;
var1=convnsep({kernel1D.^2,kernel1D.^2,kernel1D.^2,kernel1Ds.^2},ones(size(mu0)),'same');
var1 = var1./var1;
z1=mu1./sqrt(var1);

theta1=normcdf(z1);% estimation of true theta
theta1(mask == 0) = 0;
error = theta1 - xc_theta1;

figure(67),imagesc(error(:,:,45,5));caxis([-0.3 0.3]);colorbar

figure(68),imagesc(theta1(:,:,45,5));colorbar

s = error.^2;
mse = sum(s(:))/(91*109*91*10)
rmse_percent = sqrt(mse/mean(theta1(:)>0))


ratio = Rmat./Nmat;
xc_ratio = convnsep({val,val,val,val1},ratio,'same');

xc_ratio = xc_ratio./normalizing_image;
figure(69),imagesc(xc_ratio(:,:,45,5));colorbar

error_rn = theta1 - xc_ratio;
s_rn = error_rn.^2;
mse_rn = sum(s_rn(:))/(91*109*91*10)
rmse_percent_rn = sqrt(mse_rn/mean(theta1(:)>0))

% save('experiment_rn_details250.mat','theta1','xc_theta1','error','error_rn','ratio','mse','mse_rn',...
%     'rmse_percent','rmse_percent_rn');