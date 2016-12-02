% PERMUTATION ANALYSIS

%% Loading the essentials

mask = load('MNI152_T1_2mm_brain_mask.mat');
mask = mask.mask_image > 0;

%% Data accummuation from OXVASC in cluster

addpath('/home/fs0/vaanathi/Documents/MATLAB/');

filepath = '/vols/Data/Oxvasc/';

oxvasc = load('../Information-excels/oxvasc_masterdata474.mat');

oxvasc = oxvasc.OXVASCforVaanu;

filenames = {oxvasc{:,1}};
ages = [];
for i = 1:numel(filenames)
    age = oxvasc{i,3};
    ages = [ages;age];
end

index = find(~cellfun(@isempty,oxvasc));
[r,c] = ind2sub(size(oxvasc),index);
ind = find(c == 2);
reject = r(ind);

accept = setdiff(1:numel(filenames),reject);
nfilenames = filenames(accept);
nages = ages(accept);

%% Correct age order analysis (ascending order)
[sorted_value,sorted_index] = sort(nages,'ascend');

sfilenames = nfilenames(sorted_index);
sages = sorted_value;

sorted_images = zeros(91,109,91,numel(sfilenames));
wrong_indices = [];
for i = 1:numel(sfilenames)
    %try
        %inp_image = load_untouch_nii([filepath sfilenames{i} '/structural/testsub' sfilenames{i} 'seg_HIGHles_2knoborder10knnles_GMsubEXCL_thr09bin_2std_nlin_Thrbin05.nii.gz']);
        iimage = load_untouch_nii('../../Rotation_project1_brainMRI/Kernel_density_Estimation/smoothed_spline.nii.gz');
        fprintf('Data got');
        numel(sfilenames)
        inp_image = double(inp_image.img);
        sorted_images(:,:,:,i) = inp_image;
    %catch 
        %wrong_indices = [wrong_indices,i];
%         sorted_images(:,:,:,i) = [];
%         sorted_value(i)=[];
    %end
end
save('sorted_images.mat','sorted_images');
sorted_images(:,:,:,wrong_indices) = [];
sorted_value(wrong_indices) = [];

[h,c] = hist(sorted_value,27);
edges = [min(sorted_value) c(2:end-1) max(sorted_value)];

[indices] = discretize(sorted_value,edges);

Rmat = zeros(91,109,91,27);
Nmat = zeros(91,109,91,27);

for j = 1:27
    ind = find(indices == j);
    Rmat(:,:,:,j) = sum(sorted_images(:,:,:,ind),4);
    Nmat(:,:,:,j) = j.*ones(91,109,91,1);
end

[x,histout,costdata,xc_theta_sorted] = maximizing_function(Nmat, Rmat, mask>0);

%% Permutation analysis

niterations = 200;
theta_map = zeros(91,109,91,27,niterations);

for iter = 1:niterations
    
    [indices] = randsample(27,numel(sorted_value),true);

    Rmat = zeros(91,109,91,27);
    Nmat = zeros(91,109,91,27);

    for j = 1:27
        ind = find(indices == j);
        Rmat(:,:,:,j) = sum(sorted_images(:,:,:,ind),4);
        Nmat(:,:,:,j) = j.*ones(91,109,91,1);
    end

    [x,histout,costdata,xc_theta_unsorted] = maximizing_function(Nmat, Rmat, mask>0);
        
    theta_map(:,:,:,:,iter) = xc_theta_unsorted;
    
end

% save('final_details.mat','theta_map','xc_theta_sorted','wrong_indices','sorted_images','sorted_value');

%% Finding the significantly varying voxels
mask_img = repmat(mask,[1,1,1,size(theta_map,4)]);
correctly_sorted_theta = xc_theta_sorted;
diff_sortedmap = diff(correctly_sorted_theta,4);
sum_sqdiff_sortedmap = sum(diff_sortedmap.^2,4);
permuted_theta_map = theta_map;
max_thresholds = zeros(1, size(theta_map,5));

for ii = 1:size(theta_map,5)
    
    permuted_map = permuted_theta_map(:,:,:,:,ii);
    diff_permutedmap = diff(permuted_map,4);
    sum_sqdiff_permutedmap = sum(diff_permutedmap.^2,4);
    maskv = mask(:);
    sum_sqdiff_permutedmap_vector = sum_sqdiff_permutedmap(:);
    
%     top10_intensity = prctile(sqdiff_permutedmap_vector(maskv>0),90);
%     mean_topintensity = mean(sqdiff_permutedmap_vector(sqdiff_permutedmap_vector >= top10_intensity));
    max_intensity = max(sum_sqdiff_permutedmap_vector(maskv>0));
    max_thresholds(ii) = max_intensity;
end

threshold95 = prctile(max_thresholds,95);

significance_map95 = sum_sqdiff_sortedmap > threshold95;
% significance_map90 = sum_sqdiff_sortedmap > threshold90;

