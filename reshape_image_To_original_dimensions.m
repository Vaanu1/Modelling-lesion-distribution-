function reshaped_image = reshape_image_To_original_dimensions(urimage,k,sz)

n=length(sz);
indx2=nan(1,n);
%dimensions other k-th dimension, along which convolution will happen:
otherdims = 1:n; otherdims(k)=[];

% permute order: place k-th dimension as 1st, followed by all others:
indx1=[k otherdims];

% inverse permute order:
indx2(indx1)=1:n;

J = reshape(urimage,sz(indx1));
%5. undo the permutation of step 1.
reshaped_image = permute(J,indx2);
