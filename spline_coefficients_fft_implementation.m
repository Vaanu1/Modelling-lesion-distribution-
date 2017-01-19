length = 7;

% smoothing spline and its x-shift
sp = spmak(1:1:length,[1 1 1]);
val = fnval(sp,1:length);
val = val./sum(val);

% a = rand(91,109,91,10);
% a1 = convnsep({val,val,val,val},a,'same')./convnsep({val,val,val,val},ones(size(a)),'same');
output = theta1;

%  Creating M-matrix (spline coefficients)
[row,col,stack,time] = size(output);

flag = 0;
for i = [1 2 3 4]
    
    
    [output_ur,output_sz] = unravel_image(output,i);
    S = zeros(1, size(output,i));
    S(end-2:end) = val(1:3);
    S(1:4) = val(4:end);
%     S = val;
    Yf = fft(output_ur);
    Sf = fft(S');
    Ck = Yf./repmat(Sf,[1,size(output_ur,2)]);
    M = zeros(numel([0:1:size(output_ur,1)-1]),size(output_ur,1));
    for k = 1:size(output_ur,1)
        column = exp((-1i* 2*pi * (k-1) * [0:1:size(output_ur,1)-1])./size(output_ur,1));
        M(:,k) = column';
    end
    M = M';
    inv_term = M'*M;
    coeff= (inv_term)\M'*Ck;
    ind = ones(1,4);
    ind(i) = 2;
    coeff_matrix = reshape_image_To_original_dimensions(coeff,i,round(output_sz));%./ind));
    output = real(coeff_matrix);
    
end

aout = convnsep({val,val,val,val},output,'same')./convnsep({val,val,val,val},ones(size(output)),'same');
