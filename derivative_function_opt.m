function [value,grad_matrix] = derivative_function_opt(C,N,R,ni,mask)
%     sp1 = spmak(1:3:19,[1 1 1]);
%     Nf = fnval(sp1,1:19);
%     sp1 = spmak(1:2:15,[1 1 1]);
%     Nf = fnval(sp1,1:15);
    sp1 = spmak(1:6:37,[1 1 1]);
    Nf = fnval(sp1,1:37);
    Nf = Nf./10;
    if ~isempty(size(C,5)) || size(C,5) < 1
        C = squeeze(C);
    end
    if numel(size(C)) < 4
        C = reshape(C,size(N));
    end
    
    cmat = (1 + tanh(C))./2;
    den_term = convnsep({Nf,Nf,Nf,Nf},cmat,'same');
    %den_term = den_term./ni;
    den_term(~mask) =0;
%     den_term(den_term < 0.005) = 0.005;
    
    term1 = R./den_term;
    term1(isnan(term1)) = 0;
    term1(isinf(term1)) = R(isinf(term1));
    term1(~mask) = 0;
    term1_smooth = convnsep({Nf./7,Nf./7,Nf./7,Nf./7},term1,'same');
    term1_smooth = term1_smooth./ni;
    term1_smooth(~mask) = 0;
    
    dNRmat = N - R;
    term2 = dNRmat./(1 - den_term +eps);
    term2(isnan(term2)) = 0;
    term2(isinf(term2)) = 0;
    term2(~mask) = 0;
    term2_smooth = convnsep({Nf./7,Nf./7,Nf./7,Nf./7},term2,'same');
    term2_smooth = term2_smooth./ni;
    term2_smooth(~mask) = 0;
    
    
    fun_value = log(den_term.^R .* (max(den_term(:))+eps - den_term).^dNRmat);
    fun_value(fun_value == -Inf) = eps;
    
    value = -sum(fun_value(:))/(91*109*91*10)
    
    grad_matrix1 = (term1_smooth - term2_smooth).*0.5.*((sech(C)).^2);
    grad_matrix1(~mask) = 0;
    grad_matrix = -grad_matrix1(:);
    norm_value = norm(grad_matrix);
    norm_value
    
    
% %     figure(66),imagesc(C(:,:,45,12));colorbar;
% %     fprintf('done');
% % %     Numerator1 = normalise(convnsep({Nf,Nf,Nf,Nf},R,'same'));
%     Denominator1 = normalise(convnsep({Nf,Nf,Nf,Nf},C,'same'));
%     dNRmat = N - R;
%     Numerator2 = normalise(convnsep({Nf,Nf,Nf,Nf},dNRmat,'same'));
%     Denominator2 = normalise(convnsep({Nf,Nf,Nf,Nf},C,'same') - 1);
%     value = (sum(Numerator1(:))/sum(Denominator1(:))) + (sum(Numerator2(:))/sum(Denominator2(:)));
end