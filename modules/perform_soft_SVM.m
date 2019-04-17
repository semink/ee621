function [w, theta] = perform_soft_SVM(H, gamma, mu, lo)
    H_offset = [ones(1,size(H,2));H];
    w_offset = zeros(size(H_offset,1),1);
    N = length(gamma);
    M = size(H,1);
    A = blkdiag(1,(1-2*mu*lo)*eye(M));
    for n = 1:N
        hn_offset = H_offset(:,n);
        gamma_hat = hn_offset'*w_offset;
        w_offset = A*w_offset + mu*gamma(n)*double(gamma(n)*gamma_hat<=1)*hn_offset;
    end
    theta = w_offset(1);
    w = w_offset(2:end);
end


