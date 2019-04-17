function [w, theta] = perform_logistic_regression(H, gamma, mu)
    H_offset = [ones(1,size(H,2));H];
    w_offset = zeros(size(H_offset,1),1);
    N = length(gamma);
    for n = 1:N
        hn_offset = H_offset(:,n);
        gamma_hat = hn_offset'*w_offset;
        w_offset = w_offset + mu*gamma(n)*hn_offset/(1+exp(gamma(n)*gamma_hat));
    end
    theta = w_offset(1);
    w = w_offset(2:end);
end

