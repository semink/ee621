function c = cost(h,w,theta,gamma)
    h_offset = [ones(1,size(h,2)); h];
    gamma_hat = h_offset'*[theta;w];
    c = sum(double(gamma.*gamma_hat<=0));
end

