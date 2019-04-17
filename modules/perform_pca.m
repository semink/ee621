function h_hat = perform_pca(h_by_col, dimension)
    h_mu = mean(h_by_col,2);
    N = size(h_by_col, 2);
    h_center = h_by_col - h_mu*ones(1,N);
    v = N/(N-1)*mean(h_center.^2,2);
    S = diag(sqrt(v));
    h_normalized =inv(S)* h_center;
    R = 1/(N-1)*h_normalized*h_normalized';    
    [U,D] = eigs(R);
    h_hat = U(:,1:dimension)'*h_normalized;
end
