function [mu, R, p, r] = perform_em(y, K, options)
N = size(y,1);
r = zeros(N,K);
M = zeros(K,1);

% initial
p = 1/K*ones(K,1);
mu = rand(K,2);
R = zeros(2,2,K);  % Random R (covariance)
for k = 1 : K
    R(:,:,k) = random_cov(2);
end

for m = 1 : options.niter
    % E-step
    for k = 1 : K
        for n = 1 : N
            s  = 0;
            for j = 1 : K
                s = s + p(j) * mvnpdf(y(n,:), mu(j,:), R(:,:,j));
            end
            r(n,k) = p(k)*mvnpdf(y(n,:), mu(k,:), R(:,:,k)) / s;
        end
        M(k) = sum(r(:,k));
    end
    
    % M-step
    for k = 1 : K
        mu(k,:) = 1/M(k) * r(:,k)'*y;
        R(:,:,k) = 1/M(k) *(y-ones(N,1)*mu(k,:))'*diag(r(:,k))*(y-ones(N,1)*mu(k,:));
        p = M/N;
    end
    
end
end

