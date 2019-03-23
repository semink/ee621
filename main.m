close all;
rng(30); % A fixed random seed
%% Expectation-Maximization algorithm
addpath("modules")
n = 300;
K = 5;  % Number of clusters
p = rand(K,1);
p = p/sum(p);

n_p = round(p*n);

mu_true = 10*rand(K,2); % Random mu
R_true = zeros(2,2,K);  % Random R (covariance)
y = [];     % Generate y
for k = 1 : K
    R_true(:,:,k) = random_cov(2);
    y = [y;mvnrnd(mu_true(k,:), R_true(:,:,k), n_p(k))];
end


options.niter = 50;   % number of iteration 
[mu, R, p, r] = perform_em(y, K, options);


%%
figure,
hold on
x1 = -3:.2:17; x2 = -4:.2:15;
[X1,X2] = meshgrid(x1,x2);
start_idx = 1;

c = {'r','b','k','m','g'};
cn = cumsum(n_p);
for k = 1 : K
mu_t= mu_true(k,:);
Sigma = R_true(:,:,k);
F = mvnpdf([X1(:) X2(:)],mu_t, Sigma);
F = reshape(F,length(x2),length(x1));
contour(x1,x2,F, c{k});

scatter(y(start_idx:cn(k),1),y(start_idx:cn(k),2), c{k},'filled');
start_idx = cn(k)+1;
end
hold off
box on;


figure,
hold on
start_idx = 1;

c = {'r','b','k','m','g'};
cn = cumsum(n_p);
for k = 1 : K
mu_t= mu(k,:);
Sigma = R(:,:,k);
F = mvnpdf([X1(:) X2(:)],mu_t, Sigma);
F = reshape(F,length(x2),length(x1));
contour(x1,x2,F, c{k});

scatter(y(start_idx:cn(k),1),y(start_idx:cn(k),2), c{k},'filled');
start_idx = cn(k)+1;
end
hold off
box on;
