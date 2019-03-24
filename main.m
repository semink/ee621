close all;
rng(8); % A fixed random seed
%% Expectation-Maximization algorithm
addpath("modules")
n = 300;
N_C = 4;  % Number of clusters
p = rand(N_C,1);
p = p/sum(p);

n_p = round(p*n);
m = 5;
mu_true = [0,0;m,0;m,m;0,m]; % Random mu
R_true = zeros(2,2,N_C);  % Random R (covariance)
y = [];     % Generate y
for k = 1 : N_C
    R_true(:,:,k) = random_cov(2);
    y = [y;mvnrnd(mu_true(k,:), R_true(:,:,k), n_p(k))];
end


% Choose the number of clusters
eva = evalclusters(y,@perform_em,'Silhouette','klist',[1:6]);
[score, gmm] = perform_em(y, N_C);
[score_optimal, gmm_optimal] = perform_em(y, eva.OptimalK);

%% Figures


figure,
plot(eva.CriterionValues);
ylabel("Silhouette value");
box on;
axis tight;
saveas(gcf, "figures/Silhouette.png");


figure,
hold on
x1 = -3:.2:12; x2 = -4:.2:10;
[X1,X2] = meshgrid(x1,x2);
start_idx = 1;

c = {'r','b','k','m','g'};
cn = cumsum(n_p);
for k = 1 : N_C
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
axis tight;
saveas(gcf, "figures/true.png");


figure,
hold on
c = {'r','b','k','m','g'};
for k = 1 : N_C
    mu= gmm.mu(k,:);
    sigma = gmm.R(:,:,k);
    F = mvnpdf([X1(:) X2(:)],mu, sigma);
    F = reshape(F,length(x2),length(x1));
    contour(x1,x2, F, c{k});

    scatter(y(gmm.cluster == k,1),y(gmm.cluster == k, 2), c{k},'filled');
end
hold off
box on;
axis tight;
saveas(gcf, "figures/gmm_result_fixed_K.png");



figure,
hold on
c = {'r','b','k','m','g'};
for k = 1 : eva.OptimalK
    mu= gmm_optimal.mu(k,:);
    sigma = gmm_optimal.R(:,:,k);
    F = mvnpdf([X1(:) X2(:)],mu, sigma);
    F = reshape(F,length(x2),length(x1));
    contour(x1,x2, F, c{k});
    scatter(y(gmm_optimal.cluster == k,1),y(gmm_optimal.cluster == k, 2), c{k},'filled');
end
hold off
box on;
axis tight;
saveas(gcf, "figures/gmm_result_optimal_K.png");
