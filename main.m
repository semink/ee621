close all;
rng(3);
%% Douglas-Rachford algorithm
addpath("modules")
n = 200;
p = n/4;
A = randn(p,n);
y = randn(p,1);

q = @(x)norm(x,1);
Proxq = @(x, alpha)perform_soft_thresholding(x, alpha);

pA = A'*(A*A')^(-1);
ProxE = @(x)x - pA*(A*x - y);

options.report = @(x)struct('q', q(x));
options.alpha = 0.6;  % regularization parameter
options.mu = 0.001; % step size
options.niter = 4000;   % number of iteration 

% run algorithm
[x, R] = perform_dr(zeros(n,1), Proxq, ProxE, options);

Q = s2v(R,'q');


% Plot
figure;
semilogy(Q(2:end)-min(Q(2:end)), 'k', 'LineWidth', 2);
xlabel("iteration index n")
ylabel("q(w_n)-q(w_n^*)")
axis tight; grid on; title('Learning curve');


%% Subgradient learning algorithm


sP = @(w, alpha)(alpha*sign(w)+2*A'*(A*w-y)/length(y));
q = @(w)norm(w,1);
E = @(w)norm(y-A*w)^2/length(y);
P = @(w, alpha) alpha*q(w) + E(w);

% parameters
sub_options.report = @(w, alpha)struct('P', P(w, alpha));
sub_options.mu = 0.001;
sub_options.niter = 4000;
sub_options.alpha = 0.25;
sub_options.kappa = 0.3;

% run algorithm with smoothing
[w_smooth, R_smooth] = perform_subgradient(zeros(n,1), sP, sub_options);
p_smooth = s2v(R_smooth,'P');
sub_options.kappa = 0;

% run algorithm without smoothing
[w, R] = perform_subgradient(zeros(n,1), sP, sub_options);
p = s2v(R,'P');

% plot
figure,
semilogy(p-min(p_smooth), 'r--', 'LineWidth', 2)
hold on
semilogy(p_smooth-min(p_smooth), 'b', 'LineWidth',2)
axis tight; grid on;
title("Learning curve")
xlabel("iteration index n")
ylabel("P(w_n)-P(w_n^*)")
legend("wo smoothing", "with smoothing")
hold off

figure,
plot(w_smooth)
hold on
plot(w)
hold off