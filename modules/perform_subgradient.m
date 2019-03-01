function [wn, R] = perform_subgradient(initial_w, sP, options)
mu = getoptions(options, 'mu', 0.1);
niter = getoptions(options, 'niter', 100);
report = getoptions(options, 'report', @(x, alpha)0);
kappa = getoptions(options, 'kappa', 0);    % default = no smoothing
alpha = getoptions(options, 'alpha', 1);
w_pre = initial_w;
wn = w_pre;
S = 0;

clear R;
for i = 1:niter
    w = w_pre - mu*sP(w_pre, alpha);
    S = kappa * S + 1;
    wn = (1 - 1/S)*wn + 1/S * w;
    w_pre = w;
    R(i) = report(wn, alpha);
end

end
