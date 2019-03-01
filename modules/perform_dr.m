function [w, R] = perform_dr(initial_z, Proxq, ProxE, options)
z = initial_z;

alpha = getoptions(options, 'alpha', 1);
mu = getoptions(options, 'mu', 0.1);
niter = getoptions(options, 'niter', 100);
report = getoptions(options, 'report', @(x)0);

clear R;

for i = 1:options.niter
    w = Proxq(z, mu*alpha);
    t = ProxE(2*w-z);
    z = t - w + z;
    R(i) = report(z);
end
end