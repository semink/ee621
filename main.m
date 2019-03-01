n = 200;
p = n/4;
A = randn(p,n);
y = randn(p,1);

ProxF = @(x, alpha)perform_soft_thresholding(x, alpha);

pA = A'*(A*A')^(-1);
ProxG = @(x)x + pA*(y-A*x);

F = @(x)norm(x,1);
Constr = @(x)1/2*norm(y-A*x)^2;

options.report = @(x)struct('F', F(x), 'Constr', Constr(x));
options.alpha = 1;  % regularization parameter
options.mu = 0.001; % step size
options.niter = 2000;   % number of iteration 

[x, R] = perform_dr(zeros(n,1), ProxF, ProxG, options);

clf;
plot(x);
axis tight;

f = s2v(R,'F');
constr = s2v(R,'Constr');
clf;
subplot(2,1,1);
plot(f(2:end));
axis tight; title('Objective');
subplot(2,1,2);
plot(constr(2:end));
axis tight; title('Constraint');