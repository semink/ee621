function y = Bernoulli(p, n)
    y = double(rand(n,1) >= p);
end
