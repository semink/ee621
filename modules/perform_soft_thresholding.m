function y = perform_soft_thresholding(x, alpha)
    y = zeros(size(x));
    y(x >= alpha) = x(x >= alpha) - alpha;
    y(x <= -alpha) = x(x <= -alpha) + alpha;
end