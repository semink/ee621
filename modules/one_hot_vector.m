function g = one_hot_vector(y)
    g = zeros(max(y)+1,length(y));
    for i = 1 : length(y)
        g(y(i)+1,i) = 1;
    end
end

