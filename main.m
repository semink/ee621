close all;
rng(1); % random seed
addpath("modules");
d = load("dataset/mnist.mat");
d.testX = double(d.testX')/255;
d.trainX = double(d.trainX')/255;
gamma_train = one_hot_vector(d.trainY);

options.number_of_neurons = [size(d.testX,1), 400, 200, 10];
options.mu = 0.1;
options.rho = 0;
options.p = [0.4, 0.3, 0.3];

fileID = fopen('result.txt','w');
fprintf(fileID, 'Init_method, dropout, error_rate\n');
for dropout = [1, 0]
    for init_method = {'random','auto_encoder','dbm'}
        options.dropout = dropout;
        nn = neural_network(options);
        params.mu = 0.1;
        params.rho = 0;
        nn.initial_param(init_method{1}, params, d.trainX)
        nn.train(d.trainX, gamma_train);
        result = zeros(size(d.testY));
        for n = 1 : size(d.testX,2)
            result(n) = nn.test(d.testX(:,n));
        end
        e = sum((result - double(d.testY))~=0)/length(d.testY);
        clear nn
        fprintf(fileID, [init_method{1}, ', ']);
        fprintf(fileID, '%d, ', dropout);
        fprintf(fileID, '%f\n', e);
    end
end
fclose(fileID);
