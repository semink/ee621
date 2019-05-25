classdef neural_network < handle
    properties
        options
        weights
        offset
    end
    methods
        function obj = neural_network(options)
            obj.options = options;
            L = length(obj.options.number_of_neurons);
            obj.weights = cell(L-1,1);
            obj.offset = cell(L-1,1);
            for l = 1:L-1
                    obj.weights{l} = 1/sqrt(obj.options.number_of_neurons(l))*randn(obj.options.number_of_neurons(l+1), obj.options.number_of_neurons(l));
                    obj.offset{l} = randn(obj.options.number_of_neurons(l+1),1);
            end
            
        end
        function initial_param(obj, method, params, trainX)
            L = length(obj.options.number_of_neurons);
            if strcmp(method, 'auto_encoder')
                Y = trainX;
                for l = 1:L-1
                    options_ae.mu = params.mu;
                    options_ae.rho = params.rho;
                    options_ae.dropout = 0;
                    options_ae.init_method = 'random';
                    options_ae.number_of_neurons = [obj.options.number_of_neurons(l), obj.options.number_of_neurons(l+1), obj.options.number_of_neurons(l)];
                    nn = neural_network(options_ae);
                    nn.train(Y, Y);
                    obj.weights{l} = nn.weights{1};
                    obj.offset{l} = nn.offset{1};
                    Y = obj.activate(nn.weights{1}*Y-nn.offset{1}*ones(1,size(trainX,2)));
                 end
            elseif strcmp(method, 'dbm')
                Y = trainX;
                mu = params.mu;
                W = obj.weights;
                offset = obj.offset;
                offset_r = cell(L-1,1);
                y = cell(L-1,1);
                yb = cell(L-1,1);
                h = cell(L-1,1);
                yp = cell(L-1,1);
                ybp = cell(L-1,1);
                
                for l = 1:L-1
                    offset_r{l} = randn(obj.options.number_of_neurons(l),1);
                end
                for n = 1:size(trainX,2)
                    hl = trainX(:,n);
                    for l = 1:L-1
                        y{l} = obj.activate(W{l}*hl-offset{l});
                        yb{l} = double(rand(size(y{l}))<=y{l});
                        h{l} = obj.activate(W{l}'*yb{l}-offset_r{l});
                        yp{l} = obj.activate(W{l}*h{l}-offset{l});
                        ybp{l} = double(rand(size(yp{l}))<=yp{l});
                        W{l} = W{l} + mu*(y{l}*hl'-yp{l}*h{l}');
                        offset{l} = offset{l} + mu*(yp{l} - y{l});
                        offset_r{l} = offset_r{l} + mu*(h{l} - hl);
                        hl = y{l};
                    end
                end
                for l = 1:L-1
                    obj.weights = W;
                    obj.offset = offset;
                end
            end

        end
        function obj = set.weights(obj, W)
            obj.weights = W;
        end
        function obj = set.offset(obj, offset)
            obj.offset = offset;
        end
        function train(obj, trainX, trainY)
            L = length(obj.options.number_of_neurons);
            y = cell(L, 1);
            z = cell(L, 1);
            delta = cell(L, 1);
            W = obj.weights;
            offset = obj.offset;
            mu = obj.options.mu;
            rho = obj.options.rho;
            for n = 1 : size(trainX,2)
                hn = trainX(:,n);
                y{1} = hn;
                % Feed forward
                a = cell(L-1,1);
                for l = 1:L-1
                    if(obj.options.dropout)
                        a{l} = Bernoulli(obj.options.p(l), obj.options.number_of_neurons(l));
                    else
                        a{l} = ones(obj.options.number_of_neurons(l),1);
                    end
                    z{l+1} = W{l}*(y{l}.*a{l})- offset{l};
                    y{l+1} = obj.activate(z{l+1});
                end
                gn = trainY(:,n); 
                delta{L} = y{L} - gn;
                % Back propagation
                for l = L-1:-1:1
                    if l >=2
                        delta{l} = obj.dactivate(z{l}).*(W{l}'*delta{l+1}).*a{l};
                    end
                    W{l} = (1-2*mu*rho)*W{l} - mu*delta{l+1}*(y{l}.*a{l})';
                    offset{l} = offset{l} + mu*delta{l+1};
                end
            end
            obj.weights = W;
            obj.offset = offset;
        end
        function g = test(obj, x)
            L = length(obj.options.number_of_neurons);
            W = obj.weights;
            offset = obj.offset;
            
            y = x;
            % Feed forward
            for l = 1:L-1
                z = W{l}*y-offset{l};
                y = obj.activate(z);
            end
            [~,g] = max(y);
            g = g - 1;
        end
        function y = activate(obj, x)
            y = 1./(1+exp(-x));
        end
        function y = dactivate(obj, x)
            y = obj.activate(x).*(1-obj.activate(x));
        end
        
    end
end
