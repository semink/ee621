close all;
rng(1);
addpath("modules");
raw_data = readtable("dataset/wifi_localization.txt");
H = table2array(raw_data(:,1:end-1));
r = table2array(raw_data(:,end));
gamma = ones(size(r));
room = 4;
gamma(r ~= room) = -1;
dimension = 2;

reduced_H = perform_pca(H', dimension);


%% Learning
% 1. Perceptron
mu = 0.005;
[w_perceptron, theta_perceptron] = perform_perceptron(reduced_H, gamma, mu);
[w_logistic_regression, theta_logistic_regression] = perform_logistic_regression(reduced_H, gamma, mu);

cv = cvpartition(size(reduced_H,2),'HoldOut',0.3);
idx = cv.test;
% Separate to training and test data
dataTrain = reduced_H(:,~idx);
dataTest  = reduced_H(:,idx);
c_optimal = inf;
for lo = 0 : 0.0001 : 0.1
    [w, theta] = perform_soft_SVM(dataTrain, gamma(~idx), mu, lo);
    c = cost(dataTest,w,theta,gamma(idx));
    if c <= c_optimal
        lo_optimal = lo;
        c_optimal = c;
    end
end
[w_soft_SVM, theta_soft_SVM] = perform_soft_SVM(reduced_H, gamma, mu, lo_optimal);
%% Figures

figure,
hold on;
scatter(reduced_H(1,r==1),reduced_H(2,r==1), 'o', 'filled');
scatter(reduced_H(1,r==2),reduced_H(2,r==2), 's', 'filled');
scatter(reduced_H(1,r==3),reduced_H(2,r==3), 'd', 'filled');
scatter(reduced_H(1,r==4),reduced_H(2,r==4), '*');
hold off;
xlim([-6 4.5])
ylim([-4.5 4])
box on;
legend("Room 1", "Room 2", "Room 3", "Room 4", 'Location', 'southwest')
xlabel("First coordinate");
ylabel("Second coordinate");
saveas(gcf, "figures/pca.png");

figure,
hold on;
scatter(reduced_H(1,r~=room),reduced_H(2,r~=room), [], 'r','o');
scatter(reduced_H(1,r==room),reduced_H(2,r==room), [], 'b','o', 'filled');
hold off;
xlim([-6 4.5])
ylim([-4.5 4])
box on;
legend(['Not Room ' num2str(room)],['Room ' num2str(room)], 'Location', 'southwest')
xlabel('First coordinate');
ylabel('Second coordinate');
saveas(gcf, 'figures/pca_binary.png');

x = -6:0.1:4.5;
y_perceptron = -(w_perceptron(1)*x+theta_perceptron)/w_perceptron(2);
y_logistic_regression = -(w_logistic_regression(1)*x+theta_logistic_regression)/w_logistic_regression(2);
y_soft_SVM = -(w_soft_SVM(1)*x+theta_soft_SVM)/w_soft_SVM(2);

figure,
hold on;
scatter(reduced_H(1,r~=room),reduced_H(2,r~=room), [], 'r','o');
scatter(reduced_H(1,r==room),reduced_H(2,r==room), [], 'b','o', 'filled');

plot(x,y_perceptron,'k','LineWidth',2);
plot(x,y_logistic_regression,'b','LineWidth',2);
plot(x,y_soft_SVM,'g','LineWidth',2);
hold off;
xlim([-6 4.5])
ylim([-4.5 4])
box on;
legend(['Not Room ' num2str(room)],['Room ' num2str(room)], 'Perceptron','Logistic regression','soft SVM','Location', 'southwest')
xlabel('First coordinate');
ylabel('Second coordinate');
saveas(gcf, 'figures/classifications.png');


