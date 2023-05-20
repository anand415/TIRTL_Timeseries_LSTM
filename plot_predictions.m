clear all;close all;
load 'predictions_300s_20_128_64_50_1.mat'
look_back=10;
trainPredictPlot = nan(1,length(dataset))';
trainPredictPlot(look_back:length(trainPredict)+look_back-1) = trainPredict;

valPredictPlot = nan(1,length(dataset));
valPredictPlot(length(trainPredict)+look_back*2:-1+length(trainPredict)+length(valPredict)+look_back*2) = valPredict;
testPredictPlot = nan(1,length(dataset));
testPredictPlot(length(dataset)-length(testPredict):-1+length(dataset)) = testPredict;
sqrt(immse(testY1,double(testPredict)'))
sqrt(immse(trainY1,double(trainPredict)'))

plot(dataset);
hold on;
plot(trainPredictPlot);
plot(valPredictPlot);
plot(testPredictPlot)