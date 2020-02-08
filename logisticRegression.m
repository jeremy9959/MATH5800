function [Accuracy] = logisticRegression(Percentage,GradientIterations)
%INPUTS:
%Percentage = percentage of the data to train the model to
%GradientIterations = number of iterations the gradient search will take to
%%%find regression coefficients

%OUTPUTS:
%Accuracy = percentage of the test data that the regression correctly found
%%%%%

%Read Binary 0-1 Data and get sizes of each matrix
Data = csvread('train01.csv',0,0);
DataSize = size(Data,1);
TrainSize = round(Percentage*DataSize);
TestSize = DataSize - TrainSize;

%Preallocation
TestValues = zeros(TestSize,1);
Theta = zeros(784,1);

%Split Data into train set and test set
%Distinguish between data and labels
TrainLabels = Data(1:TrainSize,1);
TrainDigits = Data(1:TrainSize,2:785);
TestLabels = Data(TrainSize +1:DataSize,1);
TestDigits = Data(TrainSize+1:DataSize,2:785);

%Perform Gradient Search for Logistic Regression Coefficients
for i = 1:GradientIterations
    Train = 1./(1+exp(-TrainDigits*Theta));
    grad = (1/784)*(TrainDigits'*(Train-TrainLabels));
    Theta = Theta - 0.01*grad;
end

%Calculate Test Values
for i = 1:TestSize
    Input = TestDigits(i,:);
    TestValues(i) = 1/(1+ exp(-dot(Input,Theta)));
end

%Determine Accuracy
TestValues = round(TestValues);
Error = abs(TestLabels - TestValues);
Accuracy = 1 - (sum(Error)/size(Error,1));