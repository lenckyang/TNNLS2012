function [TrainingTime, test_time,  TrainingAccuracy, TestingAccuracy] = D_ELM(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction,kkk)

% Usage: elm-MultiOutputRegression(TrainingData_File, TestingData_File, No_of_Output, NumberofHiddenNeurons, ActivationFunction)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm-MultiOutputRegression(TrainingData_File, TestingData_File, No_of_Output, NumberofHiddenNeurons, ActivationFunction)
%
% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set
% No_of_Output          - Number of outputs for regression
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%kkk                   -number of hidden nodes
%
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression

%
    %%%%    Authors:    Yimin Yang
    %%%%    Hunan University, Changsha
    %%%%    EMAIL:      yangyi_min@126.com
    %%%%    WEBSITE:    www.yiminyang.com
    %%%%    DATE:       APRIL 2012

%%%%%%%%%%% Load training dataset




%%%%%%%%%%% Load training dataset
train_data=TrainingData_File;
T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
test_data=TestingData_File;
TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                    %   Release raw testing data array

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);


%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
                                        %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;



D_YYM=[];
D_Input=[];
D_beta=[];
D_beta1=[];
TY=[];
FY=[];
BiasofHiddenNeurons1=[];

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
start_time_train=cputime;

for i=1:kkk
    %%%%%%%%%% B-ELM when number of hidden nodes L=2n-1 %%%%%%%
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
BiasofHiddenNeurons1=[BiasofHiddenNeurons1;BiasofHiddenNeurons];
tempH=P'*InputWeight';
YYM=pinv(P')*tempH;
YJX=P'*YYM;

 tempH=tempH';                                         %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
            %%%%%%%% More activation functions can be added here                
end
                                     %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
OutputWeight=pinv(H') * T';                        % slower implementation
% OutputWeight=inv(H * H') * H * T';                         % faster
% implementation
Y=(H' * OutputWeight)'; 


%%%%%%%%%% B-ELM when number of hidden nodes L=2n %%%%%%%
if i==1
    FY=Y;
else
FY=FY+Y;
end
E1=T-Y;
E1_2n_1(i)=norm(E1,2);
TrainingAccuracy2=sqrt(mse(E1));
Y2=E1'*pinv(OutputWeight);
Y2=Y2';
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        [Y22,PS(i)]=mapminmax(Y2,0.1,0.9);
    case {'sin','sine'}
        %%%%%%%% Sine
       [Y22,PS(i)]=mapminmax(Y2,0,1);
end

Y222=Y2;
Y2=Y22';

T1=(Y2* OutputWeight)';
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
Y3=1./Y2; 
Y3=Y3-1;
Y3=log(Y3);
Y3=-Y3';
    case {'sin','sine'}
        %%%%%%%% Sine
       Y3=asin(Y2)';
end

T2=(Y3'* OutputWeight)';







Y4=Y3;

YYM=pinv(P')*Y4';
YJX=P'*YYM;



BB1=size(Y4);
BB(i)=sum(YJX-Y4')/BB1(2);



GXZ1=P'*YYM-BB(i);

cc=pinv(P')*(GXZ1-Y4');
Y5=P'*cc-(GXZ1-Y4');
GXZ11=P'*(YYM-cc)-BB(i);
BBB(i)=mean(GXZ11-Y4');
GXZ111=P'*(YYM-cc)-BB(i)-BBB(i);
BBBB(i)=BB(i)+BBB(i);
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
GXZ2=1./(1+exp(-GXZ111'));
    case {'sin','sine'}
        %%%%%%%% Sine
GXZ2=sin(GXZ111');
end


FYY = mapminmax('reverse',GXZ2,PS(i));

%FYY=GXZ2;
OutputWeight1=pinv(FYY') * E1'; 
FT1=FYY'*OutputWeight1;
FY=FY+FT1';
TrainingAccuracy=sqrt(mse(FT1'-E1));
D_Input=[D_Input;InputWeight];
D_beta=[D_beta;OutputWeight];
D_beta1=[D_beta1;OutputWeight1];
D_YYM=[D_YYM;(YYM-cc)'];
T=FT1'-E1;
E1_2n(i)=norm(T,2);

end
end_time_train=cputime;
TrainingTime=end_time_train-start_time_train;







    




start_time_test=cputime;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%test%%%%%%%%%%%%%%%%%%%%%

tempH_test=D_Input*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons1(:,ind);              
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
            %%%%%%%% More activation functions can be added here        
end
TY1=(H_test' *D_beta)';                       %   TY: the actual output of the testing data
E1=TV.T - TY1;
TY=TY1;
for i=1:kkk
GXZ1=D_YYM(i,:)*TV.P-BBBB(i);
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
GXZ2=1./(1+exp(-GXZ1'));
    case {'sin','sine'}
        %%%%%%%% Sine
GXZ2=sin(GXZ1');
end

FYY = mapminmax('reverse',GXZ2',PS(i));
%FYY=GXZ2;
TY2=FYY'*D_beta1(i,:);
TestingAccuracy=sqrt(mse(TY2'-E1));
E1=TY2'-E1;
TY=TY+TY2';

end


end_time_test=cputime;
 test_time=end_time_test-start_time_test;
end
