%% intialize my Unet
disp('3D ResNet June 19 - L1 loss - VIVO Dataset');
myUnet = create3DResNetBN([48,48,48,1]);
disp(myUnet.Layers)
% %% training set data;

%% training optins 
initialLearningRate = 0.001;
maxEpochs = 40;
minibatchSize = 32;
l2reg = 0.00000;

options = trainingOptions('adam',...
    'L2Regularization',l2reg,...
    'MaxEpochs',maxEpochs,...
    'MiniBatchSize',minibatchSize,...
    'VerboseFrequency',20,...  
    'Shuffle','every-epoch',...
    'ExecutionEnvironment','multi-gpu');
%% training function;
% parpool(2);
[net, info] = trainNetwork(TrainSet, TrainLabel, myUnet, options);
