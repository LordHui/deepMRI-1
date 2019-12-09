%% intialize my Unet
%% test this training file is the same as the one directely write the date on to ROM; 
clear
clc

load Mask_variable_ratio_4.mat; 

volReader1 = @(x) yangDataRead(x, Mask, 'k_full');
inputs = imageDatastore('./**/img_sub_*.mat', ...
'FileExtensions','.mat','ReadFcn',volReader1);

volReader2 = @(x) yangDataRead(x, Mask, 'k_full');
labels = imageDatastore('./**/img_full_*.mat', ...
'FileExtensions','.mat','ReadFcn',volReader2);

inputs
labels

NumFiles = length(inputs.Files);
disp(NumFiles)
%% 
patchSize = [48, 48, 48];
patchPerImage = 128;
miniBatchSize = 32;
patchds = randomPatchExtractionDatastore(inputs,labels,patchSize, ...
    'PatchesPerImage',patchPerImage);
patchds.MiniBatchSize = miniBatchSize;
%% 
disp('3D U-net 08_DEC,OLD - L2 loss - 4 EPO');
myUnet = create3DResNetBN([48,48,48,2]);
disp(myUnet.Layers)
% %% training set data;

%% training optins 
initialLearningRate = 0.001;
maxEpochs = 4
minibatchSize = miniBatchSize
l2reg = 0.00000;

options = trainingOptions('adam',...
    'L2Regularization',l2reg,...
    'MaxEpochs',maxEpochs,...
    'MiniBatchSize',minibatchSize,...
    'VerboseFrequency',20,...  
    'Shuffle','every-epoch',...
    'ExecutionEnvironment','gpu');
%% training function;
% parpool(2);
[net, info] = trainNetwork(patchds, myUnet, options);
%% 
disp('save trainning results')
save oriUnet_4EPO_MSE_4ACC_08_DEC_OLDLOAD.mat net; 
disp('saving complete!');



