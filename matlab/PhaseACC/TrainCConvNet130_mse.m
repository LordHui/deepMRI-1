%% intialize my Unet
clear
clc

load Mask_variable_ratio_4.mat; 

volReader1 = @(x) yangDataRead(x, Mask, 'img_sub');
% inputs = imageDatastore('../inputPatch48Mask/*.mat', ...
% 'FileExtensions','.mat','ReadFcn',volReader);
imgs = imageDatastore('./**/k_full_*.mat', ...
'FileExtensions','.mat','ReadFcn',volReader1);

volReader2 = @(x) yangDataRead(x, Mask, 'img_full');
ks = imageDatastore('./**/k_full_*.mat', ...
'FileExtensions','.mat','ReadFcn',volReader2);

NumFiles = length(imgs.Files);
disp(NumFiles)
%% 
patchSize = [48, 48, 48];
patchPerImage = 1;
miniBatchSize = 1;
patchds = randomPatchExtractionDatastore(ks,imgs,patchSize, ...
    'PatchesPerImage',patchPerImage);
patchds.MiniBatchSize = miniBatchSize;
%% 
disp('3D Complex Res-Unet 07 DEC, - L2 loss - 3 EPO');
[myUnet , info_net] = create3DCConvNet130BN([48,48,48,2]);
disp(myUnet.Layers)
% %% training set data;

%% training optins 
initialLearningRate = 0.001;
maxEpochs = 3;
minibatchSize = miniBatchSize;
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
[net, info] = trainNetwork(patchds, myUnet, options);
%% 
disp('save trainning results')
save CCUnet_3EPO_L2_test.mat net; 
disp('saving complete!');



