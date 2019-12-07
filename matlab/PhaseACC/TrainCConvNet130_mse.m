%% intialize my Unet
clear
clc

volReader2 = @(x) matRead2(x);
% inputs = imageDatastore('../inputPatch48Mask/*.mat', ...
% 'FileExtensions','.mat','ReadFcn',volReader);
imgs = imageDatastore('./img_labels/*.mat', ...
'FileExtensions','.mat','ReadFcn',volReader2);

volReader = @(x) matRead(x);
ks = imageDatastore('./k_labels/*.mat', ...
'FileExtensions','.mat','ReadFcn',volReader2);

NumFiles = length(imgs.Files);
%% preprocessing: subsampling; 
% load('Mask_variable_ratio_4');
% k_subs = transform(ks, @(x) Subsampling(x, Mask));
% imgs_subs = transform(k_subs, @(x) fftn(fftshift(x)));
% %% preprocessing: concatenation
% imgs_subs = transform(imgs_subs, @(x) Concat(x));
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



