
%% database for training; 
inputs = imageDatastore('../Field_VIVO/*.nii', ...
'FileExtensions','.nii','ReadFcn',@(x) niftiread(x));

labels = imageDatastore('../QSM_VIVO/*.nii', ...
'FileExtensions','.nii','ReadFcn',@(x) niftiread(x));

%% Check Data Length; 
disp('Data Length: ')
disp(length(labels.Files))
disp(length(inputs.Files))
 disp('Input Files');  % can check the file names; 
 inputs
 disp('Label Files')
 labels
%% combine the labels and inputs into one database. 
patchSize = [48, 48, 48];
patchPerImage = 1;
miniBatchSize = 30;
patchds = randomPatchExtractionDatastore(inputs,labels,patchSize, ...
    'PatchesPerImage',patchPerImage);
patchds.MiniBatchSize = miniBatchSize;
%% split the trianing data baset into trianing and validation datasets;  
[imdsTrain,imdsValidation] = splitEachLabel(patchds,0.05);
%% Create Network
disp('creating network')
[xQSM, info] = CreateXQSM([48,48,48,1]);
disp(xQSM.Layers) % check layers information. 

%% training optins 
initialLearningRate = 0.001;
maxEpochs = 100; % 
minibatchSize = miniBatchSize; % mini-batch size; 
l2reg = 1e-5;  % weight decay factor;

options = trainingOptions('adam',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',40,...
    'LearnRateDropFactor',0.1,...
    'L2Regularization',l2reg,...
    'MaxEpochs',maxEpochs,...
    'MiniBatchSize',minibatchSize,...
    'VerboseFrequency',20,...  
    'ValidationData', imdsValidation,...
    'ValidationFrequency', 300,...
    'Shuffle','every-epoch',...
    'ExecutionEnvironment','multi-gpu');

%% training
[net, info] = trainNetwork(imdsTrain, xQSM, options);
%% 
disp('save trainning results')
save xQSM.mat net; 
disp('saving complete!');



