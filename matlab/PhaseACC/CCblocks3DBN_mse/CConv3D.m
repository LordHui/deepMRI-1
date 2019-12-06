function [lgraph, info] = CConv3D(info)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
lgraph = layerGraph();
%% parameters definition. 
convFilterSize = info.convFilterSize;
numInputChannels = info.numInputChannels; 
numOutputChannels = info.numOutputChannels; 
SectionName = info.SectionName; 

f = 0.01;
XR_channels = numInputChannels  / 2;
XI_channels = numInputChannels  / 2; 

YR_channels = numOutputChannels / 2;
YI_channels = numOutputChannels / 2; 

%% traditional convolution blocks. 
%% number of parameterrs: numInputChannels * convFilterSize^3 * numOutputChannes.
%% complex Convolutional kernels have the same paramter size with the traditional ones.
%% number of parameters (Alpha = 0.5): numInputChannels/2 * convFilterSize^3 * numOutputChannes/2 * 4;
%% Conv RR; 
conv_RR = strcat(SectionName, '-Conv-RR');

convRR = convolution3dLayer(convFilterSize,YR_channels,...
    'Padding', 'same',...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...    
    'Name',conv_RR);

convRR.Weights = f  * randn(convFilterSize,convFilterSize, convFilterSize,XR_channels,YR_channels);
convRR.Bias = zeros(1,1,1,YR_channels);

lgraph = addLayers(lgraph,convRR);
%%
BN_RR = strcat(SectionName, '-BN_RR');
BNRR = batchNormalizationLayer('Name',BN_RR);

lgraph = addLayers(lgraph,BNRR);
%% conv RI
conv_RI = strcat(SectionName, '-Conv-RI');

convRI = convolution3dLayer(convFilterSize,YI_channels,...
    'Padding', 'same',...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...    
    'Name',conv_RI);

convRI.Weights = f  * randn(convFilterSize,convFilterSize, convFilterSize,XR_channels,YI_channels);
convRI.Bias = zeros(1,1,1,YI_channels);

lgraph = addLayers(lgraph,convRI);
%%
BN_RI = strcat(SectionName, '-BN_RI');
BNRI = batchNormalizationLayer('Name',BN_RI);

lgraph = addLayers(lgraph,BNRI);
%% conv II
conv_II = strcat(SectionName, '-Conv-II');

convII = convolution3dLayer(convFilterSize,YI_channels,...
    'Padding', 'same',...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...    
    'Name',conv_II);

convII.Weights = f  * randn(convFilterSize,convFilterSize, convFilterSize,XI_channels,YI_channels);
convII.Bias = zeros(1,1,1,YI_channels);

lgraph = addLayers(lgraph,convII);
%%
BN_II = strcat(SectionName, '-BN_II');
BNII = batchNormalizationLayer('Name',BN_II);

lgraph = addLayers(lgraph,BNII);
%% conv IR
conv_IR = strcat(SectionName, '-Conv-IR');

convIR = convolution3dLayer(convFilterSize,YR_channels,...
    'Padding', 'same',...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...    
    'Name',conv_IR);

convIR.Weights = f  * randn(convFilterSize,convFilterSize, convFilterSize,XI_channels,YR_channels);
convIR.Bias = zeros(1,1,1,YR_channels);

lgraph = addLayers(lgraph,convIR);
%%
BN_IR = strcat(SectionName, '-BN_IR');
BNIR = batchNormalizationLayer('Name',BN_IR);

lgraph = addLayers(lgraph,BNIR);
%% addition layer
addR_name = strcat(SectionName, '-AddR');
addR = additionLayer(2,'Name',addR_name);
lgraph = addLayers(lgraph,addR);

addI_name = strcat(SectionName, '-AddI');
addI = additionLayer(2,'Name',addI_name);
lgraph = addLayers(lgraph,addI);
%% activation layers
reluR_name = strcat(SectionName, '-ReLU_R');
ReluR = reluLayer('Name',reluR_name);
lgraph = addLayers(lgraph,ReluR);

reluI_name = strcat(SectionName, '-ReLU_I');
ReluI = reluLayer('Name',reluI_name);
lgraph = addLayers(lgraph,ReluI);
%% Addition layer
addR1 = strcat(addR_name, '/in1');
addR2 = strcat(addR_name, '/in2');

addI1 = strcat(addI_name, '/in1');
addI2 = strcat(addI_name, '/in2');
%% Connect Layers
%% ConvRR ---> BN_RR, ConvII ----> BN_II
lgraph = connectLayers(lgraph,conv_RR,BN_RR);
lgraph = connectLayers(lgraph,BN_RR,addR1);
lgraph = connectLayers(lgraph,conv_II,BN_II);
lgraph = connectLayers(lgraph,BN_II,addI1);
%% ConvRI ----> BN_RI, ConvIR ----> BN_IR
lgraph = connectLayers(lgraph,conv_RI,BN_RI);
lgraph = connectLayers(lgraph,BN_RI,addI2);
lgraph = connectLayers(lgraph,conv_IR,BN_IR);
lgraph = connectLayers(lgraph,BN_IR,addR2);
%% BN_RR, BN_IR ---> addR, BN_II, BN_RI ---> addI
lgraph = connectLayers(lgraph,addR_name,reluR_name);
lgraph = connectLayers(lgraph,addI_name,reluI_name);
%% updata portal information;
info.portalR = reluR_name;
info.portalI = reluI_name;
end

