function [CConvnet, info] = CConvUnpool3D(CConvnet, info, sections, encoding_depth)
%Summary of this function goes here
%   Detailed explanation goes here
f = 0.01;

UpconvFilterSize = info.UpconvFilterSize;
numOutputChannels = info.numOutputChannels; 

YR_channels = numOutputChannels / 2;
YI_channels = numOutputChannels / 2; 

%% R channels unpooling 
upconvRR = ['Decoder-Section-' num2str(sections) '-UpConvRR'];
upConv_RR = transposedConv3dLayer(UpconvFilterSize, YR_channels,...
    'Stride',2,...
    'Cropping', [0 0 0; 0 0 0],...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...    
    'Name',upconvRR);

upConv_RR.Weights = f * randn(UpconvFilterSize,UpconvFilterSize, UpconvFilterSize ,YR_channels,YR_channels);
upConv_RR.Bias = zeros(1,1,1, YR_channels);

CConvnet = addLayers(CConvnet,upConv_RR);


upconvIR = ['Decoder-Section-' num2str(sections) '-UpConvIR'];
upConv_IR = transposedConv3dLayer(UpconvFilterSize, YR_channels,...
    'Stride',2,...
    'Cropping', [0 0 0; 0 0 0],...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...    
    'Name',upconvIR);

upConv_IR.Weights = f * randn(UpconvFilterSize, UpconvFilterSize, UpconvFilterSize ,YI_channels,YR_channels);
upConv_IR.Bias = zeros(1,1,1,YR_channels);

CConvnet = addLayers(CConvnet,upConv_IR);
%% Addition
addR_name = ['Decoder-Section-' num2str(sections) '-addR'];
addR = additionLayer(2,'Name',addR_name);
CConvnet = addLayers(CConvnet,addR);
%% BN
BN_R = ['Decoder-Section-' num2str(sections) '-BN_R'];
BNR = batchNormalizationLayer('Name',BN_R);

CConvnet = addLayers(CConvnet,BNR);
%% RELU
upreluR = ['Decoder-Section-' num2str(sections) '-UpReLUR'];
upReLU_R = reluLayer('Name',upreluR);

CConvnet = addLayers(CConvnet,upReLU_R);

%% I channels
upconvII = ['Decoder-Section-' num2str(sections) '-UpConvII'];
upConv_II = transposedConv3dLayer(UpconvFilterSize, YI_channels,...
    'Stride',2,...
    'Cropping', [0 0 0; 0 0 0],...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...    
    'Name',upconvII);

upConv_II.Weights = f * randn(UpconvFilterSize,UpconvFilterSize, UpconvFilterSize ,YI_channels,YI_channels);
upConv_II.Bias = zeros(1,1,1, YI_channels);

CConvnet = addLayers(CConvnet,upConv_II);
%% 
upconvRI = ['Decoder-Section-' num2str(sections) '-UpConvRI'];
upConv_RI = transposedConv3dLayer(UpconvFilterSize, YI_channels,...
    'Stride',2,...
    'Cropping', [0 0 0; 0 0 0],...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...    
    'Name',upconvRI);

upConv_RI.Weights = f * randn(UpconvFilterSize,UpconvFilterSize, UpconvFilterSize ,YR_channels,YI_channels);
upConv_RI.Bias = zeros(1,1,1, YI_channels);

CConvnet = addLayers(CConvnet,upConv_RI);
%% addition
addI_name = ['Decoder-Section-' num2str(sections) '-addI'];
addI = additionLayer(2,'Name',addI_name);
CConvnet = addLayers(CConvnet,addI);
%% BN
BN_I = ['Decoder-Section-' num2str(sections) '-BN_I'];
BNI = batchNormalizationLayer('Name',BN_I);

CConvnet = addLayers(CConvnet,BNI);
%% Relu
upreluI = ['Decoder-Section-' num2str(sections) '-UpReLUI'];
upReLU_I = reluLayer('Name',upreluI);
CConvnet = addLayers(CConvnet,upReLU_I);

%% connect layers.
CConvnet = connectLayers(CConvnet,info.portalR,upconvRR);
CConvnet = connectLayers(CConvnet,info.portalR,upconvRI);
CConvnet = connectLayers(CConvnet,info.portalI,upconvII);
CConvnet = connectLayers(CConvnet,info.portalI,upconvIR);

addR1 = strcat(addR_name, '/in1');
addR2 = strcat(addR_name, '/in2');

addI1 = strcat(addI_name, '/in1');
addI2 = strcat(addI_name, '/in2');

CConvnet = connectLayers(CConvnet,upconvRR,addR1);
CConvnet = connectLayers(CConvnet,upconvIR,addR2);
CConvnet = connectLayers(CConvnet,upconvII,addI1);
CConvnet = connectLayers(CConvnet,upconvRI,addI2);

CConvnet = connectLayers(CConvnet,addR_name,BN_R);
CConvnet = connectLayers(CConvnet,BN_R,upreluR);
CConvnet = connectLayers(CConvnet,addI_name,BN_I);
CConvnet = connectLayers(CConvnet,BN_I,upreluI);

info.portalH = upreluR;
info.portalL = upreluI;
%% concatenation due to U-net backbone. 
concatR = ['Decoder-Section-' num2str(sections) '-DepthConcatenationR'];
depthConcatLayerR = concatenationLayer(4, 2,'Name',concatR);
CConvnet = addLayers(CConvnet,depthConcatLayerR);

concatI = ['Decoder-Section-' num2str(sections) '-DepthConcatenationI'];
depthConcatLayerI = concatenationLayer(4, 2,'Name',concatI);
CConvnet = addLayers(CConvnet,depthConcatLayerI);

%% concatenate layers:
temp = encoding_depth + 1 - sections; 
temp = num2str(temp);
%% always concatenate the second convolution part of encoding blocks to decoding blocks.
temp_portalR = ['EncoderSection-',temp,'-CConv2-ReLU_R'];
temp_portalI = ['EncoderSection-',temp,'-CConv2-ReLU_I'];

temp_strR1 = strcat(concatR, '/in1');
temp_strR2 = strcat(concatR, '/in2');

temp_strI1 = strcat(concatI, '/in1');
temp_strI2 = strcat(concatI, '/in2');

CConvnet = connectLayers(CConvnet, upreluR,temp_strR1);
CConvnet = connectLayers(CConvnet, temp_portalR,temp_strR2);

CConvnet = connectLayers(CConvnet, upreluI,temp_strI1);
CConvnet = connectLayers(CConvnet, temp_portalI,temp_strI2);
%%  updata portal informaiton;
info.numOutputChannels = 4 * YR_channels; % concatenation results. 

info.portalR = concatR;
info.portalI = concatI;
end

