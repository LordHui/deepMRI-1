function [CConvnet, info_net] = create3DCConvNet130BN(inputTileSize)
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here
addpath ./CCblocks3DBN_mse
%% common parameters 
info.convFilterSize = 3;
info.UpconvFilterSize = 2; 
info.numInputChannels = 64; 
info.numOutputChannels = 64; 
%% initialisation 
info.SectionName = 'ComplexResUnet'; 
info.portalH = '';
info.portalL = '';
%%
% [1, 64, 64, 64, 128, 128, 128, 256, 256, 256, 128, 128, 128];
%% input section; 
info_input = info; 
info_input.numInputChannels = 2; 
info_input.SectionName = 'Input';
[CConv_input, info_input] = CConvInput3D(info_input, inputTileSize);

CConvnet = CConv_input;
info_net = info_input;
%% encoding path: encoding depth: 2
encoding_depth = 2;
for i = 1 : encoding_depth
    %% CConvave conv block 1; 
    SectionName = ['EncoderSection', '-', num2str(i), '-CConv1']; 
    info.SectionName = SectionName; 
    info_temp = info; 
    info_temp.numInputChannels = info_net.numOutputChannels;
    info_temp.numOutputChannels = i * info_net.numOutputChannels;  
    [CConv_temp, info_temp] = CConv3D(info_temp);
    %% updata layers information;
    [CConvnet, info_net] = connectCConv3D(CConvnet,CConv_temp,info_net, info_temp);
    %% CConvave conv block 2
    SectionName = ['EncoderSection', '-', num2str(i), '-CConv2']; 
    info.SectionName = SectionName; 
    info_temp = info; 
    info_temp.numInputChannels = info_net.numOutputChannels;
    info_temp.numOutputChannels = info_net.numOutputChannels;  
    [CConv_temp, info_temp] = CConv3D(info_temp);
    %% updata layers information;
    [CConvnet, info_net] = connectCConv3D(CConvnet,CConv_temp,info_net, info_temp);    
    %% max pooling layers. 
    maxpool1 = ['EncoderSection', '-', num2str(i), '-MaxPooling1'];
    maxPoolLayer1 = maxPooling3dLayer(2, 'Stride', 2, 'Name',maxpool1);
    CConvnet = addLayers(CConvnet,maxPoolLayer1);
    
    maxpool2 = ['EncoderSection', '-', num2str(i), '-MaxPooling2'];
    maxPoolLayer2 = maxPooling3dLayer(2, 'Stride', 2, 'Name',maxpool2);
    CConvnet = addLayers(CConvnet,maxPoolLayer2);
    %% connect pooling layers.   
    CConvnet = connectLayers(CConvnet,info_net.portalR,maxpool1);
    CConvnet = connectLayers(CConvnet,info_net.portalI,maxpool2);
    %% update portals with maxpooling output; 
    info_net.portalR = maxpool1; 
    info_net.portalI = maxpool2;
end
%% mid layers
factors = [2, 1/2];
for i = 1 : 2
    %% construct CConvave convs.
    SectionName = ['MidSection', '-', num2str(i), '-CConv'];
    info.SectionName = SectionName;
    info_temp = info;
    info_temp.numInputChannels = info_net.numOutputChannels;
    info_temp.numOutputChannels = factors(i) * info_net.numOutputChannels;
    [CConv_temp, info_temp] = CConv3D(info_temp);
    %% updata layers information;
    [CConvnet, info_net] = connectCConv3D(CConvnet,CConv_temp,info_net, info_temp);    
end

%% expanding paths: encoding depth: 2
for i = 1 : encoding_depth
    %% first unpooling and contactation
    [CConvnet, info_net] = CConvUnpool3D(CConvnet, info_net, i, encoding_depth);
    %% CConvave conv block 1;
    SectionName = ['DecoderSection', '-', num2str(i), '-CConv1']; 
    info.SectionName = SectionName; 
    info_temp = info; 
    info_temp.numInputChannels = info_net.numOutputChannels;
    info_temp.numOutputChannels = info_net.numOutputChannels / 2;  
    [CConv_temp, info_temp] = CConv3D(info_temp);
    %% updata layers information;
    [CConvnet, info_net] = connectCConv3D(CConvnet,CConv_temp,info_net, info_temp);
    %% CConvave conv block 2;
    if i == encoding_depth
        SectionName = ['DecoderSection', '-', num2str(i), '-CConv2'];
        info.SectionName = SectionName;
        info_temp = info;
        info_temp.numInputChannels = info_net.numOutputChannels;
        info_temp.numOutputChannels = info_net.numOutputChannels;
        [CConv_temp, info_temp] = CConv3D(info_temp);
        %% updata layers information;
        [CConvnet, info_net] = connectCConv3D(CConvnet,CConv_temp,info_net, info_temp);
    else
        SectionName = ['DecoderSection', '-', num2str(i), '-CConv2'];
        info.SectionName = SectionName;
        info_temp = info;
        info_temp.numInputChannels = info_net.numOutputChannels;
        info_temp.numOutputChannels = info_net.numOutputChannels / 2;
        [CConv_temp, info_temp] = CConv3D(info_temp);
        %% updata layers information;
        [CConvnet, info_net] = connectCConv3D(CConvnet,CConv_temp,info_net, info_temp);
    end
end

%% output section: 
info_out = info; 
info_out.numInputChannels = info_net.numOutputChannels; 
info_out.numOutputChannels = 64; 
info_out.SectionName = 'Output';
[CConv_out, info_out] = CConvOutput3D(info_out);

[CConvnet, info_net] = connectCConvOut(CConvnet,CConv_out,info_net, info_out);
rmpath ./CCblocks3DBN_mse
end

