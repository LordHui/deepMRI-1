function [lgraph, info] = connectCConv3D(CConv1,CConv2,info1, info2)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%% CConv.layers: the network layers.
%% CConv.connections: the network connection lines. 
lgraph = CConv1;
%% get the layers of the second block. 
tempLayers = CConv2.Layers;
for i = 1 : 1 : length(tempLayers)
templ = tempLayers(i);
lgraph = addLayers(lgraph, templ);
end
%% section1 RELU ----> convs of Section2. 
Section1_portalR = info1.portalR; 
Section1_portalI = info1.portalI;

SectionName2 = info2.SectionName; 
Section2_portalRR = strcat(SectionName2, '-Conv-RR');
Section2_portalII = strcat(SectionName2, '-Conv-II');
Section2_portalIR = strcat(SectionName2, '-Conv-IR');
Section2_portalRI = strcat(SectionName2, '-Conv-RI');

lgraph = connectLayers(lgraph,Section1_portalR,Section2_portalRR);
lgraph = connectLayers(lgraph,Section1_portalR,Section2_portalRI);
lgraph = connectLayers(lgraph,Section1_portalI,Section2_portalII);
lgraph = connectLayers(lgraph,Section1_portalI,Section2_portalIR);

%% connect the second part. 
tempConns = CConv2.Connections; % get the original connection lines.
%% connect the second part as it was. 
tempConns = table2array(tempConns);
for i = 1 : 1 : length(tempConns)
    lgraph = connectLayers(lgraph,tempConns{i,1},tempConns{i,2});
end

%% updata portal information;
info = info2;
info.portalR = info2.portalR;
info.portalI = info2.portalI;
end

