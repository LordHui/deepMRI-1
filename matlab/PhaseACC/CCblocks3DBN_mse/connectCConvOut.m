function [lgraph, info] = connectCConvOut(CConv1,CConv2,info1, info2)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
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
%% 
info = info2;
info.portalR = info2.portalR;
info.portalI = info2.portalI;
%% create Residual block:
add_finalR = additionLayer(2,'Name','add_finalR');
lgraph = addLayers(lgraph, add_finalR);
lgraph = connectLayers(lgraph, 'ImageInputLayerR','add_finalR/in1');
lgraph = connectLayers(lgraph, 'Final-ConvolutionLayerR','add_finalR/in2');

add_finalI = additionLayer(2,'Name','add_finalI');
lgraph = addLayers(lgraph, add_finalI);
lgraph = connectLayers(lgraph, 'ImageInputLayerI','add_finalI/in1');
lgraph = connectLayers(lgraph, 'Final-ConvolutionLayerI','add_finalI/in2');
%% concatenation layer 
concat_final = 'DepthConcatenation_Final';
depthConcatLayer_final = concatenationLayer(4, 2,'Name',concat_final);
lgraph = addLayers(lgraph,depthConcatLayer_final);

temp_str1 = strcat(concat_final, '/in1');
temp_str2 = strcat(concat_final, '/in2');

lgraph = connectLayers(lgraph, 'add_finalR',temp_str1);
lgraph = connectLayers(lgraph, 'add_finalI',temp_str2);
%% regression layer

regLayer = regressionLayer('Name', 'Output_Layer');
lgraph = addLayers(lgraph,regLayer);
lgraph = connectLayers(lgraph,concat_final,'Output_Layer');

end

