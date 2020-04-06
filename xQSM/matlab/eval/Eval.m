% Eval performs the dipole inversin on a local field map (V_Field)
% Recon = Eval(V_Field, 'xQSM', 'gpu'); returns xQSM reconstruction on the
% local field map V_Field with GPU.
% Recon = Eval(V_Field, 'Unet', 'gpu'); returns Unet reconstruction on the
% local field map V_Field with GPU.

function Recon = Eval(V_Field, NetType, ComputeEvn)
%EVAL Summary of this function goes here
imSize= size(V_Field);
net_temp = load(NetType);
net = net_temp.net;
newInput = image3dInputLayer(imSize,...
    'Name','ImageInputLayer', 'Normalization', 'none');
%% matlab neeed we manually replace the input layer
%% to fit the size of input, which is not necessary for pytorch codes;
L1Net = net;
L1Net = replaceLayer(layerGraph(L1Net),'ImageInputLayer',newInput);
L1Net = assembleNetwork(L1Net);
clear net;
%% prediction
%% for image size over 200* 300* 200, at least 32 GB memory is necessary;
predict(L1Net,  zeros(imSize), 'ExecutionEnvironment', ComputeEvn); % to pre-load the parameters into the memory;
tic, Recon = predict(L1Net,  V_Field, 'ExecutionEnvironment', ComputeEvn);toc
end
