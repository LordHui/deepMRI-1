function data = yangDataRead(filename, Mask, mode)
% data load. including proprocessing to get the undersampling images. 
%% load K_full data; 
inp = load(filename);
f = fields(inp);
k_full = inp.(f{1}); %% normalised k_Full data;
%% subsample k-space for training. 
switch mode
    case 'k_full'
        data = k_full; 
    case 'k_sub'
        data = k_full .* Mask; 
    case 'img_full'
        data = fftn(fftshift(k_full));
        data = data ./ max(abs(data(:)));
        disp(max(abs(data(:))))
    case 'img_sub'
        k_sub = k_full .* Mask; 
        data = fftn(fftshift(k_sub));
        data = data ./ max(abs(data(:)));
        disp(max(abs(data(:))))
end
%% concatenate the complex data to be 2-chanell real tensors. 
ll = size(data);
ll = [ll 2];
temp = zeros(ll);
temp(:,:,:,1) = real(data);
temp(:,:,:,2) = imag(data);
data = temp; 
end