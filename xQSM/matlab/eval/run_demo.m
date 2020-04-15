%% navigate MATLAB to the 'eval' folder
% cd('~/matlab/deepMRI/xQSM/eval');


%% add NIFTI matlab toolbox for read and write NIFTI format
addpath ./NIFTI 


%% read in field map and COSMOS map (3rd dimension is z/B0 direction)
nii = load_nii('../../field_input.nii'); % replace the file name with yours. 
field = double(nii.img);
mask = field ~= 0; % brain tissue mask

% note the size of the field map input needs to be divisibel by 8
% otherwise 0 padding should be done first
imSize = size(field);
if mod(imSize, 8)
    [field, pos] = ZeroPadding(field, 8);
end
% illustration of one central axial slice of the input field 
figure, imagesc(field(:,:,80)'); colormap gray; axis equal tight; colorbar; caxis([-0.05, 0.05])
title('Slice 80 of the Input Field Map (ppm)');


%% read label (for evaluation purpose)
nii = load_nii('../../cosmos_label.nii'); % replace the file name with yours. 
label = double(nii.img);

% illustration of one central axial slice of the COSMOS label 
figure, imagesc(label(:,:,80)'); colormap gray; axis equal tight; colorbar; caxis([-0.1, 0.2])
title('Slice 80 of the COSMOS Label (ppm)');


%% start recons
if canUseGPU()
    % (1) if your MATLAB is configured with CUDA GPU acceleration
    Unet_invivo_recon = Eval(field, 'Unet_invivo', 'gpu');
    xQSM_invivo_recon = Eval(field, 'xQSM_invivo', 'gpu');
    Unet_syn_recon = Eval(field, 'Unet_syn', 'gpu'); 
    xQSM_syn_recon = Eval(field, 'xQSM_syn', 'gpu');
else
    % (2) otherwise if CUDA is not available, use CPU instead, this is much slower
    Unet_invivo_recon = Eval(field, 'Unet_invivo', 'cpu');
    xQSM_invivo_recon = Eval(field, 'xQSM_invivo', 'cpu');
    Unet_syn_recon = Eval(field, 'Unet_syn', 'cpu'); 
    xQSM_syn_recon = Eval(field, 'xQSM_syn', 'cpu');
end

% if zeropadding was performed, then do zero-removing before next step;
if mod(imSize, 8)
    xQSM_invivo_recon = ZeroRemoving(xQSM_invivo_recon, pos);
    Unet_invivo_recon = ZeroRemoving(Unet_invivo_recon, pos);
    xQSM_syn_recon = ZeroRemoving(xQSM_syn_recon, pos);
    Unet_syn_recon = ZeroRemoving(Unet_syn_recon, pos);
end


%% image normalization (mean of brain tissue region set to 0)
label = label - sum(label(:)) / sum(mask(:));
label = label .* mask; 

xQSM_invivo_recon = xQSM_invivo_recon - sum(xQSM_invivo_recon(:)) / sum(mask(:));
xQSM_invivo_recon = xQSM_invivo_recon .* mask; 

Unet_invivo_recon = Unet_invivo_recon - sum(Unet_invivo_recon(:)) / sum(mask(:));
Unet_invivo_recon = Unet_invivo_recon .* mask; 

xQSM_syn_recon = xQSM_syn_recon - sum(xQSM_syn_recon(:)) / sum(mask(:));
xQSM_syn_recon = xQSM_syn_recon .* mask; 

Unet_syn_recon = Unet_syn_recon - sum(Unet_syn_recon(:)) / sum(mask(:));
Unet_syn_recon = Unet_syn_recon .* mask; 


%% illustration of one central axial slice of the four different reconstructions; 
figure,
subplot(121), imagesc(xQSM_invivo_recon(:,:,80)'); colormap gray; axis equal tight; colorbar; caxis([-0.1, 0.2])
title('Slice 80 of the xQSM_{invivo}');
err  = xQSM_invivo_recon - label;
subplot(122), imagesc(err(:,:,80)'); colormap gray; axis equal tight; colorbar; caxis([-0.1, 0.2])
title('Error');

figure,
subplot(121), imagesc(Unet_invivo_recon(:,:,80)'); colormap gray; axis equal tight; colorbar; caxis([-0.1, 0.2])
title('Slice 80 of the Unet_{invivo}');
err  = Unet_invivo_recon - label;
subplot(122), imagesc(err(:,:,80)'); colormap gray; axis equal tight; colorbar; caxis([-0.1, 0.2])
title('Error');

figure,
subplot(121), imagesc(xQSM_syn_recon(:,:,80)'); colormap gray; axis equal tight; colorbar; caxis([-0.1, 0.2])
title('Slice 80 of the xQSM_{syn}');
err  = xQSM_syn_recon - label;
subplot(122), imagesc(err(:,:,80)'); colormap gray; axis equal tight; colorbar; caxis([-0.1, 0.2])
title('Error');

figure,
subplot(121), imagesc(Unet_syn_recon(:,:,80)'); colormap gray; axis equal tight; colorbar; caxis([-0.1, 0.2])
title('Slice 80 of the Unet_{syn}');
err  = Unet_syn_recon - label;
subplot(122), imagesc(err(:,:,80)'); colormap gray; axis equal tight; colorbar; caxis([-0.1, 0.2])
title('Error');


%% use default pnsr and ssim 
PSNR_xQSM_invivo = psnr(xQSM_invivo_recon, single(label));
fprintf('PSNR of xQSM_invivo is %f\n', PSNR_xQSM_invivo);
PSNR_Unet_invivo = psnr(Unet_invivo_recon, single(label));
fprintf('PSNR of Unet_invivo is %f\n', PSNR_Unet_invivo);
PSNR_xQSM_syn = psnr(xQSM_syn_recon, single(label));
fprintf('PSNR of xQSM_syn is %f\n', PSNR_xQSM_syn);
PSNR_Unet_syn = psnr(Unet_syn_recon, single(label));
fprintf('PSNR of Unet_syn is %f\n', PSNR_Unet_syn);

SSIM_xQSM_invivo = ssim(xQSM_invivo_recon, single(label));
fprintf('SSIM of xQSM_invivo is %f\n', SSIM_xQSM_invivo);
SSIM_Unet_invivo = ssim(Unet_invivo_recon, single(label));
fprintf('SSIM of Unet_invivo is %f\n', SSIM_Unet_invivo);
SSIM_xQSM_syn = ssim(xQSM_syn_recon, single(label));
fprintf('SSIM of xQSM_syn is %f\n', SSIM_xQSM_syn);
SSIM_Unet_syn = ssim(Unet_syn_recon, single(label));
fprintf('SSIM of Unet_syn is %f\n', SSIM_Unet_syn);

%% save the files for ROI measurements; 
nii = make_nii(xQSM_invivo_recon, [1, 1, 1]);
save_nii(nii, 'Chi_xQSM_invivo.nii')

nii = make_nii(Unet_invivo_recon, [1, 1, 1]);
save_nii(nii, 'Chi_Unet_invivo.nii')

nii = make_nii(xQSM_syn_recon, [1, 1, 1]);
save_nii(nii, 'Chi_xQSM_syn.nii')

nii = make_nii(Unet_syn_recon, [1, 1, 1]);
save_nii(nii, 'Chi_Unet_syn.nii')
