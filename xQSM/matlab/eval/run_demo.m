%% navigate MATLAB to the 'eval' folder
% cd('~/matlab/deepMRI/xQSM/eval');


%% add NIFTI matlab toolbox for read and write NIFTI format
addpath ./NIFTI 


%% read in field map and COSMOS map (3rd dimension is z/B0 direction)
% note the size of the field map input needs to be divisibel by 8
% otherwise 0 padding should be done first
nii = load_nii('field_input.nii'); % replace the file name with yours. 
field = double(nii.img);
mask = field ~= 0; % brain tissue mask

% ADD imagesc of field and mask of a middle brain slice

%% read label (for evaluation purpose)
nii = load_nii('cosmos_label.nii'); % replace the file name with yours. 
label = double(nii.img);

% ADD imagesc of COSMOS of a middle brain slice

%% start recons
if canUseGPU()
    % (1) if your MATLAB is configured with CUDA GPU acceleration
    Unet_recon = Eval(field, 'Unet_invivo', 'gpu');
    xQSM_recon = Eval(field, 'xQSM_invivo', 'gpu');
    % YANG: put 2* inside Eval function
    % the reconstrutcion need to be multiplied by 2 in accordance with our training scheme for
    % networks trained with synthetic datasets; 
    Unet_syn_recon = 2 * Eval(field, 'Unet_syn', 'gpu'); 
    xQSM_syn_recon = 2 * Eval(field, 'xQSM_syn', 'gpu');
else
    % (2) otherwise if CUDA is not available, use CPU instead, this is much slower
    Unet_recon = Eval(field, 'Unet_invivo', 'cpu');
    xQSM_recon = Eval(field, 'xQSM_invivo', 'cpu');
    % the reconstrutcion need to be multiplied by 2 in accordance with our training scheme for
    % networks trained with synthetic datasets; 
    Unet_syn_recon = 2 * Eval(field, 'Unet_syn', 'cpu'); 
    xQSM_syn_recon = 2 * Eval(field, 'xQSM_syn', 'cpu');
end


%% image normalization (mean of brain tissue region set to 0)
label = label - sum(label(:)) / sum(mask(:));
label = label .* mask; 

xQSM_recon = xQSM_recon - sum(xQSM_recon(:)) / sum(mask(:));
xQSM_recon = xQSM_recon .* mask; 

Unet_recon = Unet_recon - sum(Unet_recon(:)) / sum(mask(:));
Unet_recon = Unet_recon .* mask; 

xQSM_syn_recon = xQSM_syn_recon - sum(xQSM_syn_recon(:)) / sum(mask(:));
xQSM_syn_recon = xQSM_syn_recon .* mask; 

Unet_syn_recon = Unet_syn_recon - sum(Unet_syn_recon(:)) / sum(mask(:));
Unet_syn_recon = Unet_syn_recon .* mask; 

% ADD imagesc of a middle brain slice from these 4 methods
% greyscale, adjust the window level, rotate properly, add title etc.

%% use default pnsr and ssim 
PSNR_xQSM = psnr(xQSM_recon, single(label));
fprintf('PSNR of xQSM is %f\n', PSNR_xQSM);
PSNR_Unet = psnr(Unet_recon, single(label));
%fprintf: YANG to add
PSNR_syn_xQSM = psnr(xQSM_syn_recon, single(label));
%fprintf
PSNR_syn_Unet = psnr(Unet_syn_recon, single(label));
%fprintf

SSIM_xQSM = ssim(xQSM_recon, single(label));
%fprintf
SSIM_Unet = ssim(Unet_recon, single(label));
%fprintf
SSIM_syn_xQSM = ssim(xQSM_syn_recon, single(label));
%fprintf
SSIM_syn_Unet = ssim(Unet_syn_recon, single(label));
%fprintf


%% save the files for ROI measurements; 
nii = make_nii(xQSM_recon, [1, 1, 1]);
save_nii(nii, 'Chi_xQSM_invivo.nii')

nii = make_nii(xQSM_recon, [1, 1, 1]);
save_nii(nii, 'Chi_Unet_invivo.nii')

nii = make_nii(xQSM_syn_recon, [1, 1, 1]);
save_nii(nii, 'Chi_xQSM_syn.nii')

nii = make_nii(xQSM_syn_recon, [1, 1, 1]);
save_nii(nii, 'Chi_Unet_syn.nii')

% YANG: be consistent with naming style: e.g. sometimes **xQSM_syn** other times ***syn_xQSM**
