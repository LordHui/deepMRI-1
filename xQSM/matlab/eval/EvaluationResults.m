%% read data
% read the local field maps as V_Field; 
% note the size of the field maps need to be divisibel by 8;
addpath ./nii 
nii = load_nii('lfs_cosmos_5_6DOF_resharp_1e-6.nii'); % replace the file name with yours. 
V_Field = nii.img;
mask = V_Field ~= 0; 

%% read label (if exists)
nii = load_nii('cosmos_5_6DOF_resharp_1e-6.nii'); % replace the file name with yours. 
label = nii.img;

%% Recons
Unet_recon = Eval(V_Field, 'Unet_invivo', 'gpu');
xQSM_recon = Eval(V_Field, 'xQSM_invivo', 'gpu');
% the reconstrutcion need to be multiplied by 2 in accordance with our training scheme for
% networks trained with synthetic datasets; 
Unet_syn_recon = 2 * Eval(V_Field, 'Unet_syn', 'gpu'); 
xQSM_syn_recon = 2 * Eval(V_Field, 'xQSM_syn', 'gpu');

%% image normalization
label = label - sum(label(:)) / sum(mask(:));
label = label .* mask; 

xQSM_recon = xQSM_recon - sum(xQSM_recon(:)) / sum(mask(:));
xQSM_recon = xQSM_recon .* mask; 

Unet_recon = Unet_recon - sum(Unet_recon(:)) / sum(mask(:));
Unet_recon = Unet_recon .* mask; 

Unet_syn_recon = Unet_syn_recon - sum(Unet_syn_recon(:)) / sum(mask(:));
Unet_syn_recon = Unet_syn_recon .* mask; 

xQSM_syn_recon = xQSM_syn_recon - sum(xQSM_syn_recon(:)) / sum(mask(:));
xQSM_syn_recon = xQSM_syn_recon .* mask; 

%% use default pnsr and ssim calculator; 
PSNR_xQSM = psnr(xQSM_recon, single(label))
PSNR_Unet = psnr(Unet_recon, single(label))
PSNR_syn_xQSM = psnr(xQSM_syn_recon, single(label))
PSNR_syn_Unet = psnr(Unet_syn_recon, single(label))

SSIM_xQSM = ssim(xQSM_recon, single(label))
SSIM_Unet = ssim(Unet_recon, single(label))
SSIM_syn_xQSM = ssim(xQSM_syn_recon, single(label))
SSIM_syn_Unet = ssim(Unet_syn_recon, single(label))


%% save the files for ROI measurements; 
nii = make_nii(xQSM_recon, [1 , 1, 1]);
save_nii(nii, 'Chi_xQSM_invivo_lfs_cosmos.nii')

nii = make_nii(xQSM_recon, [1 , 1, 1]);
save_nii(nii, 'Chi_Unet_invivo_lfs_cosmos.nii')

nii = make_nii(xQSM_syn_recon, [1 , 1, 1]);
save_nii(nii, 'Chi_xQSM_syn_lfs_cosmos.nii')

nii = make_nii(xQSM_syn_recon, [1 , 1, 1]);
save_nii(nii, 'Chi_Unet_syn_lfs_cosmos.nii')
