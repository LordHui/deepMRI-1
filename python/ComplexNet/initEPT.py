################  New CS-Net ########################
### import packages. 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
# import scipy.io as scio

class InitEPT(nn.Module):
    def __init__(self,  BlockSize = 16):
        super(InitEPT, self).__init__()
        self.init_rr = yangEPTNet(BlockSize)
        self.init_ri = yangEPTNet(BlockSize)
        self.init_ii = yangEPTNet(BlockSize)
        self.init_ir = yangEPTNet(BlockSize)

    def forward(self, x_r, x_i):
        y_rr = self.init_rr(x_r)
        y_ri = self.init_ri(x_r)
        y_ii = self.init_ii(x_i)
        y_ir = self.init_ir(x_i)
        y_r = y_rr + y_ir
        y_i = y_ii + y_ri
        return y_r, y_i

class yangEPTNet(nn.Module):
    def __init__(self, BlockSize = 16):
        ## 1 BL for 0.01 and 2 ELs for 0.1 and 0.2
        super(yangEPTNet, self).__init__()
        self.BlockSize = BlockSize
        self.init_bl = nn.Conv3d(1, BlockSize ** 3, BlockSize, stride = BlockSize)
        #self.deep_bl = DeepRecon()

    def forward(self, x):
        ## cut the image into patches of pre-defined blocksize
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        nb = x.size(0)
        x = self.yangImageCrop(x, self.BlockSize)
        y_bl = torch.zeros(nb, self.num_patches, self.BlockSize ** 3, 1, 1, 1)
        y_bl = y_bl.to(device)
        #print(x.size())
        ## initial Recon
        for i in range(0, self.num_patches):
            temp_bl = x[:,i,:,:,:,:]
            temp_bl = temp_bl.to(device)
            temp_bl = self.init_bl(temp_bl)
            y_bl[:, i, :, :, :] = temp_bl

        ## reshape and concatenation. 
        y_IR = self.yangReshape(y_bl, self.BlockSize)
        y_IR = y_IR.to(device)
        return y_IR

    def yangImageCrop(self, x, BlockSize = 16):
        H = x.size(2)
        L = x.size(3)
        D = x.size(4)
        nb = x.size(0)
        nc = x.size(1)
        num_patches = H * L * D // (BlockSize ** 3)
        y = torch.zeros(nb, num_patches, nc, BlockSize, BlockSize, BlockSize)
        #y = y.to(device)
        ind1 = range(0, H, BlockSize)
        ind2 = range(0, L, BlockSize)
        ind3 = range(0, D, BlockSize)
        count = 0
        for i in ind1:
            for j in ind2:
                for k in ind3:
                    temp = x[:,:,i:i+ BlockSize, j:j+BlockSize, k:k+BlockSize]
                    #temp = torch.unsqueeze(temp, 1)
                    #print(temp.size())
                    #temp2 = y[:,count,:,:,:,:,]
                    #print(temp2.size())
                    y[:,count,:,:,:,:,] = temp
                    count = count + 1
        #print('Crop: %d'%count)
        #print(y.size())
        self.oriH = H
        self.oriL = L
        self.oriD = D
        self.num_patches = num_patches
        return y

    def yangReshape(self, x, BlockSize = 16):
        nb = x.size(0)
        y = torch.zeros(nb, 1, self.oriH, self.oriL, self.oriD)
        #y = y.to(device)
        ind1 = range(0, self.oriH, BlockSize)
        ind2 = range(0, self.oriL, BlockSize)
        ind3 = range(0, self.oriD, BlockSize)
        count = 0
        for i in ind1:
            for j in ind2:
                for k in ind3:
                    temp = x[:,count,:,:,:,:]
                    temp = torch.reshape(temp, [nb, 1, BlockSize, BlockSize, BlockSize])
                    y[:,:,i:i+BlockSize, j:j+BlockSize, k:k+BlockSize] = temp
                    count = count + 1
        #print('reshape: %d'% count)
        return y

def weights_init(m):
    if isinstance(m, nn.Conv3d):
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)   
    if isinstance(m, nn.ConvTranspose3d):
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)   
    if isinstance(m, nn.BatchNorm3d):
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)   

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

###################################
if __name__ == '__main__':
    yangeptnet = InitEPT(BlockSize = 16)
    print(yangeptnet.state_dict)
    print(get_parameter_number(yangeptnet))
    real = torch.randn(1, 1, 48, 48, 48, dtype=torch.float)
    imag = torch.randn(1, 1, 48, 48, 48, dtype=torch.float)
    print('real' + str(real.size()))
    print('imag' + str(imag.size()))
    real, imag = yangeptnet(real, imag)
    print('output(1): '+str(real.size()))
    print('output(2): '+str(imag.size()))