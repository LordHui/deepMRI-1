################  New CS-Net ########################
### import packages. 
from yangMFOctnet import * 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
# import scipy.io as scio

class yangCSNet(nn.Module):
    def __init__(self, SamplingPoints, BlockSize = 32):
        ## 1 BL for 0.01 and 2 ELs for 0.1 and 0.2
        super(yangCSNet, self).__init__()
        self.BlockSize = BlockSize
        self.SamplingPoints = SamplingPoints
        self.sampling = nn.Conv2d(1, SamplingPoints , BlockSize, stride = 32, padding = 0,bias=False)
        nn.init.normal_(self.sampling.weight, mean=0.0, std=0.028)
        self.init_bl = nn.Conv2d(SamplingPoints, BlockSize ** 2, 1, bias=False)
        #self.deep_bl = OctNet(2)

    def forward(self, x):
        ## cut the image into patches of pre-defined blocksize
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nb = x.size(0)
        x = self.yangImageCrop(x)
        x_bl = torch.zeros(nb, self.num_patches, self.SamplingPoints, 1, 1)
        for i in range(0, self.num_patches):
            temp = x[:, i, :, :, :]
            temp = temp.to(device)
            temp = self.sampling(temp)
            x_bl[:, i, :, :, :] = temp

        y_bl = torch.zeros(nb, self.num_patches, self.BlockSize ** 2, 1, 1)

        ## initial Recon
        for i in range(0, self.num_patches):
            temp_bl = x_bl[:,i,:,:,:]
            temp_bl = torch.squeeze(temp_bl, 1)
            temp_bl = temp_bl.to(device)
            temp_bl = self.init_bl(temp_bl)
            y_bl[:, i, :, :, :] = temp_bl

        ## reshape and concatenation. 
        y_bl = self.yangReshape(y_bl)

        ## deep recon. 
        y_bl = y_bl.to(device)
        y_IR = y_bl

        #y_bl = self.deep_bl(y_bl)
        return y_IR

    def yangImageCrop(self, x, BlockSize = 32):
        H = x.size(2)
        L = x.size(3)
        nb = x.size(0)
        nc = x.size(1)
        num_patches = H * L // (BlockSize ** 2)
        y = torch.zeros(nb, num_patches, nc, BlockSize, BlockSize)
        ind1 = range(0, H, BlockSize)
        ind2 = range(0, L, BlockSize)
        count = 0
        for i in ind1:
            for j in ind2:
                temp = x[:,:,i:i+ BlockSize, j:j+BlockSize]
                #temp = torch.unsqueeze(temp, 1)
                #print(temp.size())
                temp2 = y[:,count,:,:,:,]
                #print(temp2.size())
                y[:,count,:,:,:,] = temp
                count = count + 1
        #print('Crop: %d'%count)
        #print(y.size())
        self.oriH = H
        self.oriL = L
        self.num_patches = num_patches
        return y

    def yangReshape(self, x, BlockSize = 32):
        nb = x.size(0)
        y = torch.zeros(nb, 1, self.oriH, self.oriL)
        ind1 = range(0, self.oriH, BlockSize)
        ind2 = range(0, self.oriL, BlockSize)
        count = 0
        for i in ind1:
            for j in ind2:
                temp = x[:,count,:,:,:]
                temp = torch.squeeze(temp, 1)
                temp = torch.reshape(temp, [nb, 1, BlockSize, BlockSize])
                y[:,:,i:i+BlockSize, j:j+BlockSize] = temp
                count = count + 1
        #print('reshape: %d'% count)
        return y


###################################
if __name__ == '__main__':
    yangcsnet = yangCSNet()
    print(yangcsnet.state_dict)
    print(get_parameter_number(yangcsnet))
    x = torch.randn(2, 1, 256, 256, dtype=torch.float)
    print('input ' + str(x.size()))
    print(x.dtype)
    y_bl, y_el1, y_el2 = yangcsnet(x)
    print('output(1): '+str(y_bl.size()))
    print('output(2): '+str(y_el1.size()))
    print('output(3): '+str(y_el2.size()))