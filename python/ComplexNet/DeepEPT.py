import torch 
import torch.nn as nn
import torch.nn.functional as F

class DeepRecon(nn.Module):
    def __init__(self,  Depth = 5):
        super(DeepRecon, self).__init__()
        self.first = CConv(2, 64)
        self.mid1 = MidBlock()
        self.mid2 = MidBlock()
        self.mid3 = MidBlock()
        self.mid4 = MidBlock()
        self.mid5 = MidBlock()  
        self.final = CConv(64, 2)
        #self.final = nn.Conv3d(64, 1, 3, padding = 1)

    def forward(self, x_r, x_i):
        INPUT_r = x_r
        INPUT_i = x_i
        x_r, x_i = self.first(x_r, x_i)
        x_r = F.relu(x_r)
        x_i = F.relu(x_i)
        x_r, x_i = self.mid1(x_r, x_i)
        x_r, x_i = self.mid2(x_r, x_i)
        x_r, x_i = self.mid3(x_r, x_i)
        x_r, x_i = self.mid4(x_r, x_i)
        x_r, x_i = self.mid5(x_r, x_i)
        x_r, x_i = self.final(x_r, x_i)
        x_r = x_r + INPUT_r
        x_i = x_i + INPUT_i
        return x_r, x_i


class MidBlock(nn.Module):
    def __init__(self):
        super(MidBlock, self).__init__()
        self.first = CConv(64, 64)
        self.second = CConv(64, 64)
          
    def forward(self, x_r, x_i):
        INPUT_r = x_r
        INPUT_i = x_i
        
        x_r, x_i = self.first(x_r, x_i)
        x_r = F.relu(x_r)
        x_i = F.relu(x_i)

        x_r = x_r + INPUT_r
        x_i = x_i + INPUT_i

        x_r, x_i = self.second(x_r, x_i)
        x_r = F.relu(x_r)
        x_i = F.relu(x_i)
        return x_r, x_i


class CConv(nn.Module):
    def __init__(self, inCs, outCs):
        super(CConv, self).__init__()
        self.rrConv = nn.Conv3d(inCs // 2, outCs // 2, 3, padding = 1)
        self.riConv = nn.Conv3d(inCs // 2, outCs // 2, 3, padding = 1)
        self.iiConv = nn.Conv3d(inCs // 2, outCs // 2, 3, padding = 1)
        self.irConv = nn.Conv3d(inCs // 2, outCs // 2, 3, padding = 1)
          
    def forward(self, x_r, x_i):
        y_rr = self.rrConv(x_r)
        y_ir = self.irConv(x_i)
        y_ii = self.iiConv(x_i)
        y_ri = self.riConv(x_r)
        y_r = y_rr + y_ir 
        y_i = y_ii + y_ri
        return y_r, y_i


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

###################################
if __name__ == '__main__':
    deepept = DeepRecon()
    print(deepept.state_dict)
    print(get_parameter_number(deepept))
    real = torch.randn(1, 1, 48, 48, 48, dtype=torch.float)
    imag = torch.randn(1, 1, 48, 48, 48, dtype=torch.float)
    print('real' + str(real.size()))
    print('imag' + str(imag.size()))
    real, imag = deepept(real, imag)
    print('output(1): '+str(real.size()))
    print('output(2): '+str(imag.size()))