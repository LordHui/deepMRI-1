import torch.nn as nn
import torch.nn.functional as F

class DeepRecon(nn.Module):
    def __init__(self,  n = 5):
        super(DeepRecon, self).__init__()
        self.first = nn.Conv2d(1, 64, 3, padding = 1)
        self.mid1 = MidBlock()
        self.mid2 = MidBlock()
        self.mid3 = MidBlock()
        self.mid4 = MidBlock()
        self.mid5 = MidBlock()  
        self.final = nn.Conv2d(64, 1, 3, padding = 1)

    def forward(self, x):
        INPUT = x
        x = self.first(x)
        x = F.relu(x)
        x = self.mid1(x)
        x = self.mid2(x)
        x = self.mid3(x)
        x = self.mid4(x)
        x = self.mid5(x)
        x = self.final(x)
        x = x + INPUT
        return x


class MidBlock(nn.Module):
    def __init__(self):
        super(MidBlock, self).__init__()
        self.first = nn.Conv2d(64, 64, 3, padding = 1)
        self.second = nn.Conv2d(64, 64, 3, padding = 1)
          
    def forward(self, x):
        INPUT = x
        x = self.first(x)
        x = F.relu(x)
        x = x + INPUT
        x = self.second(x)
        x = F.relu(x)
        return x