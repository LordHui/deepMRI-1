#########  Network Training #################### 
import torch 
import torch.nn as nn
import torch.optim as optim
import time
from ResNet_yang import *
from yangDataLoad import *
#########  Section 1: DataSet Load #############
def yangDataLoad(Batch_size):
    DATA_DIRECTORY = './trainPatch48'
    DATA_LIST_PATH = './test_IDs.txt'
    dst = yangDataSet(DATA_DIRECTORY,DATA_LIST_PATH)
    print('dataLength: %d'%dst.__len__())
    trainloader = data.DataLoader(dst, batch_size = Batch_size, shuffle=True, drop_last = True)
    return trainloader

def yangSaveNet(resnet, enSave = False):
    print('save results')
    #### save the
    if enSave:
        torch.save(resnet, './yangEntireResNet.pth')
    else:
        torch.save(resnet.state_dict(), './UNET_48_Patch_3L_10EPO.pth')


def yangTrainNet(resnet, LR = 0.001, Batchsize = 32, Epoches = 40 , useGPU = False):
    print('ResNet')
    print('DataLoad')
    trainloader = yangDataLoad(Batchsize)
    print('Dataload Ends')

    print('Training Begins')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(resnet.parameters(), lr = LR)
    ## start the timer. 
    time_start=time.time()
    if useGPU:
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "Available GPUs!")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            resnet = nn.DataParallel(resnet)
            resnet.to(device)
            for epoch in range(1, Epoches + 1):
                acc_loss = 0.0
                for i, data in enumerate(trainloader):
                    Inputs, Labels, Name = data
                    Inputs = Inputs.to(device)
                    Labels = Labels.to(device)
                    ## zero the gradient buffers 
                    optimizer.zero_grad()
                    ## forward: 
                    pred = resnet(Inputs)
                    ## loss
                    loss = criterion(pred, Labels)
                    ## backward
                    loss.backward()
                    ## learning one single step
                    optimizer.step()
                    ## print statistical information 
                    ## print every 20 mini-batch size
                    if i % 19 == 0:
                        acc_loss = loss.item()   
                        time_end=time.time()
                        print('Outside: Epoch : %d, batch: %d, Loss: %f, used time: %d s' %
                            (epoch, i + 1, acc_loss, time_end - time_start))      
        else:
            pass
            print('No Cuda Device!')
            quit()        
    print('Training Ends')
    yangSaveNet(resnet)

if __name__ == '__main__':
    ## data load
    ## create network 
    resnet = ResNet(3)
    a = resnet.state_dict()
    print('10EPO')
    print(a)
    print(get_parameter_number(resnet))
    ###### use this line to check if all layers 
    ###### are leanrable in this programe. 
    ## train network
    yangTrainNet(resnet, LR = 0.001, Batchsize = 32, Epoches = 10 , useGPU = True)
