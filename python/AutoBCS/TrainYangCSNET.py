################### train yangCSNet #####################
#########  Network Training #################### 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import time
from yangCSNet import *
from yangDataLoadCSNet import *
#########  Section 1: DataSet Load #############
def yangDataLoad(Batch_size):
    DATA_DIRECTORY = './trainPatch'
    DATA_LIST_PATH = './test_IDs.txt'
    dst = yangDataSet(DATA_DIRECTORY,DATA_LIST_PATH)
    print('dataLength: %d'%dst.__len__())
    trainloader = data.DataLoader(dst, batch_size = Batch_size, shuffle=True, drop_last = True)
    return trainloader

def yangSaveNet(yangCSNet, deepReconNet, enSave = False):
    print('save results')
    #### save the
    if enSave:
        torch.save(yangCSNet, './yangEntireyangCSNet.pth')
    else:
        torch.save(yangCSNet.state_dict(), './yangCSNET_100EPO_64BATCH_15_AUG_100_l1_rep4.pth')
        torch.save(deepReconNet.state_dict(), './yangDeepReconNET_100EPO_64BATCH_15_AUG_100_l1_rep4.pth')

def yangTrainNet(yangCSNet, deepReconNet, LR = 0.001, Batchsize = 32, Epoches = 40 , useGPU = True):
    print('yangCSNet')
    print('DataLoad')
    trainloader = yangDataLoad(Batchsize)
    print('Dataload Ends')

    print('Training Begins')
    criterion = nn.L1Loss()
    optimizer1 = optim.Adam(yangCSNet.parameters())
    optimizer2 = optim.Adam(deepReconNet.parameters())
    scheduler1 = LS.MultiStepLR(optimizer1, milestones = [50, 80], gamma = 0.1)
    scheduler2 = LS.MultiStepLR(optimizer2, milestones = [50, 80], gamma = 0.1)

    ## start the timer. 
    time_start=time.time()
    if useGPU:
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "Available GPUs!")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            yangCSNet = nn.DataParallel(yangCSNet)
            yangCSNet.to(device)
            deepReconNet = nn.DataParallel(deepReconNet)
            deepReconNet.to(device)
            for epoch in range(1, Epoches + 1):
                acc_loss = 0.0
                for i, data in enumerate(trainloader):
                    Inputs, Name = data
                    Inputs = Inputs.to(device)
                    ## zero the gradient buffers 
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    ## forward: 
                    Pred_IR = yangCSNet(Inputs)
                    pred_bl = deepReconNet(Pred_IR)
                    ## loss
                    loss1 = criterion(Pred_IR, Inputs)
                    loss2 = criterion(pred_bl, Inputs)
                    ## backward
                    loss1.backward(retain_graph = True)
                    loss2.backward()
                    ##
                    optimizer1.step()
                    optimizer2.step()
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    ## print statistical information 
                    ## print every 20 mini-batch size
                    if i % 19 == 0:
                        acc_loss1 = loss1.item()   
                        acc_loss2 = loss2.item()   
                        time_end=time.time()
                        print('Outside: Epoch : %d, batch: %d, Loss1: %f, loss2 : %f, lr1: %f, lr2: %f, used time: %d s' %
                            (epoch, i + 1, acc_loss1, acc_loss2, optimizer1.param_groups[0]['lr'], optimizer2.param_groups[0]['lr'], time_end - time_start))  
                scheduler1.step()
                scheduler2.step()
        else:
            pass
            print('No Cuda Device!')
            quit()        
    print('Training Ends')
    yangSaveNet(yangCSNet, deepReconNet)

if __name__ == '__main__':
    ## data load
    ## create network 
    yangCSNet = yangCSNet(10)
    deepReconNet = OctNet(2)
    print(yangCSNet.state_dict)
    print(deepReconNet.state_dict)
    print(get_parameter_number(yangCSNet))
    print(get_parameter_number(deepReconNet))
    ###### use this line to check if all layers 
    ###### are leanrable in this programe. 
    print('100EPO yangCSNET')
    ## train network
    yangTrainNet(yangCSNet,deepReconNet, LR = 0.001, Batchsize = 64, Epoches = 100 , useGPU = True)

