################### train yangCSNet #####################
#########  Network Training #################### 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import time
from DeepEPT import *
from yangDataLoad import *
#########  Section 1: DataSet Load #############
def yangDataLoad(Batch_size):
    DATA_DIRECTORY = '.'
    DATA_LIST_PATH = './test_IDs.txt'
    dst = yangDataSet(DATA_DIRECTORY,DATA_LIST_PATH)
    print('dataLength: %d'%dst.__len__())
    trainloader = data.DataLoader(dst, batch_size = Batch_size, shuffle=True, drop_last = True)
    return trainloader

def yangSaveNet(deepReconNet, enSave = False):
    print('save results')
    #### save the
    if enSave:
        pass
    else:
        torch.save(deepReconNet.state_dict(), './DEEPEPT_MSE_100EPO_01_OCT_noinit_rep1.pth')

def yangTrainNet(deepReconNet, LR = 0.001, Batchsize = 32, Epoches = 40 , useGPU = True):
    print('yangEPTNet')
    print('DataLoad')
    trainloader = yangDataLoad(Batchsize)
    print('Dataload Ends')

    print('Training Begins')
    criterion = nn.MSELoss()
    optimizer2 = optim.Adam(deepReconNet.parameters())
    scheduler2 = LS.MultiStepLR(optimizer2, milestones = [50, 80], gamma = 0.1)

    ## start the timer. 
    time_start=time.time()
    if useGPU:
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "Available GPUs!")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            deepReconNet = nn.DataParallel(deepReconNet)
            deepReconNet.to(device)
            for epoch in range(1, Epoches + 1):
                acc_loss = 0.0
                for i, data in enumerate(trainloader):
                    image_r, image_i, label_r, label_i, Name = data
                    image_r = image_r.to(device)
                    image_i = image_i.to(device)
                    label_r = label_r.to(device)
                    label_i = label_i.to(device)
                    ## zero the gradient buffers 
                    optimizer2.zero_grad()
                    ## forward: 
                    pred_bl_r, pred_bl_i = deepReconNet(image_r, image_i)
                    ## loss
                    loss1 = criterion(pred_bl_r, label_r)
                    loss2 = criterion(pred_bl_i, label_i)
                    ## backward
                    loss1.backward(retain_graph = True)
                    loss2.backward()
                    ##
                    optimizer2.step()
                    optimizer2.zero_grad()
                    ## print statistical information 
                    ## print every 20 mini-batch size
                    if i % 19 == 0:
                        acc_loss1 = loss1.item()   
                        acc_loss2 = loss2.item()
                        time_end=time.time()
                        print('Outside: Epoch : %d, batch: %d, Loss1: %f, loss2 : %f, lr2: %f, used time: %d s' %
                            (epoch, i + 1, acc_loss1, acc_loss2, optimizer2.param_groups[0]['lr'], time_end - time_start))  
                scheduler2.step()
        else:
            pass
            print('No Cuda Device!')
            quit()        
    print('Training Ends')
    yangSaveNet(deepReconNet)

if __name__ == '__main__':
    ## data load
    ## create network 
    deepReconNet = DeepRecon()
    print(deepReconNet.state_dict)
    print(get_parameter_number(deepReconNet))
    ###### use this line to check if all layers 
    ###### are leanrable in this programe. 
    print('100EPO EPTNET')
    ## train network
    yangTrainNet(deepReconNet, LR = 0.001, Batchsize = 32, Epoches = 100, useGPU = True)

