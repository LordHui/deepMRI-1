################### train yangCSNet #####################
#########  Network Training #################### 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import time
from initEPT import *
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

def yangSaveNet(yangCSNet, deepReconNet, enSave = False):
    print('save results')
    #### save the
    if enSave:
        torch.save(yangCSNet, './yangEntireyangEPTNet.pth')
    else:
        torch.save(yangCSNet.state_dict(), './InitEPT_MSE_50EPO_26_SEP.pth')
        torch.save(deepReconNet.state_dict(), './DEEPEPT_MSE_50EPO_26_SEP.pth')

def yangTrainNet(yangCSNet, deepReconNet, LR = 0.001, Batchsize = 32, Epoches = 40 , useGPU = True):
    print('yangEPTNet')
    print('DataLoad')
    trainloader = yangDataLoad(Batchsize)
    print('Dataload Ends')

    print('Training Begins')
    criterion = nn.MSELoss()
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
                    image_r, image_i, label_r, label_i, Name = data
                    image_r = image_r.to(device)
                    image_i = image_i.to(device)
                    label_r = label_r.to(device)
                    label_i = label_i.to(device)
                    ## zero the gradient buffers 
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    ## forward: 
                    pred_IR_r, pred_IR_i = yangCSNet(image_r, image_i)
                    pred_bl_r, pred_bl_i = deepReconNet(pred_IR_r, pred_IR_i)
                    ## loss
                    loss1 = criterion(pred_IR_r, label_r)
                    loss2 = criterion(pred_IR_i, label_i)
                    loss3 = criterion(pred_bl_r, label_r)
                    loss4 = criterion(pred_bl_i, label_i)
                    ## backward
                    loss1.backward(retain_graph = True)
                    loss2.backward(retain_graph = True)
                    loss3.backward(retain_graph = True)
                    loss4.backward()
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
                        acc_loss3 = loss3.item()
                        acc_loss4 = loss4.item()   
                        time_end=time.time()
                        print('Outside: Epoch : %d, batch: %d, Loss1: %f, loss2 : %f, Loss3: %f, loss4 : %f, lr1: %f, lr2: %f, used time: %d s' %
                            (epoch, i + 1, acc_loss1, acc_loss2,acc_loss3, acc_loss4, optimizer1.param_groups[0]['lr'], optimizer2.param_groups[0]['lr'], time_end - time_start))  
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
    yangCSNet = InitEPT(BlockSize = 16)
    deepReconNet = DeepRecon()
    print(yangCSNet.state_dict)
    print(deepReconNet.state_dict)
    print(get_parameter_number(yangCSNet))
    print(get_parameter_number(deepReconNet))
    ###### use this line to check if all layers 
    ###### are leanrable in this programe. 
    print('50EPO EPTNET')
    ## train network
    yangTrainNet(yangCSNet,deepReconNet, LR = 0.001, Batchsize = 32, Epoches = 50, useGPU = True)

