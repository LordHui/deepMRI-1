Complex Res U-net for MRI acceleration. 

file structures:  
Provided for network training:
	1. folder CCblocks3DBN_mse: basic blocks of complex U-net; 
	2. Create3DCConvNet130BN.m: function to creat a complete compelx U-net; 
	3. yangDataLoad.m: data loading for training;
	4. TrainCConvNet130_mse.m: Script of training;
	5. spliitingLayer.m: to split complex data into two streams of real tensors. 
Necessary but not provided: 
	subsampling masks; 

other resources: 
	# Multi-stream-CNN


Requires Matlab 2019b or higher
UQ License


