# coding = utf-8
import scipy.stats as st
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import cv2


# Load config file 
opt_file = open(sys.argv[1], "r")
opt = json.load(opt_file)
opt_file.close()

device = torch.device(opt['device'])

#class CSDN_Tem(nn.Module):
#    def __init__(self):
#        super(CSDN_Tem, self).__init__()
#        self.depth_conv = nn.Conv2d(
#            in_channels=3,
#            out_channels=3,
#            kernel_size=3,
#            stride=1,
#            padding='SAME',
#            groups=3
#        )

#   def forward(self, input):
#       out = self.depth_conv(input)
#       return out


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    #out_filter = out_filter.reshape((kernlen, kernlen))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter

def blur(x):
    #x.to(device)
    kernel_var = gauss_kernel(21, 3, 3) 
    #(21,21,3,1)  
    #print("kernel_var1:",kernel_var.shape)
    kernel_var = kernel_var.transpose(2,3,0,1)
    #(3,1,21,21)
    #print("kernel_var2:",kernel_var.shape)
    #kernel_var = torch.FloatTensor(kernel_var)
    kernel_var = torch.FloatTensor(kernel_var).to(device)    
    #x = x.transpose(2, 0, 1)
    #x=torch.transpose(x,[2,0,1])
    #x = torch.FloatTensor(x)
    #x = torch.FloatTensor(x).to(device)
    #x = (torch.FloatTensor(x).unsqueeze(0))    #1,460,620,3
    y = F.conv2d(x, kernel_var,stride = 1, padding=10, groups=3).to(device)
    return y

#image = cv2.imread('/home/liuyanting/papercode/dehaze/Feature_Forwarding19/1_1.png')
#print(image.shape)
#print(blur(image).shape)
   
