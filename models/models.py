'''
References :
 Linknet from e-lab
     https://github.com/e-lab/pytorch-linknet
 Refinement layers from He,Zhang
     https://github.com/hezhangser/DCPDN
 Discrimiantor is taken from Aitor Ruano's implementation of SRGAN
     https://github.com/aitorzip/PyTorch-SRGAN
     https://arxiv.org/abs/1609.04802
     https://arxiv.org/abs/1710.05941
'''
import torch
import torch.nn as nn

from torch.nn import functional as F
#from torchvision.models import resnet, vgg16
from torchvision.models import vgg16
#resnet add attention
from models import resnet_attention

from collections import namedtuple



class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.vgg = Vgg16(requires_grad=False).cuda()
        self.loss = nn.MSELoss()
    def forward(self, output, target, weight=1):
        f_output = self.vgg(output).relu2_2
        f_target = self.vgg(target).relu2_2
        return weight * self.loss(f_output, f_target)

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x
        
class ConvertBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=1, stride=1, padding=0,bias=True):
        super(ConvertBlock, self).__init__()
        self.convert = nn.Sequential(nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias),
                                  nn.BatchNorm2d(output_size),
                                  nn.ReLU(inplace=True),)

    def forward(self, x):    
        out = self.convert(x)    
        return out
        

class DeConvertBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding,output_padding, bias=True):
        super(DeConvertBlock, self).__init__()
        self.deconvert = nn.Sequential(nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding,output_padding, bias=bias),
                                  nn.BatchNorm2d(output_size),
                                  nn.ReLU(inplace=True),)

    def forward(self, x):    
        out = self.deconvert(x)    
        return out


class UpBlock(torch.nn.Module):
    def __init__(self, channels, kernel=3, stride=1, padding=1, bias=True):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(nn.Conv2d(channels, channels, kernel, 1, padding), 
                                nn.BatchNorm2d(channels),
                                nn.ReLU(inplace=True), 
                                nn.Conv2d(channels, channels, kernel, 1, padding), 
                                nn.BatchNorm2d(channels),
                                nn.ReLU(inplace=True),  
                                nn.Conv2d(channels, channels, kernel, 1, padding),
                                nn.BatchNorm2d(channels), 
                                nn.ReLU(inplace=True), )

    def forward(self, x):    
        out = self.up(x)    
        return out

# #models1 used        
# class FIBlock(torch.nn.Module):
    # def __init__(self, channels, kernel=1, stride=1, padding=0, bias=True):
        # super(FIBlock, self).__init__()
        # self.ex_e = nn.Sequential(nn.Conv2d(channels, 1, kernel, stride, padding),                         
                                # nn.Sigmoid()) 
        # self.ex_d = nn.Sequential(nn.Conv2d(channels, 1, kernel, stride, padding),                         
                                # nn.Sigmoid())                                

    # def forward(self,x,y):
        # out_x = self.ex_d(y) * x + x     
        # out_y = self.ex_e(x) * y + y 
        # out = out_x + out_y     
        # return out

class FIBlock(torch.nn.Module):
    def __init__(self, channels, kernel=1, stride=1, padding=0, bias=True):
        super(FIBlock, self).__init__()
        self.ex_e = nn.Sequential(nn.Conv2d(channels, 1, kernel, stride, padding),                         
                                nn.Sigmoid())                                

    def forward(self,x,y):    
        out_y = self.ex_e(x) * y + y 
        out = out_y + x     
        return out
        
'''        
#add GAU(global attention upsample)
class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        #fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        return fms_att   
        
      
class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv_avg1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.conv_max1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        #self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fms_high, fms_low, fm_mask=None):
    
        b, c, h, w = fms_high.shape

        fms_high_gp_1 = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp_2 = nn.MaxPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp_1 = self.conv_avg1x1(fms_high_gp_1)
        fms_high_gp_2 = self.conv_max1x1(fms_high_gp_2)
        fms_high_gp = self.sigmoid(fms_high_gp_1+fms_high_gp_2)

        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        return fms_att   

'''
'''        
class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv_avg1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.conv_max1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        #self.relu = nn.ReLU(inplace=True)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, fms_high, fms_low, fm_mask=None):
    
        b, c, h, w = fms_high.shape

        fms_high_gp_1 = self.avg_pool(fms_high)        
        fms_high_gp_2 = self.max_pool(fms_high)
        fms_high_gp_1 = self.conv_avg1x1(fms_high_gp_1)
        fms_high_gp_2 = self.conv_max1x1(fms_high_gp_2)        
        fms_high_gp = self.sigmoid(fms_high_gp_1+fms_high_gp_2)

        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp

        return fms_att             
'''


class LinkNet50(nn.Module):
    def __init__(self, n_classes=3,pad=(0,0,0,0)):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(LinkNet50, self).__init__()

        base = resnet_attention.resnet50(pretrained=True)

        #self.in_block = nn.Sequential(
        #    base.conv1,
        #    base.bn1,
        #    base.relu,
        #    base.maxpool
        #)
        self.base_conv1 = base.conv1
        self.base_bn1 = base.bn1
        self.base_relu = base.relu
        self.base_maxpool = base.maxpool
        
        self.pad = nn.ReflectionPad2d(pad)
        self.pad_size = [int(x/2) for x in pad]
       
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = Decoder(256, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(512, 256, 3, 2, 1, 1)
        self.decoder3 = Decoder(1024, 512, 3, 2, 1, 1)
        self.decoder4 = Decoder(2048, 1024, 3, 2, 1, 1)

        #self.tp_conv1 = DeConvertBlock(128, 64, 3, 2, 1, 1),
        #self.conv2 = ConvertBlock(64, 64, 3, 1, 1),
        self.tp_conv2 = nn.ConvTranspose2d(128, n_classes, 2, 2, 0)
        self.lsm = nn.ReLU() #nn.LogSoftmax(dim=1)
        self.tp_conv2_1 = nn.Conv2d(n_classes,3, 3, 1, 1)
        self.lsm_1 = nn.ReLU(inplace=True)
        
        
        # self.ds1 = nn.Conv2d(64, 256, 1)
        # self.ds2 = nn.Conv2d(256, 512, 1, 2 )
        # self.ds3 = nn.Conv2d(512, 1024, 1, 2)
        # self.ds4 = nn.Conv2d(1024, 2048, 1, 2)
        
        #modify2 start(add edge guidance)            
        self.conv3 = ConvertBlock(64, 64, 3, 2, 1)  
        self.conv4 = ConvertBlock(256, 64)
        self.conv5 = ConvertBlock(128, 256, 3, 2, 1)                                 
        self.conv6 = ConvertBlock(256, 512, 3, 2, 1)                              
        
        self.tp_conv3 = DeConvertBlock(512, 256, 3, 2, 1, 1) 
        self.tp_conv4 = DeConvertBlock(256, 128, 3, 2, 1, 1)                                     
        #self.tp_conv5 = DeConvertBlock(128, 64, 3, 1, 1, 0)
        
        #self.edge_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
        #                              nn.BatchNorm2d(32),
        #                              nn.ReLU(inplace=True),)
        #self.edge_conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
        #                           nn.BatchNorm2d(32),
        #                           nn.ReLU(inplace=True),)
        self.edge_conv3 = nn.ConvTranspose2d(128, 3, 2, 2, 0)
        self.edge_lsm = nn.ReLU() #nn.LogSoftmax(dim=1) 

        #add mutil
        self.convert2 = ConvertBlock(512, 512)
        self.convert3 = ConvertBlock(1024, 512)
        self.convert4 = ConvertBlock(2048, 512)
        self.up2 = UpBlock(512, 3, 1, 1)
        self.up3 = UpBlock(512, 3, 1, 1)  
        self.up4 = UpBlock(512, 3, 1, 1)
        self.deconvert2_0 = DeConvertBlock(512, 128, 3, 2, 1, 1)
        self.deconvert3_0 = DeConvertBlock(512, 128, 3, 2, 1, 1)
        self.deconvert3_1 = DeConvertBlock(128, 128, 3, 2, 1, 1)
        self.deconvert4_0 = DeConvertBlock(512, 128, 3, 2, 1, 1)
        self.deconvert4_1 = DeConvertBlock(128, 128, 3, 2, 1, 1)
        self.deconvert4_2 = DeConvertBlock(128, 128, 3, 2, 1, 1)
        self.FI_2 = FIBlock(128)
        self.FI_3 = FIBlock(128)
        self.FI_4 = FIBlock(128)
        self.up_tmp2 = UpBlock(128, 3, 1, 1)
        self.up_tmp3 = UpBlock(128, 3, 1, 1)  
        self.up_tmp4 = UpBlock(128, 3, 1, 1)
        
        #self.deconvert_tmp2_0 = DeConvertBlock(128, 64, 3, 2, 1, 1)
        #self.convert_tmp2 = ConvertBlock(64, 64, 3, 1, 1)
        self.deconvert_tmp2_1 = nn.ConvTranspose2d(128, n_classes, 2, 2, 0)
        self.deconvert_tmp2_2 = nn.ReLU(inplace=True)
        self.deconvert_tmp2_3 = nn.Conv2d(n_classes,3, 3, 1, 1)
        self.deconvert_tmp2_4 = nn.ReLU(inplace=True)
        
        #self.deconvert_tmp3_0 = DeConvertBlock(128, 64, 3, 2, 1, 1)
        #self.convert_tmp3 = ConvertBlock(64, 64, 3, 1, 1)
        self.deconvert_tmp3_1 = nn.ConvTranspose2d(128, n_classes, 2, 2, 0)
        self.deconvert_tmp3_2 = nn.ReLU(inplace=True)
        self.deconvert_tmp3_3 = nn.Conv2d(n_classes,3, 3, 1, 1)
        self.deconvert_tmp3_4 = nn.ReLU(inplace=True)
        
        #self.deconvert_tmp4_0 = DeConvertBlock(128, 64, 3, 2, 1, 1)
        #self.convert_tmp4 = ConvertBlock(64, 64, 3, 1, 1)
        self.deconvert_tmp4_1 = nn.ConvTranspose2d(128, n_classes, 2, 2, 0)
        self.deconvert_tmp4_2 = nn.ReLU(inplace=True)
        self.deconvert_tmp4_3 = nn.Conv2d(n_classes,3, 3, 1, 1)
        self.deconvert_tmp4_4 = nn.ReLU(inplace=True)
              
        
        #GAU
        '''
        self.gau4 = GAU(2048,1024)
        self.gau3 = GAU(1024,512)
        '''
        #self.gau2 = GAU(512,256)
        #self.gau1 = GAU(256,64)   
        '''                                                   
        self.relu_gau4 = nn.ReLU(inplace=True)
        self.relu_gau3 = nn.ReLU(inplace=True)
        '''

    def forward(self, x):
        # Initial block
        x = self.pad(x)
        #x = self.in_block(x)
        
        x = self.base_conv1(x)
        x = self.base_bn1(x)
        x = self.base_relu(x)
        
        e1 = self.base_maxpool(x)
        # Encoder blocks
        #add attention                
        e1 = self.encoder1(e1)        
        #e1 = e1 + self.ds1(x)
        e2 = self.encoder2(e1)        
        #e2 = e2 + self.ds2(e1)
        e3 = self.encoder3(e2)        
        #e3 = e3 + self.ds3(e2)
        e4 = self.encoder4(e3)        
        #e4 = e4 + self.ds4(e3)

        
        #Encoder no modify
        #e1 = self.encoder1(x)
        #e2 = self.encoder2(e1)
        #e3 = self.encoder3(e2)
        #e4 = self.encoder4(e3)
        
        #modify2 (add edge guidance)
        x_1 = self.conv3(x)
        e1_1 = self.conv4(e1)
        ex = torch.cat([x_1,e1_1],1)
        ex = self.conv5(ex)
        ex = self.conv6(ex)
        edge = self.tp_conv3(ex)
        edge = self.tp_conv4(edge)
        edge_future = edge
        #edge = self.tp_conv5(edge)        
        
        #edge = self.edge_conv1(edge)
        #edge = self.edge_conv2(edge)
        edge = self.edge_conv3(edge)
        edge = self.edge_lsm(edge)
        edge = edge[:,
                    :,
                    self.pad_size[2]:(edge.shape[2]-self.pad_size[2]),
                    self.pad_size[3]:(edge.shape[3]-self.pad_size[3])]
        #modify2 end
        
        #Decoder add attention
        #ag4 = self.gau4(e4,e3)
        #d4 = self.relu(self.decoder4(e4)+ag4)
        #ag3 = self.gau3(d4,e2)
        #d3 = self.relu(self.decoder3(d4)+ag3)
        #ag2 = self.gau2(d3,e1)
        #d2 = self.relu(self.decoder2(d3)+ag2)
        #ag1 = self.gau1(d2,x)
        #d1 = self.relu(self.decoder1(d2)+ag1)

        #mutil output
        # Decoder blocks
        d4 = e4
        #d3 = self.relu_gau4(self.decoder4(e4) + self.gau4(e4,e3))
        #d2 = self.relu_gau3(self.decoder3(d3) + self.gau3(d3,e2))
        d3 = e3 + self.decoder4(e4)
        d2 = e2 + self.decoder3(d3)
        #d1 = e1 + self.decoder2(d3)
        #d0 = x + self.decoder1(d2)
        d4_0 = self.convert4(d4)
        d4_1 = self.up4(d4_0)
        d4_2 = self.deconvert4_0(d4_1)
        d4_2 = self.deconvert4_1(d4_2)
        d4_2 = self.deconvert4_2(d4_2)
        # d4_3 = self.up_tmp4(d4_2 + edge_future)
        d4_3 = self.up_tmp4(self.FI_4(edge_future,d4_2))
        #d4_4 = self.deconvert_tmp4_0(d4_3)
        #d4_5 =  self.convert_tmp4(d4_4)
        d4_6 = self.deconvert_tmp4_1(d4_3)
        d4_6 = self.deconvert_tmp4_2(d4_6)
        coarse_3 = self.deconvert_tmp4_3(d4_6)
        coarse_3 = self.deconvert_tmp4_4(coarse_3)
        
        d3_0 = self.convert3(d3)
        d3_1 = self.up3(d3_0)
        d3_2 = self.deconvert3_0(d3_1)
        d3_2 = self.deconvert3_1(d3_2)
        # d3_3 = self.up_tmp3(d3_2 + edge_future)
        d3_3 = self.up_tmp3(self.FI_3(edge_future,d3_2))
        #d3_4 = self.deconvert_tmp3_0(d3_3)
        #d3_5 = self.convert_tmp3(d3_4)
        d3_6 = self.deconvert_tmp3_1(d3_3)
        d3_6 = self.deconvert_tmp3_2(d3_6)
        coarse_2 = self.deconvert_tmp3_3(d3_6)
        coarse_2 = self.deconvert_tmp3_4(coarse_2)
        
        d2_0 = self.convert2(d2)
        d2_1 = self.up2(d2_0)
        d2_2 = self.deconvert2_0(d2_1)
        # d2_3 = self.up_tmp2(d2_2 + edge_future)
        d2_3 = self.up_tmp2(self.FI_2(edge_future,d2_2))
        #d2_4 = self.deconvert_tmp2_0(d2_3)
        #d2_5 = self.convert_tmp2(d2_4)
        d2_6 = self.deconvert_tmp2_1(d2_3)
        d2_6 = self.deconvert_tmp2_2(d2_6)
        coarse_1 = self.deconvert_tmp2_3(d2_6)
        coarse_1 = self.deconvert_tmp2_4(coarse_1)
      
       
        d =  d2_3 + d3_3 + d4_3            
        #y = self.tp_conv1(y)
        #y = self.conv2(y)
        y = self.tp_conv2(d)
        y = self.lsm(y) #relu
        coarse = self.tp_conv2_1(y)
        coarse = self.lsm_1(coarse) #relu
        #y = y[:,
        #      :,
        #      self.pad_size[2]:(y.shape[2]-self.pad_size[2]),
        #      self.pad_size[3]:(y.shape[3]-self.pad_size[3])]



        return y, edge, coarse, coarse_1, coarse_2, coarse_3
    
    
class FastNet50(nn.Module):
    
    def __init__(self):
        
        super(FastNet50, self).__init__()
        self.trans = LinkNet50(n_classes=32)
        self.tanh=nn.Tanh()
        self.refine0= nn.Conv2d(3, 32, kernel_size=3,stride=1,padding=1)
        self.refine1= nn.Conv2d(67, 20, kernel_size=3,stride=1,padding=1)
        self.refine2= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1)
        self.threshold=nn.Threshold(0.1, 0.1)
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        self.upsample = F.upsample_nearest
        self.relu0=nn.LeakyReLU(0.2)
        self.relu1=nn.LeakyReLU(0.2)
        self.relu2=nn.LeakyReLU(0.2)
        self.relu3=nn.LeakyReLU(0.2)
        self.relu4=nn.LeakyReLU(0.2)
        self.relu5=nn.LeakyReLU(0.2)
        self.relu6=nn.LeakyReLU(0.2)
        
    def forward(self,I):

        t,edge,coarse,coarse_1,coarse_2,coarse_3 = self.trans(I)
        #print("edge:",edge.shape)
        #print("t:",t.shape)
        #edge = torch.clamp(edge,min=0,max=1)
        # Adapted from He Zhang https://github.com/hezhangsprinter/DCPDN
        # Bring I to feature space for concatenation
        I = self.relu0((self.refine0(I)))
        dehaze=torch.cat([t,I],1)
        #print("dehaze:",dehaze.shape)
        dehaze=torch.cat([dehaze,edge],1)
        dehaze=self.relu1((self.refine1(dehaze)))
        dehaze=self.relu2((self.refine2(dehaze)))
        shape_out = dehaze.data.size()
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(dehaze, 32)
        x1010 = F.avg_pool2d(dehaze, 32)
        x102 = F.avg_pool2d(dehaze, 16)
        x1020 = F.avg_pool2d(dehaze, 16)
        x103 = F.avg_pool2d(dehaze, 8)
        x104 = F.avg_pool2d(dehaze, 4)
        x1010 = self.upsample(self.relu3(self.conv1010(x101)),size=shape_out)
        x1020 = self.upsample(self.relu4(self.conv1020(x102)),size=shape_out)
        x1030 = self.upsample(self.relu5(self.conv1030(x103)),size=shape_out)
        x1040 = self.upsample(self.relu6(self.conv1040(x104)),size=shape_out)
        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
        dehaze= self.tanh(self.refine3(dehaze))
        return dehaze,t,edge,coarse,coarse_1,coarse_2,coarse_3
    
def swish(x):
    return x * F.sigmoid(x)



if __name__ == '__main__': 
    
   
    model = FastNet50()
    #file_path = "./resnet101.txt"
    #creat_file(file_path,model)
    #print(model)
    
    
