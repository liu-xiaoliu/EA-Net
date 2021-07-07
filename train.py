import os
os.environ['CUDA_VISIBLE_DEVICES']="0,5,6,7"
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

# internal libraries
#from models.models import LinkNet,FastNet,FastNet50,DualFastNet,Discriminator,ContentLoss
from models.models import FastNet50,ContentLoss
from hezhang_dataset import HeZhangDataset
from ntire_dataset import NTIREDataset
#blur
from utils_1 import blur

# dependencies
from pytorch_ssim import ssim
from pytorch_msssim import MSSSIM

# Load config file 
opt_file = open(sys.argv[1], "r")
opt = json.load(opt_file)
opt_file.close()

# Verify weights directory exists, if not create it
if not os.path.isdir(opt['weights_path']):
    os.makedirs(opt['weights_path'])

print('Running with following config:')
print(json.dumps(opt, indent=4, sort_keys=True))

# Hyper-Parameters
num_epochs = opt['num_epochs']
learning_rate = opt['learning_rate']

# Mode - choose and load the appropriate model
MODE = opt['mode'].upper()
device = torch.device(opt['device'])
if MODE == 'TRANS':
    model = LinkNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    try:
        model.load_state_dict(torch.load(sys.argv[2]))
    except Exception as e:
        print("No weights. Training from scratch.")
elif MODE == 'ATMOS':
    model = LinkNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    try:
        model.load_state_dict(torch.load(sys.argv[2]))
    except Exception as e:
        print("No weights. Training from scratch.")
elif MODE == 'FAST50':
    model = FastNet50().to(device)
    '''
    for param in model.trans.in_block.parameters():
        param.requires_grad = False
    for param in model.trans.encoder1.parameters():
        param.requires_grad = False
    for param in model.trans.encoder2.parameters():
        param.requires_grad = False
    for param in model.trans.encoder3.parameters():
        param.requires_grad = False
    for param in model.trans.encoder4.parameters():
        param.requires_grad = False
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    try:
        model.load_state_dict(torch.load(sys.argv[2]))
    except Exception as e:
        print("No weights. Training from scratch.")
elif MODE == 'FAST':
    model = FastNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    try:
        model.load_state_dict(torch.load(sys.argv[2]))
    except Exception as e:
        print("No weights. Training from scratch.")
elif MODE == 'DUAL' or ( MODE == 'GAN' and len(opt['loss_discr']) ):
    model = DualFastNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    try:
        model.trans.load_state_dict(torch.load(sys.argv[2]))
        model.atmos.load_state_dict(torch.load(sys.argv[3]))
        '''
        for param in model.trans.parameters():
            param.requires_grad = False
        for param in model.atmos.parameters():
            param.requires_grad = False
        '''
    except Exception as e:
        try:
            model.load_state_dict(torch.load(sys.argv[2]))
        except Exception as e:
            print("No weights. Training from scratch.")
    if MODE == 'GAN':
        model_d = Discriminator().to(device)
        optimizer_d = torch.optim.Adam(model_d.parameters(), lr=learning_rate)
        try:
            model_d.load_state_dict(torch.load(sys.argv[3]))
            if opt['parallel']:
                model_d = nn.DataParallel(model_d)
        except Exception as e:
            print("No weights. Training from scratch discrim.")
else:
    print('MODE INCORRECT : TRANS or ATMOS or FAST or DUAL or GAN')
    exit()

# Wrap in Data Parallel for multi-GPU use
if opt['parallel']:
    model = nn.DataParallel(model)

# Set default early stop, if not defined
if not 'early_stop' in opt:
    opt['early_stop'] = 100

# Loss
def sdim(output,target):
    return (1.-ssim(output,target))/2.

def mssdim(output,target):
    return (1.-MSSSIM(output,target))/2.

def psnr(output, target):
    mse_criterion = nn.MSELoss()
    mse = mse_criterion(output, target)
    psnr = 10 * torch.log10(1.0 / mse)
    return (1.-psnr)
#color loss
def ColorLoss(output, target):
    output_blur = blur(output)
    target_blur = blur(target)
    mse_criterion = nn.MSELoss()
    colorloss = mse_criterion(output_blur, target_blur)
    #square = torch.sum(torch.pow(output_blur - target_blur,2))
    return colorloss

    
def cross_entropy_loss_RCF(prediction, label):
    label = label.long()
    mask = label.float()
    #print(label)
    num_positive = torch.sum((mask==1).float()).float()
    num_negative = torch.sum((mask==0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost)

#criterion = {'L1':nn.L1Loss(),'MSE':nn.MSELoss(),'BCE':nn.BCELoss(),'Huber':nn.SmoothL1Loss(),'SSIM':sdim,'MSSSIM':mssdim,'CONTENT':ContentLoss(device),'PSNR'#:psnr,'COLOR':ColorLoss,'EDGE':cross_entropy_loss_RCF}
criterion = {'L1':nn.L1Loss(),'MSE':nn.MSELoss(),'BCE':nn.BCELoss(),'Huber':nn.SmoothL1Loss(),'SSIM':sdim,'MSSSIM':mssdim,'CONTENT':ContentLoss(),'PSNR':psnr,'COLOR':ColorLoss,'EDGE':cross_entropy_loss_RCF}
trans_loss = [x.upper() for x in opt['loss_trans']]
atmos_loss = [x.upper() for x in opt['loss_atmos']]
image_loss = [x.upper() for x in opt['loss_image']]
dhaze_loss = [x.upper() for x in opt['loss_dhaze']]
discr_loss = [x.upper() for x in opt['loss_discr']]
# color loss
color_loss = [x.upper() for x in opt['loss_color']]
color_criterion = [ criterion[x] for x in color_loss ]
#add edge loss
edge_loss = [x.upper() for x in opt['loss_edge']]
edge_criterion = [ criterion[x] for x in edge_loss ]
#
trans_criterion = [ criterion[x] for x in trans_loss ]
atmos_criterion = [ criterion[x] for x in atmos_loss ]
image_criterion = [ criterion[x] for x in image_loss ]
dhaze_criterion = [ criterion[x] for x in dhaze_loss ]
discr_criterion = [ criterion[x] for x in discr_loss ]

# Train Dataset
if opt['dataset'].upper() == 'NTIRE':
    train_dataset = NTIREDataset(opt)
else:
    train_dataset = HeZhangDataset(opt)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=opt['batch_size'], 
                                           shuffle=True)

# Validation Dataset and Setup
if opt['validate']:
    val_opt_file = open(opt['validation_config'], "r")
    val_opt = json.load(val_opt_file)
    val_opt_file.close()
    if val_opt['dataset'].upper() == 'NTIRE':
        validation_dataset = NTIREDataset(val_opt)
    else:
        validation_dataset = HeZhangDataset(val_opt)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                    batch_size=opt['batch_size'])
    validation_loss = [x.upper() for x in val_opt['loss_image']]
    validation_criterion = [ criterion[x] for x in validation_loss ]


total_step = len(train_loader)    
best_loss = np.inf
early_stop = 0

if 'pad' in opt:
    pad = nn.ReflectionPad2d((0,0,opt['pad'][0],opt['pad'][1])).to(device)
    crop = nn.ReflectionPad2d((0,0,-opt['pad'][0],-opt['pad'][1])).to(device)

# Training Loop
for epoch in range(num_epochs):
    epoch_loss = 0
    latest_msg = ''
    model.train()

    for i, (haze,image,image_trans,image_atmos,image_edge) in enumerate(train_loader):
        haze = haze.to(device)
        image_edge = image_edge.to(device)
        if 'pad' in opt:
            haze = pad(haze)
        loss_msg = ''       

        # copy required data to device
        if len(image_loss):
            image = image.to(device)
        if len(trans_loss):
            image_trans = image_trans.to(device)
        if len(atmos_loss):
            image_atmos = image_atmos.to(device)

        # Train transmission map network
        if MODE == 'TRANS':
            output = model(haze)
            if 'pad' in opt:
                output = crop(output)
            loss = sum([ c(output, image_trans) for c in trans_criterion ])
            loss_msg += ' Trans Loss : {:.4f}'.format(loss.item())

        # Train atmospheric light estimation network
        elif MODE == 'ATMOS':
            output = model(haze)
            if 'pad' in opt:
                output = crop(output)
            loss = sum([ c(output, image_atmos) for c in atmos_criterion ])
            loss_msg += ' Atmos Loss : {:.4f}'.format(loss.item())


        # Train network with 1 LinkNet and no physics model
        elif MODE == 'FAST' or MODE == 'FAST50':
            '''
            output,ft = model(haze)
            if 'pad' in opt:
                output = crop(output)
            ilosses = [ c(output, image)*w for c,w in zip(image_criterion,opt['loss_image_w'])]
            iloss = sum(ilosses)
            loss = iloss
            if len(image_loss):
                loss_msg += ' Total I: {:.4f}, '.format(iloss.item())
                for idx, l in enumerate(opt['loss_image']):
                    if idx == len(opt['loss_image'])-1:
                        loss_msg += l + ': {:.4f} '.format(ilosses[idx].item())
                    else:
                        loss_msg += l + ': {:.4f}, '.format(ilosses[idx].item())
            '''
            output,ft,edge,coarse,coarse_1,coarse_2,coarse_3  = model(haze)
            if 'pad' in opt:
                output = crop(output)
                edge = crop(edge)
            ilosses = [c(output, image)*w for c,w in zip(image_criterion,opt['loss_image_w'])]
            iloss = sum(ilosses)
            # co_ilosses = [c(coarse, image)*w for c,w in zip(image_criterion,opt['loss_image_w'])]
            # co_iloss = sum(co_ilosses)
            # co_ilosses_1 = [c(coarse_1, image)*w for c,w in zip(image_criterion,opt['loss_image_w'])]
            # co_iloss_1 = sum(co_ilosses_1)
            # co_ilosses_2 = [c(coarse_2, image)*w for c,w in zip(image_criterion,opt['loss_image_w'])]
            # co_iloss_2 = sum(co_ilosses_2)
            # co_ilosses_3 = [c(coarse_3, image)*w for c,w in zip(image_criterion,opt['loss_image_w'])]
            # co_iloss_3 = sum(co_ilosses_3)
            closs = sum([ c(output, image)*w for c,w in zip(color_criterion,opt['loss_color_w'])])
            eloss = sum([ c(edge, image_edge)*w for c,w in zip(edge_criterion,opt['loss_edge_w'])])
            # loss = iloss + closs + eloss + co_iloss + co_iloss_1 + co_iloss_2 + co_iloss_3
            loss = iloss + closs + eloss
            if len(color_loss):
                loss_msg += ' C: {:.4f},'.format(closs.item())
            if len(edge_loss):
                loss_msg += ' E: {:.4f},'.format(eloss.item())
            if len(image_loss):
                loss_msg += ' Total I: {:.4f}, '.format(iloss.item())
                for idx, l in enumerate(opt['loss_image']):
                    if idx == len(opt['loss_image'])-1:
                        loss_msg += l + ': {:.4f} '.format(ilosses[idx].item())
                    else:
                        loss_msg += l + ': {:.4f} '.format(ilosses[idx].item())
            # if len(image_loss):
                # loss_msg += ' Total I-coarse: {:.4f}, '.format(co_iloss.item())
            # if len(image_loss):
                # loss_msg += ' Total I-coarse1: {:.4f}, '.format(co_iloss_1.item())
            # if len(image_loss):
                # loss_msg += ' Total I-coarse2: {:.4f}, '.format(co_iloss_2.item())
            # if len(image_loss):
                # loss_msg += ' Total I-coarse3: {:.4f} '.format(co_iloss_3.item())


        # Train full network (light atmospheric estimation, transmission map, dehazed image; with or without GAN loss)
        elif MODE == 'DUAL' or MODE == 'GAN':
            output,trans,atmos,dehaze = model(haze)
            if 'pad' in opt:
                output = crop(output)
                trans = crop(trans)
                atmos = crop(atmos)
                dehaze = crop(dehaze)
            tloss = sum([ c(trans, image_trans)*w for c,w in zip(trans_criterion,opt['loss_trans_w'])])
            aloss = sum([ c(atmos, image_atmos)*w for c,w in zip(atmos_criterion,opt['loss_atmos_w'])])
            dloss = sum([ c(dehaze, image)*w for c,w in zip(dhaze_criterion,opt['loss_dhaze_w'])])
            ilosses = [ c(output, image)*w for c,w in zip(image_criterion,opt['loss_image_w'])]
            iloss = sum(ilosses)
            loss = iloss + tloss + aloss + dloss
            if len(trans_loss):
                loss_msg += ' T : {:.4f},'.format(tloss.item())
            if len(atmos_loss):
                loss_msg += ' A : {:.4f},'.format(aloss.item())
            if len(dhaze_loss):
                loss_msg += ' J : {:.4f},'.format(dloss.item())
            if len(image_loss):
                loss_msg += ' Total I: {:.4f}, '.format(iloss.item())
                for idx, l in enumerate(opt['loss_image']):
                    if idx == len(opt['loss_image'])-1:
                        loss_msg += l + ': {:.4f}, '.format(ilosses[idx].item())
                    else:
                        loss_msg += l + ': {:.4f}, '.format(ilosses[idx].item())
            if MODE == 'GAN':
                ones_const = Variable(torch.ones(image.shape[0], 1)).to(device)
                target_real = Variable(torch.rand(image.shape[0],1)*0.7 + 0.5).to(device)
                target_fake = Variable(torch.rand(image.shape[0],1)*0.3).to(device)
                real = Variable(image)                
                dloss_r = sum([ c(model_d(image), target_real)*w for c,w in zip(discr_criterion,opt['loss_discr_w'])])
                dloss_f = sum([ c(model_d(Variable(output)), target_fake)*w for c,w in zip(discr_criterion,opt['loss_discr_w'])])
                loss_d = dloss_r + dloss_f
                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()
                gloss = sum([ c(model_d(Variable(output)), ones_const)*w for c,w in zip(discr_criterion,opt['loss_discr_w'])])
                loss_msg += ' advs : {:.4f}'.format(gloss.item())
                loss_msg += ' dreal : {:.4f}'.format(dloss_r.item())
                loss_msg += ' dfake : {:.4f}'.format(dloss_f.item())
                loss += gloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        latest_msg = "Epoch: [{}/{}], Step: [{}/{}], Avg Epoch Loss: {:.4f}, Loss: {:.4f},".format(epoch+1, num_epochs, i+1, total_step, epoch_loss/(i+1), loss.item())+loss_msg
        print(latest_msg)
    early_stop += 1

    # Test model on validation dataset [only supported for FastNet, FastNet50, and DualFastNet]
    if opt['validate']:
        vloss = 0
        vcloss = 0
        veloss = 0        
        vvmsg = ''
        model.eval()
        with torch.no_grad():
            for j, (vhaze,vimage,_,_,vedge) in enumerate(validation_loader):
                vhaze = vhaze.to(device)
                vimage = vimage.to(device)
                vedge = vedge.to(device)
                vmsg = ''
                if 'pad' in opt:
                    vhaze = pad(vhaze)
                voutput = model(vhaze)[0]
                voutput_edge = model(vhaze)[2]
                
                if 'pad' in opt:
                    voutput = crop(voutput)
                vlosses = [c(voutput, vimage)*w for c,w in zip(validation_criterion,val_opt['loss_image_w'])]
                vloss += sum(vlosses)
                #add color and edge loss
                vcloss = sum([ c(voutput, vimage)*w for c,w in zip(color_criterion,val_opt['loss_color_w'])])
                veloss = sum([ c(voutput_edge, vedge)*w for c,w in zip(edge_criterion,val_opt['loss_edge_w'])])
                epoch_loss = vloss+vcloss+veloss
                '''
                vmsg = "Validation Loss: {:.4f}, ".format(epoch_loss.item())
                vmsg = "edge Loss: {:.4f}, ".format(epoch_loss.item())
                vmsg = "color Loss: {:.4f}, ".format(epoch_loss.item())
                vmsg = "image Loss: {:.4f}, ".format(epoch_loss.item())
                '''
                if len(val_opt['loss_color']):
                        vmsg += ' VC: {:.4f}, '.format(vcloss.item())
                if len(val_opt['loss_edge']):
                        vmsg += ' VE: {:.4f}, '.format(veloss.item())
                for idx, l in enumerate(val_opt['loss_image']):
                    if idx == len(val_opt['loss_image'])-1:
                        vmsg += l + ': {:.4f} '.format(vlosses[idx].item()/val_opt['loss_image_w'][idx])
                    else:
                        vmsg += l + ': {:.4f} '.format(vlosses[idx].item()/val_opt['loss_image_w'][idx])
            vvmsg = "Validation Loss: {:.4f}, ".format(epoch_loss.item())+ vmsg            
#            epoch_loss = vloss
            #epoch_loss = vloss+vcloss+veloss
        latest_msg = latest_msg + ", " + vvmsg           
        print(vvmsg)

    # Save weights and JSON logs
    if epoch_loss < best_loss:
        early_stop = 0
        best_loss = epoch_loss
        if opt['parallel']:
            torch.save(model.module.state_dict(), opt['weights_path'] + "/" + MODE + "_" + str(epoch) + ".ckpt")
        else:
            torch.save(model.state_dict(), opt['weights_path'] + "/" + MODE + "_" + str(epoch) + ".ckpt")
        if MODE == 'GAN':
            if opt['parallel']:
                torch.save(model_d.module.state_dict(), opt['weights_path'] + "/" + MODE + "_D_" + str(epoch) + ".ckpt")
            else:
                torch.save(model_d.state_dict(), opt['weights_path'] + "/" + MODE + "_D_" + str(epoch) + ".ckpt")
        if opt['log'] == 1:
            with open(opt['weights_path'] + "/" + MODE + "_" + str(epoch) + ".json", "w") as js:
                json.dump(dict(msg.split(':') for msg in latest_msg.replace(" ","").split(',')), js, indent=4, separators=(',',': '))

    # Stop early if loss is not improving
    if early_stop == opt['early_stop']:
        print("Loss has not improved in {} epochs. Stopping early.".format(opt['early_stop']))
        break
