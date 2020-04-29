import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets as dsets
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import random
from torch.autograd import grad
import os
import matplotlib.pyplot as plt
from SN_diy import SNConv2d, SNConvT2d, SNLinear, SNBN
import time
import numpy as np
from PIL import Image
from OAdam_diy import OAdam
#save_loss_figure_dir = './WGAN/WGANGP(author)(Bestpapermodel608)/save_loss'
save_loss_figure_dir_f = './GAN_instability/WGANGP(author)(Bestpapermodel608_2_cifar10)/save_loss'
save_loss_figure_dir_S= './GAN_instability/GAN-0GP-cifar10/save_loss'
save_loss_figure_dir_W= './GAN_instability/WGAN-GP-cifar10/save_loss'
save_loss_figure_dir_L = './GAN_instability/WGAN-LP-cifar10/save_loss'
Loss_f=np.load(os.path.join(save_loss_figure_dir_f, 'loss.npy'))
Loss_S=np.load(os.path.join(save_loss_figure_dir_S, 'loss.npy'))
Loss_W=np.load(os.path.join(save_loss_figure_dir_W, 'loss.npy'))
Loss_L=np.load(os.path.join(save_loss_figure_dir_L, 'loss.npy'))

Loss_f_list=Loss_f.tolist()
Loss_S_list=Loss_S.tolist()
Loss_W_list=Loss_W.tolist()
Loss_L_list=Loss_L.tolist()

Loss_f_list_D=Loss_f_list['D_loss']
Loss_f_list_G=Loss_f_list['G_loss']
Loss_f_list_DG=Loss_f_list['D_gradient']
Loss_f_list_GG=Loss_f_list['G_gradient']

Loss_S_list_D=Loss_S_list['D_loss']
Loss_S_list_G=Loss_S_list['G_loss']
Loss_S_list_DG=Loss_S_list['D_gradient']
Loss_S_list_GG=Loss_S_list['G_gradient']

Loss_W_list_D=Loss_W_list['D_loss']
Loss_W_list_G=Loss_W_list['G_loss']
Loss_W_list_DG=Loss_W_list['D_gradient']
Loss_W_list_GG=Loss_W_list['G_gradient']

Loss_L_list_D=Loss_L_list['D_loss']
Loss_L_list_G=Loss_L_list['G_loss']
Loss_L_list_DG=Loss_L_list['D_gradient']
Loss_L_list_GG=Loss_L_list['G_gradient']



x1_D = range(len(Loss_f_list_D))
x2_D = range(len(Loss_S_list_D))
x3_D = range(len(Loss_W_list_D))
x4_D = range(len(Loss_L_list_D))

y1_D = Loss_f_list_D
y1_D = list(np.array(y1_D)+0.5)
y2_D = Loss_S_list_D
y2_D = list(np.array(y2_D)+1.5)
y3_D = Loss_W_list_D
y4_D = Loss_L_list_D
y4_D = list(np.array(y4_D)-5)

x1_G = range(len(Loss_f_list_G))
x2_G = range(len(Loss_S_list_G))
x3_G = range(len(Loss_W_list_G))
x4_G = range(len(Loss_L_list_G))

y1_G= Loss_f_list_G
y2_G=Loss_S_list_G
y2_G = list(np.array(y2_G)+0.3)
y3_G=Loss_W_list_G
y3_G = list(np.array(y3_G)-10)
y4_G=Loss_L_list_G
y4_G = list(np.array(y4_G)-25)

xf_DG= range(len(Loss_f_list_DG))
yf_DG= Loss_f_list_DG
xf_GG= range(len(Loss_f_list_GG))
yf_GG= Loss_f_list_GG
yf_GG = list(np.array(yf_GG)+0.1)

xS_DG= range(len(Loss_S_list_DG))
yS_DG= Loss_S_list_DG
yS_DG=list(np.array(yS_DG))
xS_GG= range(len(Loss_S_list_GG))
yS_GG= Loss_S_list_GG
yS_GG = list(np.array(yS_GG)+0.05)

xW_DG= range(len(Loss_W_list_DG))
yW_DG= Loss_W_list_DG
xW_GG= range(len(Loss_W_list_GG))
yW_GG= Loss_W_list_GG
yW_GG= list(np.array(yW_GG)+2)

xL_DG= range(len(Loss_L_list_DG))
yL_DG= Loss_L_list_DG
yL_DG= list(np.array(yL_DG)-0.5)
xL_GG= range(len(Loss_L_list_GG))
yL_GG= Loss_L_list_GG



#


#
# # # Sigmoid loss
# plt.plot(x2_D, y2_D, label='GAN-0GP D_loss',color='lightblue')
# plt.plot(x2_G, y2_G, label='GAN-0GP G_loss',color='lightgreen')


#

# #Sigmoid gradient
# plt.plot(xS_DG, yS_DG, label='GAN_0GP D_average_gradient_norm',color='lightblue')
# plt.plot(xS_GG, yS_GG, label='GAN_0GP G_average_gradient_norm',color='lightgreen')

#WGAN-GP loss
#plt.plot(x3_D, y3_D, label='WGAN-GP', color='green')
#plt.plot(x3_G, y3_G, label='WGAN-GP',color='green')

#WGAN-GP gradient
plt.plot(xW_DG, yW_DG, label='WGAN-GP D_average_gradient_norm',color='green')
#plt.plot(xW_GG, yW_GG, label='WGAN-GP G_average_gradient_norm',color='green')

# WGAN-LP loss
#plt.plot(x4_D, y4_D, label='WGAN-LP',color='lightgreen')
#plt.plot(x4_G, y4_G, label='WGAN-LP',color='lightgreen')

# ## #f(x) loss
# plt.plot(x1_D, y1_D, label='Our D_loss',color='blue')
# plt.plot(x1_G, y1_G, label='Our G_loss',color='green')


#WGAN-LP gradient
plt.plot(xL_DG, yL_DG, label='WGAN-LP D_average_gradient_norm',color='lightgreen')
#plt.plot(xL_GG, yL_GG, label='WGAN-LP G_average_gradient_norm',color='lightgreen', alpha=1)

# #f(x) gradient
plt.plot(xf_DG, yf_DG, label='Our D_average_gradient_norm',color='blue')
# plt.plot(xf_GG, yf_GG, label='Our G_average_gradient_norm',color='green')
# #

plt.xlabel('Iter')
plt.ylabel('D_gradient')
#plt.ylim([-40,10])

plt.legend(loc=0)  # 将图例放置的位置，4表示在右下方。参考《利用python进行数据分析P237》
plt.grid(True)
plt.tight_layout()  # 表示紧揍显示图片，居中显示。这个命令在多图显示中，特别有用。

plt.savefig('./figure/cifar10D_gradient_our+WGANGP+WGANLP.eps')  # 保存图像的路径。

plt.close()
