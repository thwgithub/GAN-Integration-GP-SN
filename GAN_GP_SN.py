
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
from torch.nn.utils.clip_grad import *
from torch.autograd import grad
import os
import matplotlib.pyplot as plt
from SN_diy import SNConv2d, SNConvT2d, SNLinear, SNBN
import time
import numpy as np
from PIL import Image
from OAdam_diy import OAdam

class Config:
    max_iterations = 100000 # =1 when debug
    D_ITERS = 1  # discriminator critic iters
    lrG = 0.0002
    lrD = 0.0002
    beta1 = 0.5
    beta2 = 0.9
    nz = 128  # noise dimension
    nc = 3  # chanel of img
    ngf = 64  # generate channel
    ndf = 64  # discriminative channels
    batchsize = 64
    workers = 0  # dataloader numworks
    cuda = True  # use gpu or not
    mg = 4  # feature condense size
    ch = 8 * ngf  # generator channel,512
    manualSeed = None
    labda = 10#for gradient penalty
    dataset = 'cifar10'  # celeba,LSUN,SVHN,STL10
    L = 0
    G_cn = 1
opt = Config()

#####################################Saving Path########################################################################
if opt.dataset == 'cifar10':
    save_dir = './Sparse_GAN/test24'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_loss_figure_dir = './Sparse_GAN/test24/save_loss'
    if not os.path.exists(save_loss_figure_dir):
        os.makedirs(save_loss_figure_dir)
    save_model_dir = './Sparse_GAN/test24/save_model'
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)


    transform = transforms.Compose(
        [
            # transforms.Resize(size=(re_size, re_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

    traindata = dsets.CIFAR10('./data_cifar10', transform=transform)
    dataloader = torch.utils.data.DataLoader(traindata,
                                             batch_size=opt.batchsize,
                                             shuffle=True, drop_last=True)
if opt.dataset == 'SVHN':
    save_dir = './Sparse_GAN/SVHN'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_loss_figure_dir = './Sparse_GAN/SVHN/save_loss'
    if not os.path.exists(save_loss_figure_dir):
        os.makedirs(save_loss_figure_dir)
    save_model_dir = './Sparse_GAN/SVHN/save_model'
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

    traindata = dsets.SVHN('./data_SVHN', split='train', transform=transform)
    dataloader = torch.utils.data.DataLoader(traindata,
                                             batch_size=opt.batchsize,
                                             shuffle=True, drop_last=True)
if opt.dataset == 'STL10':
    save_dir = './Sparse_GAN/STL10'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_loss_figure_dir = './Sparse_GAN/STL10/save_loss'
    if not os.path.exists(save_loss_figure_dir):
        os.makedirs(save_loss_figure_dir)
    save_model_dir = './Sparse_GAN/STL10/save_model'
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    transform = transforms.Compose(
        [
            transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    traindata = dsets.STL10('data_STL10', split='unlabeled', transform=transform)

    dataloader = torch.utils.data.DataLoader(traindata,
                                             batch_size=opt.batchsize,
                                             shuffle=True, drop_last=True)

if opt.dataset == 'celeba':
    save_dir = './Sparse_GAN/CelebA'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_loss_figure_dir = './Sparse_GAN/CelebA/save_loss'
    if not os.path.exists(save_loss_figure_dir):
        os.makedirs(save_loss_figure_dir)
    save_model_dir = './Sparse_GAN/CelebA/save_model'
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    transform = transforms.Compose(
        [
            transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    traindata = dsets.ImageFolder('data_celeba', transform=transform)

    dataloader = torch.utils.data.DataLoader(traindata,
                                             batch_size=opt.batchsize,
                                             shuffle=True, drop_last=True)

if opt.dataset == 'LSUN':
    save_dir = './Sparse_GAN/LSUN'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_loss_figure_dir = './Sparse_GAN/LSUN/save_loss'
    if not os.path.exists(save_loss_figure_dir):
        os.makedirs(save_loss_figure_dir)
    save_model_dir = './Sparse_GAN/LSUN/save_model'
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    transform = transforms.Compose(
        [
            transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    traindata = dsets.LSUN('data_lsun_bedroom',
                           classes=['bedroom_train'], transform=transform)
    dataloader = torch.utils.data.DataLoader(traindata,
                                             batch_size=opt.batchsize,
                                             shuffle=True, drop_last=True)

########################################################################################################################


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    torch.cuda.set_device(0)

cudnn.benchmark = True  # accelerating training
###########################损失函数图及训练时间统计#####################################################
train_hist = {}  # 建立一个空字典。
train_hist['D_loss'] = []  # 给每个字典赋空值。这个方法特别有用。
train_hist['G_loss'] = []
train_hist['per_epoch_time'] = []
train_hist['total_time'] = []
train_hist['D_gradient'] = []
train_hist['G_gradient'] = []

def loss_plot(hist, path):
    x1 = range(len(hist['D_loss']))
    x2 = range(len(hist['G_loss']))
    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x1, y1, label='D_loss')
    plt.plot(x2, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=0)  # 将图例放置的位置，4表示在右下方。参考《利用python进行数据分析P237》
    plt.grid(True)
    plt.tight_layout()  # 表示紧揍显示图片，居中显示。这个命令在多图显示中，特别有用。

    plt.savefig(path)  # 保存图像的路径。

    plt.close()

def loss_gradientplot(hist, path):
    x1 = range(len(hist['D_gradient']))
    x2 = range(len(hist['G_gradient']))
    y1 = hist['D_gradient']
    y2 = hist['G_gradient']

    plt.plot(x1, y1, label='D_gradient')
    plt.plot(x2, y2, label='G_gradient')

    plt.xlabel('Iter')
    plt.ylabel('gradient')

    plt.legend(loc=0)  # 将图例放置的位置，4表示在右下方。参考《利用python进行数据分析P237》
    plt.grid(True)
    plt.tight_layout()  # 表示紧揍显示图片，居中显示。这个命令在多图显示中，特别有用。

    plt.savefig(path)  # 保存图像的路径。

    plt.close()


########################################################################################################



##################################penalty function######################################################################
def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    # gradient penalty
    z = Variable(z, requires_grad=True).cuda()
    o = f(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
    g_norm = g.norm(p=2, dim=1) ** 2 - opt.L
    g_norm_size = g_norm.size(0)
    zero = Variable(torch.zeros(g_norm_size).cuda())
    gp = torch.max(zero, g_norm).mean()
    g_norm_D=g.norm(p=2, dim=1).mean()
    # gp = ((g.norm(p=2, dim=1) - 1)**2).mean()  #min(0,(g.norm(p=2, dim=1)**2-1)).mean()   #
    # gp = ((g.norm(p=2, dim=1)) ** 2).mean()
    return gp, g_norm_D


def gradient_penalty_G(z, f, g):
    # interpolation
    o = f(g(z))
    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
    g_norm_G = g.norm(p=2, dim=1).mean()
    return g_norm_G

########################################################################################################################


######################生成器#############################################################################
class netG(nn.Module):
    def __init__(self):
        super(netG, self).__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(opt.nz, opt.ch, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ch),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.ch, opt.ch // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ch // 2),
            nn.ReLU(),

            nn.ConvTranspose2d(opt.ch // 2, opt.ch // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ch // 4),
            nn.ReLU(),

            nn.ConvTranspose2d(opt.ch // 4, opt.ch // 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ch // 8),
            nn.ReLU(),

            nn.ConvTranspose2d(opt.ch // 8, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, out):
        out = self.deconv(out)

        return out


###########################################################################################################

######################判别器################################################################################
class netD(nn.Module):
    def __init__(self):
        super(netD, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            # SNConv2d()
            SNConv2d(opt.nc, opt.ndf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(opt.ndf, opt.ndf, 4, 2, 1, bias=False),
          #  nn.BatchNorm2d(opt.ndf),
            nn.LeakyReLU(0.2, inplace=True),

            SNConv2d(opt.ndf, opt.ndf * 2, 3, 1, 1, bias=False),
           # nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(opt.ndf * 2, opt.ndf * 2, 4, 2, 1, bias=False),
           # nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            SNConv2d(opt.ndf * 2, opt.ndf * 4, 3, 1, 1, bias=False),
           # nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(opt.ndf * 4, opt.ndf * 4, 4, 2, 1, bias=False),
           # nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # output,ndf*4 ,4,4
            SNConv2d(opt.ndf * 4, opt.ndf * 8, 3, 1, 1, bias=False),
            #nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            SNConv2d(opt.ndf * 8, 1, 4, 1, 0, bias=True)
        )

    def forward(self, input):
        output = self.main(input)
        output = output.view(output.size(0), -1)
        return output.squeeze()


##########################################################################################################

######################参数初始化############################################################################
def weight_filler(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d' or 'SNConv2d') != -1:
        nn.init.orthogonal(m.weight.data)
    if classname.find('SNConvT2d' or 'ConvTranspose2d') != -1:
        nn.init.xavier_normal(m.weight.data)

    elif classname.find('BatchNorm' or 'SNBN') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


#########################################################################################################

####################训练过程##############################################################################

netG = netG()
netD = netD()
print(netG)
print(netD)
netG.apply(weight_filler)
netD.apply(weight_filler)

if opt.cuda:
    netG.cuda()
    netD.cuda()

# optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
# optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))
optimizerG = OAdam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
optimizerD = OAdam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))

fixed_noisev = Variable(torch.randn(opt.batchsize, opt.nz, 1, 1). cuda(), volatile=True)

print('begin training, be patient')
start_time = time.time()
for iteration in range(opt.max_iterations):
    iteror = iter(dataloader)

    #############################################################
    # (1) Update netD network: maximize log(netD(x)) + log(1 - netD(netG(z)))
    ############################################################
    # train with real
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True
    for i in range(opt.D_ITERS):
        real_data = next(iteror)[0]
        input = Variable(real_data)
        noise = Variable(torch.randn(input.shape[0], opt.nz, 1, 1))
        if opt.cuda:
            input = input.cuda()
            noise = noise.cuda()
        netD.zero_grad()
        output_real = netD(input)
        D_realS = output_real.sigmoid_()
        # train with fake
        fake = netG(noise)
        output_fake = netD(fake.detach())  # 这个detach在这里尤其重要。如果不使用detach，则会引发错误。如果不用detach则要在下面的errD_fake.backward(retain_graph=True)加入True参数。
        D_fakeS = output_fake.sigmoid_()
        gp, gnorm = gradient_penalty(input.data, fake.data, netD)
        D_loss = -(D_realS.log().mean()) - (1 - D_fakeS).log().mean() + opt.labda * gp
        D_loss.backward()
        clip_grad_norm_(netD.parameters(),max_norm=0.01)
        optimizerD.step()
        train_hist['D_loss'].append(D_loss.data[0])
        train_hist['D_gradient'].append(gnorm.data[0])
    ############################
    # (2) Update netG network: minimize -log(netD(netG(z)))
    ###########################

    for p in netD.parameters():
        p.requires_grad = False
    netG.zero_grad()
    noise = Variable(torch.randn(input.shape[0], opt.nz, 1, 1))
    if opt.cuda:
        noise = noise.cuda()
    fake = netG(noise)
    G_loss = -torch.log(netD(fake).sigmoid_()).mean()
    G_loss.backward()
    clip_grad_norm_(netG.parameters(), max_norm=opt.G_cn)
    optimizerG.step()
    train_hist['G_loss'].append(G_loss.data[0])
    z=Variable(noise.data, requires_grad=True)
    gnormg=gradient_penalty_G(z, netD, netG)
    train_hist['G_gradient'].append(gnormg.data[0])

    print('[%d/%d] Loss_D: %f   Loss_G: %f '
          % (iteration, opt.max_iterations, D_loss.data[0], G_loss.data[0]))

    if (iteration + 1) % 50 == 0:
        vutils.save_image(real_data, '{0}/real_sample.png'.format(save_dir), normalize=True)
        fake = netG(fixed_noisev)
        vutils.save_image(fake.data, '{0}/fake_samples_epoch_{1}.png'.format(save_dir, iteration + 1), normalize=True)

########################################################################################################################
print("Generator Trains %d times with total time: %.2fs" % (opt.max_iterations, time.time() - start_time))

#######################################saving results###################################################################
loss_plot(train_hist, os.path.join(save_loss_figure_dir, 'lossfigure.png'))  # plot loss figure
np.save(os.path.join(save_loss_figure_dir, 'loss'), train_hist)  # 保存train loss data

loss_gradientplot(train_hist, os.path.join(save_loss_figure_dir, 'gradientfigure.png'))


torch.save(netG.state_dict(), '{0}/netG_iteration_{1}k.pkl'.format(save_model_dir, int(opt.max_iterations / 1000)))
torch.save(netD.state_dict(), '{0}/netD_iteration_{1}k.pkl'.format(save_model_dir, int(opt.max_iterations / 1000)))
######################################END###############################################################################