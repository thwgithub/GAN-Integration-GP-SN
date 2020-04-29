
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.nn.modules import conv#包含所有的卷积模块
from torch.nn.modules.utils import _pair
from torch.nn.modules import Linear
from torch.nn.modules.batchnorm import _BatchNorm
import random
import argparse
from torch.nn.parameter import Parameter
from torch.nn import Module
######################求解最大奇异值，即谱范数########################################################
def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)#加入eps 是防止torch.norm(v)=0
def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    Ip is iteration step
    """
    #xp = W.data
    if not Ip >= 1:
        raise ValueError("Power iteration should be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1)) * _v)
    return sigma, _u
####################################################################################################

######################对卷积层做SN###################################################################
class SNConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)#_pair表示由stride组成的两个元素的一个元组，
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())

    @property#Python内置的@property装饰器就是负责把一个方法变成属性调用的
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
        return F.conv2d(input, self.W_, self.bias, self.stride,self.padding, self.dilation, self.groups)#self.W_能这样调用就是因为装饰器@property
########################################################################################################

####################对线性层做SN##########################################################################
class SNLinear(Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('u', torch.Tensor(1, out_features).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
       return F.linear(input, self.W_, self.bias)
#######################################################################################################

###########################SNBN########################################################################
class SNBN(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(SNBN, self).__init__(num_features, eps=1e-5, momentum=0.1, affine=True)
        self.register_buffer('u', torch.Tensor(1, num_features).normal_())

    @property
    def W_(self):
        w_mat = torch.diag(self.weight)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
       return F.batch_norm(input,self.running_mean, self.running_var,self.W_,self.bias,self.training,self.momentum,self.eps)
########################################################################################################


########################################################################################################
class SNConvT2d(conv._ConvTransposeMixin, conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(SNConvT2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)
        self.register_buffer('u', torch.Tensor(1, in_channels).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat,self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose2d(
            input, self.W_, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
#self.W_能这样调用就是因为装饰器@property
##########################################################################################################


##########################################################################################################
#关于嵌入说明：
#1.调用方式是：embedding=nn.SNEmbedding(n_classes,num_feature(embeding dimension))
#2           y=Variable(torch.LongTensor())
#            (1)y一定是一个长类型的变量，不能是cuda型。如果是其他类型要用y=y.type(torch.LongTensor)转换。
#            (2)y类别数一定要小于等于n_classes,否则会出错。即，如果n_classes=10,则y中的值（标签）只能是在（0,10）之间变化，
#             (3) num_feature可能自定义
#3            output= embedding(y)
#4            输出：如果y是一个一维的张量，则输出是batchsize（=len(y))*num_feature的一个二维张量（矩阵）。
#5                 如何y是一个二维的张量，则输出是2*batchsize*num_feature.详情可参考pytorch手册383page
#6                 输出类型是变量浮点型，有点情况可能要进行转化为cuda（）。
class SNEmbedding(Module):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False):
        super(SNEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.sparse = sparse

        self.reset_parameters()
        self.register_buffer('u', torch.Tensor(1, num_embeddings).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def forward(self, input):
        padding_idx = self.padding_idx
        if padding_idx is None:
            padding_idx = -1
        return self._backend.Embedding.apply(
            input, self.W_,
            padding_idx, self.max_norm, self.norm_type,
            self.scale_grad_by_freq, self.sparse
        )

    def __repr__(self):
        s = '{name}({num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
############################################################################################################


