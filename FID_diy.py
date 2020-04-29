'''
使用说明：
１．FID 计算，取的是最后一个池化层作为x的编码层，当然可以修改dims选取其他层；
２.分别传入真实数据和生成数据，数据的格式是传入的是值必须在（０，１）张量，（Ｎ，３，Ｈ，Ｗ）
３.首先将两组数据分别用calculate_activation_statistics得到多元正态的均值和协方差
４.最后用函数calculate_frechet_distance得到FID,这个指标是越小越好。
5.利用全体数据集估计真实数据的均值和协方差，然后在生成数据中任意挑选N副图片计算生成数据的均值和协方差。由于正态分布的假设，N要尽量大，ＦＩＤ才越准确，如N=50000。
6.ＦＩＤ要数据量大时才能使用，对于ＩＳ在每个ｂａｔｃｈｓｉｚｅ完成后就能计算，但ＦＩＤ必须要达到一定的量层方能计算。
7.由于传入的数据量一般较大，所以，ｂａｔｃｈｓｉｚｅ＝６４，一般是满足条件的。
8.在调用这个模型时，只需注意数据的范围是（０，１）即可，数组和张量都可以。
９.计算ＦＩＤ,只需要调用ＦＩＤ函数，直接将要计算的数据放入其中即可。
'''

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
from scipy import linalg
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d
import torchvision.datasets as dsets
from inception_diy import InceptionV3

dims = 2048 #64,192,768,2048,不同的值对应不同的feature map。即x 的　coding layer
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
#model = InceptionV3([block_idx])


def get_activations(images, model=InceptionV3([block_idx]), batch_size=64, dims=2048,
                    cuda=True, verbose=True):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.这里务必注意:传入的是值在（０，１）张量（Ｎ，３，Ｈ，Ｗ）
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()  # 模块进入评估模式

    d0 = images.shape[0]
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size  # 取整运算
    n_used_imgs = n_batches * batch_size
    pred_arr = torch.zeros((n_used_imgs,
                            dims))  # 原来的pred_arr = np.empty((n_used_imgs, dims))#我希望传入的是张量这里要修改成
    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        batch = images[
                start:end]  # 原来的batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
        batch = Variable(batch, volatile=True)
        if cuda:
            batch = batch.cuda()
            model = model.cuda()  # 这里是添加的，ｍｏｄｅｌ也要放置在ｃｕｄａ上。

        pred = model(batch)[
            0]  # pred产生的是列表，下面的操作对列表不适用，所以取出数据。

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.data.view(batch_size,
                                             -1).cpu()  # pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr
    # 经过上面修改，这里返回的就给张量。


def calculate_activation_statistics(images, model=InceptionV3([block_idx]), batch_size=64,
                                    dims=2048, cuda=True, verbose=True):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(images, model, batch_size, dims, cuda, verbose)  # 这个函数返回的是张量
    act = act.numpy()
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def FID(image1,image2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1,sigma1 = calculate_activation_statistics(image1, model=InceptionV3([block_idx]), batch_size=64,dims=2048, cuda=True, verbose=True)
    mu2,sigma2 = calculate_activation_statistics(image2, model=InceptionV3([block_idx]), batch_size=64,dims=2048, cuda=True, verbose=True)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2),
                              disp=False)  # 计算矩阵的平方根,注意有些是复数。
    if not np.isfinite(
            covmean).all():  # 判断是否是有限数，是否有无穷大。
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
    if np.iscomplexobj(
            covmean):  # 判断复数。
        if not np.allclose(np.diagonal(covmean).imag, 0,
                           atol=1e-3):  # np.diagonal(covmean).imag(real) 协方差矩阵对角元素的虚部(实部）。  np.allclose判断np.diagonal(covmean).imag,与 0之间的绝对误差在ａｔｏｌ内则放回真，否则假。
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    FID = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return FID

