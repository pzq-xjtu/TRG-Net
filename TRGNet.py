from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

from options import copy_opt_file

class TRGNet(nn.Module):
    def __init__(self, channel_num, dict_num, kernel_num, kernel_size, theta_num, init_a, init_b, writer, unpaired=False, path=None):
        super(TRGNet, self).__init__()
        if path is not None:
            copy_opt_file(__file__, path)
        self.theta = GetTheta(theta_num)
        self.tau = GetParams(0.5)
        self.s_l = GetParams(1)
        self.s_w = GetParams(1)
        self.alpha = GetAlpha(dict_num, kernel_num)
        self.unpaired = unpaired
        self.writer = writer
        self.Encoder = Encoder(channel_num, kernel_num, kernel_size, init_a, init_b)

    def forward(self, B, X0, ifshow=0, step=0, logger=None):
        theta = self.theta(B.size(0))
        tau = self.tau(B.size(0))
        s_l = self.s_l(B.size(0))
        s_w = self.s_w(B.size(0))
        alpha = self.alpha(B.size(0))
        X, R, X_temp, R3, Mask, Rkern3 = self.Encoder(B, theta, s_l, s_w, tau, alpha, X0, ifshow, step, self.writer, unpaired=self.unpaired, logger=logger)

        return X, R, X_temp, R3, theta

    def show_factor(self, batchsize):
        theta = self.theta(batchsize)
        tau = self.tau(batchsize)
        s_l = self.s_l(batchsize)
        s_w = self.s_w(batchsize)
        alpha = self.alpha(batchsize)
        return theta, tau, s_l, s_w, alpha

    def test_factor(self, B, theta=None, tau=None, s_l=None, s_w=None, alpha=None, Z=None):
        if theta == None:
            theta = self.theta(B.size(0))
        if tau == None:
            tau = self.tau(B.size(0))
        if s_l == None:
            s_l = self.s_l(B.size(0))
        if s_w == None:
            s_w = self.s_w(B.size(0))
        if alpha ==None:
            alpha = self.alpha(B.size(0))
        X0 = 0
        ifshow = 0
        step = 0
        X, R, X_temp, R3, Mask, Rkern3 = self.Encoder(B, theta, s_l, s_w, tau, alpha, X0, ifshow, step, self.writer, unpaired=self.unpaired, Z=Z)

        return X, R, X_temp, R3, theta, Mask, Rkern3


class Encoder(nn.Module):
    def __init__(self, channel_num, kernel_num, kernel_size, init_a, init_b):
        super(Encoder, self).__init__()
        self.up = nn.Parameter(torch.ones([1, channel_num, 1, 1]) / 2, requires_grad=True)
        self.up2 = nn.Parameter(torch.ones([1, channel_num, 1, 1]), requires_grad=True)
        self.outadjust = nn.Parameter(torch.FloatTensor([0.1]), requires_grad=True)

        self.getBasis = GetBasis(kernel_size)
        temp_a = torch.FloatTensor(np.tile(np.expand_dims(init_a, axis=2), [1, 1, 3]))
        temp_b = torch.FloatTensor(np.tile(np.expand_dims(init_b, axis=2), [1, 1, 3]))
        weights = torch.cat([temp_a, temp_b], dim=0)  
        self.coef1 = nn.Parameter(weights, requires_grad=True)
        self.coef2 = nn.Parameter(weights, requires_grad=True)
        self.coef3 = nn.Parameter(weights, requires_grad=True)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.channel = channel_num
        self.relu = nn.ReLU()
        self.MapNet1 = MapNet1(channel_num, kernel_num, 3)
        self.MapNet2 = MapNet2(channel_num, kernel_num, 3)
        self.MapNet3 = MapNet3(channel_num, kernel_num, 3)
        self.MerNet = MerNet(channel_num, 3)

    def forward(self, B, theta, s_l, s_w, tau0, alpha, X0, ifshow, step, writer, unpaired=False, Z=None, logger=None):
        if Z == None:
            Z = torch.randn(B.size(0), self.kernel_num, B.size(2), B.size(3)).cuda()
        tau = tau0.unsqueeze(2).unsqueeze(3)
        M1 = self.relu(self.MapNet1(Z, theta)-tau)
        BasisC, BasisS, Mask = self.getBasis(theta, s_w, s_l)
        Basis = torch.cat([BasisC, BasisS], dim=4)
        if Basis.shape[1] == self.coef3.shape[1]:
            Dict1 = torch.einsum('bnijk,knm->bmnij', Basis, self.coef1)
        else:
            Dict1 = torch.einsum('bcijk,knm->bmnij', Basis, self.coef1)
        Rker1 = torch.einsum('bmnij,bnk->bmkij', Dict1, alpha)
        M1t = M1.reshape(1, M1.size(0) * M1.size(1), M1.size(2), M1.size(3))
        Rker1t = Rker1.reshape(Rker1.size(0) * Rker1.size(1), Rker1.size(2), Rker1.size(3), Rker1.size(4))
        R1 = F.conv2d(M1t, Rker1t, padding=self.kernel_size//2, groups=B.size(0))
        R1 = R1.reshape(B.size(0), B.size(1), B.size(2), B.size(3))

        M2 = self.relu(self.MapNet2(R1, theta)-tau)
        if Basis.shape[1] == self.coef3.shape[1]:
            Dict2 = torch.einsum('bnijk,knm->bmnij', Basis, self.coef2)
        else:
            Dict2 = torch.einsum('bcijk,knm->bmnij', Basis, self.coef2)
        Rker2 = torch.einsum('bmnij,bnk->bmkij', Dict2, alpha)
        M2t = M2.reshape(1, M2.size(0) * M2.size(1), M2.size(2), M2.size(3))
        Rker2t = Rker2.reshape(Rker2.size(0) * Rker2.size(1), Rker2.size(2), Rker2.size(3), Rker2.size(4))
        R2 = F.conv2d(M2t, Rker2t, padding=self.kernel_size//2, groups=B.size(0))
        R2 = R2.reshape(B.size(0), B.size(1), B.size(2), B.size(3))

        M3 = self.MapNet3(R2, theta) * self.outadjust
        if Basis.shape[1] == self.coef3.shape[1]:
            Dict3 = torch.einsum('bnijk,knm->bmnij', Basis, self.coef3)
        else:
            Dict3 = torch.einsum('bcijk,knm->bmnij', Basis, self.coef3)
        Rker3 = torch.einsum('bmnij,bnk->bmkij', Dict3, alpha)
        M3t = M3.reshape(1, M3.size(0) * M3.size(1), M3.size(2), M3.size(3))
        Rker3t = Rker3.reshape(Rker3.size(0) * Rker3.size(1), Rker3.size(2), Rker3.size(3), Rker3.size(4))
        R3 = F.conv2d(M3t, Rker3t, padding=self.kernel_size//2, groups=B.size(0))
        R3 = R3.reshape(B.size(0), B.size(1), B.size(2), B.size(3))
        X_temp = B + R3
        X, R = self.MerNet(R3, B, theta)

        return X, R, X_temp, R3, Mask, Rker3


class rotTV(nn.Module):
    def __init__(self):
        super(rotTV, self).__init__()
        self.getBasis_new = GetBasis(3)

        theta = torch.Tensor([[0]]).cuda()
        BasisC, BasisS, Mask = self.getBasis_new(theta, torch.ones_like(theta).cuda(), torch.ones_like(theta).cuda())

        kernel = torch.Tensor([[1., 1., 1.], [0., 0., 0.], [-1., -1., -1.]]).unsqueeze(2).cuda()
        a = torch.einsum('ij,ik->kj', BasisC[0, 0].reshape([9, 9]), kernel.reshape(9, 1)).unsqueeze(0)/9
        b = torch.einsum('ij,ik->kj', BasisS[0, 0].reshape([9, 9]), kernel.reshape(9, 1)).unsqueeze(0)/9
        weight = torch.cat([a, b], dim=2)
        self.weights = nn.Parameter(weight, requires_grad=False)
        self.loss = nn.L1Loss()

    def forward(self, x, theta):
        BasisC, BasisS, Mask = self.getBasis_new(theta, torch.ones_like(theta), torch.ones_like(theta))
        tempW = torch.einsum('tijk,smnk->tmnij', torch.cat([BasisC[:, 0], BasisS[:, 0]], dim=3), self.weights.unsqueeze(0))
        _filter = tempW.repeat(1,3,3,1,1).view(theta.size(0)*3, 3, 3, 3)
        x = x.view(1, x.size(0) * x.size(1), x.size(2), x.size(3))
        x = F.conv2d(x, _filter, stride=1, padding=0, groups=theta.size(0), dilation=1)
        return self.loss(x, torch.zeros_like(x))


class MapNet1(nn.Module):
    def __init__(self, hid_channel, channel, kernel_size):
        super(MapNet1, self).__init__()
        self.conv1 = nn.Conv2d(channel, hid_channel, kernel_size=1)
        self.rotResBlock1 = rotResBlock(hid_channel, kernel_size)
        self.rotResBlock2 = rotResBlock(hid_channel, kernel_size)
        self.conv2 = nn.Conv2d(hid_channel, channel, kernel_size=1)
        
    def forward(self, input, theta):
        output = self.conv1(input)
        output = self.rotResBlock1(output, theta)
        output = self.rotResBlock2(output, theta)
        output = self.conv2(output)
        return output


class MapNet2(nn.Module):
    def __init__(self, hid_channel, kernel_num, kernel_size):
        super(MapNet2, self).__init__()
        self.rotConv1 = rotConv(kernel_size, 3, hid_channel)
        self.bn = nn.BatchNorm2d(hid_channel)
        self.relu = nn.ReLU()
        self.rotResBlock = rotResBlock(hid_channel, kernel_size)
        self.conv2 = nn.Conv2d(hid_channel, kernel_num, kernel_size=1)
        
    def forward(self, input, theta):
        output = self.rotConv1(input, theta)
        output = self.bn(output)
        output = self.relu(output)
        output = self.rotResBlock(output, theta)
        output = self.conv2(output)
        return output


class MapNet3(nn.Module):
    def __init__(self, hid_channel, channel, kernel_size):
        super(MapNet3, self).__init__()
        self.conv1 = nn.Conv2d(3, hid_channel, kernel_size=1)
        self.rotResBlock1 = rotResBlock(hid_channel, kernel_size)
        self.rotResBlock2 = rotResBlock(hid_channel, kernel_size)
        self.conv2 = nn.Conv2d(hid_channel, channel, kernel_size=1)
        self.relu = nn.ReLU()
    def forward(self, input, theta):
        output = self.conv1(input)
        output = self.rotResBlock1(output, theta)
        output = self.rotResBlock2(output, theta)
        output = self.conv2(output)
        output = self.relu(output)
        return output


class MerNet(nn.Module):
    def __init__(self, channel_num, kernel_size):
        super(MerNet, self).__init__()
        self.getZ = rotConv(3, 6, 27)
        self.Res1 = rotResBlock(channel_num, kernel_size, .1)
        self.Res2 = rotResBlock(channel_num, kernel_size, .1)
        self.Res3 = rotResBlock(channel_num, kernel_size, .1)
        self.Res4 = rotResBlock(channel_num, kernel_size, .1)

    def forward(self, R, B, theta):
        Z = self.getZ.forward(torch.cat((R, B / 20), 1), theta)
        X = torch.cat((R, Z), 1)
        X = self.Res1.forward(X, theta)
        X = self.Res2.forward(X, theta)
        X = self.Res3.forward(X, theta)
        X = self.Res4.forward(X, theta)
        outR = X[:, [0, 1, 2], :, :]
        return outR + B, outR


class GetTheta(nn.Module):
    def __init__(self, channelNum):
        super(GetTheta, self).__init__()
        self.NLTras = nn.Sequential(ThetaNNLevel(1, 5),
                                    ThetaNNLevel(5, 5),
                                    ThetaNN_pi(5, channelNum))

    def forward(self, batchsize):
        e = torch.randn(batchsize, 1).cuda()
        theta = self.NLTras(e)
        return theta

class GetParams(nn.Module):
    def __init__(self, iniT, outC=1):
        super(GetParams, self).__init__()
        self.NLTras = nn.Sequential(ParamsNNLevel(1, 5), nn.ReLU(),
                                    ParamsNNLevel(5, 5), nn.ReLU(),
                                    ParamsNNLevel(5, outC))
        self.iniT = nn.Parameter(torch.FloatTensor([iniT]), requires_grad=True)

    def forward(self, batchsize):
        e = torch.randn(batchsize, 1).cuda()
        param = self.NLTras(e) + self.iniT
        return param

class GetAlpha(nn.Module):
    def __init__(self, ndict, nc):
        super(GetAlpha, self).__init__()
        self.Map = torch.arange(10).float().cuda().unsqueeze(0) / 10 + 0.05
        self.sig = nn.Parameter(torch.ones(1), requires_grad=True)
        self.NLTras = nn.Sequential(NNLevel(10, 10), nn.ReLU(),
                                    NNLevel(10, 10), nn.ReLU(),
                                    NNLevel(10, 10), nn.ReLU(),
                                    NNLevel(10, ndict * nc))
        self.nc = nc
        self.ndict = ndict

    def forward(self, batchsize):
        e = torch.randn(batchsize, 1).cuda()
        inCoef = torch.exp((self.Map - e) * (self.Map - e) * self.sig)
        Mat = self.NLTras(inCoef)
        Mat = Mat.reshape(Mat.size(0), self.ndict, self.nc)
        Mat = Mat / (torch.einsum('ijk->ik', Mat).unsqueeze(1))
        return Mat



class rotResBlock(nn.Module):
    def __init__(self, channels, sizeP=7, change=0.1):
        super(rotResBlock, self).__init__()
        self.Conv1 = rotConv(sizeP, channels, channels)
        self.BN1 = nn.BatchNorm2d(channels)
        self.ReLU = nn.ReLU()
        self.Conv2 = rotConv(sizeP, channels, channels)
        self.BN2 = nn.BatchNorm2d(channels)

        self.change = change

    def forward(self, X0, theta):
        X = self.Conv1.forward(X0, theta)
        X = self.BN1(X)
        X = self.ReLU(X)
        X = self.Conv2.forward(X, theta)
        X = self.BN2(X)
        X = X * self.change + X0
        return X

class rotConv(nn.Module):
    def __init__(self, sizeP, inChannel, outChannel):
        super(rotConv, self).__init__()
        self.getBasis = GetBasis(sizeP)
        self.c = nn.Parameter(torch.zeros(1, outChannel, 1, 1), requires_grad=True)
        self.sizeP = sizeP
        self.outChannel = outChannel
        weights = torch.cat([torch.randn(outChannel, sizeP * sizeP, inChannel) / sizeP / sizeP / 100,
                             torch.randn(outChannel, sizeP * sizeP, inChannel) / sizeP / sizeP / 100],
                            dim=1)
        self.weights = nn.Parameter(weights, requires_grad=True)

    def forward(self, X, theta):
        BasisC, BasisS, Mask = self.getBasis.forward(theta, 1 * torch.ones(1, 1).cuda(),
                                             1 * torch.ones(1, 1).cuda())
        Basis = torch.cat([BasisC, BasisS],dim=4)
        if Basis.shape[1] == self.weights.shape[2]:
            Ck = torch.einsum('bmijk,nkm->bnmij', Basis, self.weights)
        else:
            Ck = torch.einsum('bcijk,nkm->bnmij', Basis, self.weights)
        Ck = Ck.reshape(Ck.size(0) * Ck.size(1), Ck.size(2), Ck.size(3), Ck.size(4))
        X = X.reshape(1, X.size(0) * X.size(1), X.size(2), X.size(3))
        X = F.conv2d(X, Ck, padding=self.sizeP//2, groups=theta.size(0))
        X = X.reshape(theta.size(0), self.outChannel, X.size(2), X.size(3)) + self.c
        return X


class GetBasis(nn.Module):
    def __init__(self, sizeP, inP=None):
        super(GetBasis, self).__init__()
        if inP == None:
            inP = sizeP
        p = (sizeP - 1) / 2
        x = np.arange(-p, p + 1) / p
        X, Y = np.meshgrid(x, x)
        inX, inY, = torch.FloatTensor(X), torch.FloatTensor(Y)
        self.inX = inX.unsqueeze(0).unsqueeze(1).cuda()
        self.inY = inY.unsqueeze(0).unsqueeze(1).cuda()
        self.sizeP = sizeP
        self.inP = inP
        self.k = torch.FloatTensor(np.reshape(np.arange(self.inP), [1, 1, 1, 1, self.inP, 1])).cuda()
        self.l = torch.FloatTensor(np.reshape(np.arange(self.inP), [1, 1, 1, 1, 1, self.inP])).cuda()


    def forward(self, theta, s_w, s_l):
        theta = theta.unsqueeze(2).unsqueeze(3)
        s_w = s_w.unsqueeze(2).unsqueeze(3)
        s_l = s_l.unsqueeze(2).unsqueeze(3)
        X = torch.cos(theta) * self.inX - torch.sin(theta) * self.inY
        Y = torch.sin(theta) * self.inX + torch.cos(theta) * self.inY
        X = X * s_w
        Y = Y * s_l
        X = X.unsqueeze(4).unsqueeze(5)
        Y = Y.unsqueeze(4).unsqueeze(5)
        Mask = torch.exp(-torch.maximum(X ** 2 + Y ** 2 - 1, torch.Tensor([0]).cuda()) / 0.2)
        v = np.pi / self.inP * (self.inP - 1)
        p = self.inP / 2
        BasisC = torch.cos((self.k - self.inP * (self.k > p)) * v * X + (self.l - self.inP * (self.l > p)) * v * Y) * Mask
        BasisS = torch.sin((self.k - self.inP * (self.k > p)) * v * X + (self.l - self.inP * (self.l > p)) * v * Y) * Mask
        BasisC = torch.reshape(BasisC, [BasisC.shape[0], BasisC.shape[1], self.sizeP, self.sizeP,
                                        self.inP * self.inP])
        BasisS = torch.reshape(BasisS, [BasisS.shape[0], BasisS.shape[1], self.sizeP, self.sizeP,
                                        self.inP * self.inP])

        return BasisC, BasisS, Mask


class ThetaNNLevel(nn.Module):
    def __init__(self, m, n):
        super(ThetaNNLevel, self).__init__()
        self.w = nn.Parameter((torch.randn(n, m) / 1.5 + 1) / m / 2.5, requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1, n), requires_grad=True)

    def forward(self, X):
        X = torch.tensordot(X, self.w, ([1], [1])) + self.b
        X = nn.Tanh()(X) + X
        return X

class ThetaNN_pi(nn.Module):
    def __init__(self, m, n):
        super(ThetaNN_pi, self).__init__()
        self.w = nn.Parameter(torch.randn(n, m), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1, n), requires_grad=True)

    def forward(self, X):
        X = torch.tensordot(X, self.w, ([1], [1])) + self.b
        X = nn.Tanh()(X) * np.pi/2
        return X

class ParamsNNLevel(nn.Module):
    def __init__(self, m, n):
        super(ParamsNNLevel, self).__init__()
        self.w = nn.Parameter((torch.randn(n, m) +1) / m, requires_grad=True)
        self.b = nn.Parameter(torch.ones(1, n) * (-0.1), requires_grad=True)

    def forward(self, input):
        output = torch.tensordot(input, self.w, ([1], [1])) + self.b
        return output

class NNLevel(nn.Module):
    def __init__(self, m, n):
        super(NNLevel, self).__init__()
        self.w = nn.Parameter((torch.randn(n, m) / 3 + 1) / m, requires_grad=True)
        self.b = nn.Parameter(torch.ones(1, n) * (-0.1), requires_grad=True)

    def forward(self, input):
        output = torch.tensordot(input, self.w, ([1], [1])) + self.b
        return output