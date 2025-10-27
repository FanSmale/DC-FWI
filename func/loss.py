# -*- coding: utf-8 -*-
"""
@Time: 2024/6/28

@author: Zeng Zifei
"""
import torch
import torch.nn as nn
import numpy as np
from ParamConfig import *
import lpips
import torch.autograd as autograd

# 加载预训练的LPIPS模型
# verbose为false:使用LPIPS时删除通知
lpips_model = lpips.LPIPS(net="alex", verbose=False)

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")


# logcosh损失函数，用的np.cosh，没有梯度计算
def logcosh(pred, true):
    tensor_cuda = pred - true  # {Tensor:(,1,70,70)}
    # 将CUDA张量移动到CPU上
    tensor_cpu = tensor_cuda.cpu().detach().numpy()  # {ndarray:(,1,70,70)}
    loss = np.log(np.cosh(tensor_cpu))  # {ndarray:(,1,70,70)}
    loss_logcosh = np.mean(loss)  # {float32:()}
    return loss_logcosh


# 消融实验：只跑logcosh
# log_cosh 损失，用的torch
def logcosh3(pred, true):
    tensor_cuda = torch.abs(pred - true)  # {Tensor:(,1,70,70)}
    loss = torch.log(torch.cosh(tensor_cuda))  # {ndarray:(,1,70,70)}
    loss_logcosh = torch.mean(loss)  # {float32:()}
    return loss_logcosh


# 用的torch，针对未归一化的seg
def logcosh4(pred, true):
    x = pred - true  # {Tensor:(,1,201,301)}
    # s始终为正
    s = torch.sign(x) * x
    p = torch.exp(-2 * s)
    # log1p就是求log(1 + x)，即ln(x+1)，log1p的使用就像是将一个数据压缩到了一个区间，与数据的标准化类似
    # np.log这个对数函数是以e为底的对数函数
    log_cosh_x = s + torch.log1p(p) - torch.log(torch.tensor(2.0)).to(device)
    loss_logcosh = torch.mean(log_cosh_x)
    return loss_logcosh


# 0.3*mae + logcosh
def criterion1(pred, gt):
    l1loss = nn.L1Loss()
    loss_mae = l1loss(pred, gt)  # tensor(0.6831, device='cuda:0', grad_fn=<L1LossBackward0>)
    # loss_logcosh = logcosh(pred, gt)  # {ndarray:(,1,201,301)}  {float32:()}
    # loss_logcosh = logcosh3(pred, gt)
    loss_logcosh = logcosh4(pred, gt)  # {ndarray:(,1,201,301)}  {float32:()}
    loss = loss_weight * loss_mae + loss_logcosh  # tensor(0.4859, device='cuda:0', grad_fn=<AddBackward0>)
    return loss, loss_mae, loss_logcosh


# 0.3*mae + 0.7*logcosh
def criterion1_2(pred, gt):
    l1loss = nn.L1Loss()
    loss_mae = l1loss(pred, gt)  # tensor(0.6831, device='cuda:0', grad_fn=<L1LossBackward0>)
    # loss_logcosh = logcosh(pred, gt)  # {ndarray:(,1,201,301)}  {float32:()}
    # loss_logcosh = logcosh3(pred, gt)
    loss_logcosh = logcosh4(pred, gt)  # {ndarray:(,1,201,301)}  {float32:()}
    loss = loss_weight * loss_mae + (1-loss_weight) * loss_logcosh  # tensor(0.4859, device='cuda:0', grad_fn=<AddBackward0>)
    return loss, loss_mae, loss_logcosh


# 0.3*mae + 0.7*logcosh，返回loss
def criterion1_2_2(pred, gt):
    l1loss = nn.L1Loss()
    loss_mae = l1loss(pred, gt)  # tensor(0.6831, device='cuda:0', grad_fn=<L1LossBackward0>)
    # loss_logcosh = logcosh(pred, gt)  # {ndarray:(,1,201,301)}  {float32:()}
    # loss_logcosh = logcosh3(pred, gt)
    loss_logcosh = logcosh4(pred, gt)  # {ndarray:(,1,201,301)}  {float32:()}
    loss = loss_weight * loss_mae + (1-loss_weight) * loss_logcosh  # tensor(0.4859, device='cuda:0', grad_fn=<AddBackward0>)
    return loss

# mse
def criterion2(pred, gt):
    l2loss = nn.MSELoss()
    loss = l2loss(pred, gt)
    return loss


# mae
def criterion3(pred, gt):
    l1loss = nn.L1Loss()
    loss = l1loss(pred, gt)
    return loss


# Calculate the MAE + MSE
def criterion4(pred, gt):
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    loss_mae = l1loss(pred, gt)
    loss_mse = l2loss(pred, gt)
    loss = loss_mae + loss_mse
    return loss, loss_mae, loss_mse


l1loss = nn.L1Loss()
l2loss = nn.MSELoss()

def criterion_g(pred, gt, net_d=None):
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    loss_g1v = l1loss(pred, gt)
    loss_g2v = l2loss(pred, gt)
    loss = 100 * loss_g1v + 100 * loss_g2v
    if net_d is not None:
        loss_adv = -torch.mean(net_d(pred))
        loss += loss_adv
    return loss, loss_g1v, loss_g2v

class Wasserstein_GP(nn.Module):
    def __init__(self, device, lambda_gp):
        super(Wasserstein_GP, self).__init__()
        self.device = device
        self.lambda_gp = lambda_gp

    def forward(self, real, fake, model):
        gradient_penalty = self.compute_gradient_penalty(model, real, fake)
        loss_real = torch.mean(model(real))
        loss_fake = torch.mean(model(fake))
        loss = -loss_real + loss_fake + gradient_penalty * self.lambda_gp
        return loss, loss_real-loss_fake, gradient_penalty

    def compute_gradient_penalty(self, model, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = model(interpolates)
        # 计算梯度
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(real_samples.size(0), d_interpolates.size(1)).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
