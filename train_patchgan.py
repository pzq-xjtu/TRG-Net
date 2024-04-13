import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import numpy as np
from tensorboardX import SummaryWriter
from math import log, ceil
import torch
import scipy.io as sio
from os import path as osp

from Discriminators import PatchDiscriminator 
from ganloss import GANLoss
from Datasets import Dataset
from TRGNet import *
from options import parse_options, copy_opt_file

root_path = osp.abspath(osp.join(__file__, osp.pardir))
opt, args = parse_options(root_path)
from logger import Logger

_modes = ['train', 'val']

try:
    os.makedirs(opt['path']['models'])
except OSError:
    pass
try:
    os.makedirs(opt['path']['log'])
except OSError:
    pass



os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu_id']
log = Logger(opt['path']['experiments_root'])
logger = log.get_log()

copy_opt_file(__file__, opt['path']['experiments_root'])
copy_opt_file(args.opt, opt['path']['experiments_root'])

writer = SummaryWriter(opt['path']['log'])

def train_model(netG, netD, datasets, optimizerG, lr_schedulerG, optimizerD, lr_schedulerD):
    batch_size = {'train': opt['datasets']['batchSize'], 'val': 1}
    data_loader = {phase: DataLoader(datasets[phase], batch_size=batch_size[phase],
                                     shuffle=True, num_workers=int(opt['datasets']['workers']), pin_memory=True) for phase in _modes}
    num_data = {phase: len(datasets[phase]) for phase in _modes}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in _modes}
    step_img = {x: 0 for x in _modes}
    step = 0
    step_val = 0
    ifshow = 0
    Ganloss = GANLoss('lsgan').cuda()
    if opt['model']['use_rot_tv']:
        logger.info('*'*100)
        logger.info('Use rotTV!')
        lambda_tv = opt['train']['lambda_tv']
    else:
        logger.info('*' * 100)
        logger.info('Not use rotTV!')
        lambda_tv = 0
    logger.info('lambda_tv={}'.format(lambda_tv))
    for epoch in range(opt['train']['epoch'], opt['train']['niter']):
        mse_per_epoch = {x: 0 for x in _modes}
        tic = time.time()
        # train stage
        lrG = optimizerG.param_groups[0]['lr']
        lrD = optimizerD.param_groups[0]['lr']

        logger.info('lrG %f' % lrG)
        logger.info('lrD %f' % lrD)

        phase = 'val'
        if epoch % 1 == 0:
            for ii, data in enumerate(data_loader[phase]):
                if ii > 9:
                    break
                rain, norain = data[0].cuda(), data[1].cuda()
                with torch.no_grad():
                    ifshow = 0
                    O = rain
                    B = norain
                    fake, fake_R, fake_temp, R_temp, _ = netG(norain, rain, ifshow, ii + 200 * epoch)
                Ims1_climp = np.hstack(
                    (np.clip(fake_R[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy(), 0.0, 1.0),
                     np.clip((O-B)[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy(), 0.0, 1.0)))
                Ims2_climp = np.hstack((np.clip(fake[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy(), 0.0, 1.0),
                                        np.clip(O[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy(), 0.0, 1.0)))
                writer.add_image("val", np.vstack((Ims1_climp, Ims2_climp)), step_val, dataformats='HWC')
                step_val += 1

        phase = 'train'
        for ii, data in enumerate(data_loader[phase]):
            input_pair, gt_spa = data[0].cuda(), data[1].cuda()
            input_spa = input_pair

            # =========================================================
            # (1) Update Discriminator
            # =========================================================
            # train with real
            netD.train()
            netD.zero_grad()
            d_out_real = netD(input_spa)
            d_loss_real = Ganloss(d_out_real, target_is_real=True)

            # train with fake
            fake, R, fake_temp, R_temp, theta = netG(gt_spa, input_pair, ifshow, step)


            d_out_fake = netD(fake.detach())
            d_loss_fake = Ganloss(d_out_fake, target_is_real=False)

            if epoch < opt['train']['PreTrainstep']:
                d_out_fake_temp = netD(fake_temp.detach())
                d_loss_fake_temp = Ganloss(d_out_fake_temp, target_is_real=False)
                errD = d_loss_real + 0.5 * d_loss_fake + 0.5 * d_loss_fake_temp
                errD.backward()
                optimizerD.step()
            else:
                errD = d_loss_real + d_loss_fake
                errD.backward()
                optimizerD.step()
            # =========================================================
            # (2) Update Gen network
            # =========================================================
            if step % opt['train']['n_dis'] == 0:
                netG.train()
                netG.zero_grad()
                g_out_fake = netD(fake)
                g_loss_fake = Ganloss(g_out_fake, target_is_real=True)

                TV_theta = rotTV()
                R_TV_theta_loss = TV_theta(R, theta[:, -1].unsqueeze(1))
                R_temp_TV_theta_loss = TV_theta(R_temp, theta[:, -1].unsqueeze(1))

                errG = 0.5 * g_loss_fake + lambda_tv * R_TV_theta_loss
                if epoch < opt['train']['PreTrainstep']:
                    g_out_fake_temp = netD(fake_temp)
                    g_loss_fake_temp = Ganloss(g_out_fake_temp, target_is_real=True)

                    Resloss = torch.mean(torch.abs(fake - fake_temp))
                    errG = 0.5 * g_loss_fake + 0.5 * g_loss_fake_temp + 0.1 * Resloss + lambda_tv * R_temp_TV_theta_loss + lambda_tv * R_TV_theta_loss
                errG.backward()
                optimizerG.step()

            if ii % 1000 == 0:
                template = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>5d}/{:0>5d}, DLoss={:5.2e}, GLoss={:5.2e}'
                logger.info(template.format(epoch + 1, opt['train']['niter'], phase, ii, num_iter_epoch[phase], errD.item(),
                                      g_loss_fake.item()))
                writer.add_scalar('TotalDloss',
                                  d_loss_real.item() + d_loss_fake.item(), step)
                writer.add_scalar('Dloss', errD.item(), step)
                writer.add_scalar('drloss', d_loss_real.item(), step)
                writer.add_scalar('dfloss', d_loss_fake.item(), step)
                writer.add_scalar('gfloss', g_loss_fake.item(), step)
                writer.add_scalar('R_TV_theta_loss', R_TV_theta_loss.item(), step)
                ifshow = 1
            else:
                ifshow = 0
            step += 1
        mse_per_epoch[phase] /= (ii + 1)
        logger.info('{:s}: Loss={:+.2e}'.format(phase, mse_per_epoch[phase]))
        logger.info('-' * 100)

        logger.info('-' * 100)

        # adjust the learning rate
        lr_schedulerG.step()
        lr_schedulerD.step()
        # save model
        if (epoch+1) % 1 == 0:
            model_prefix = 'model_'
            save_path_model = os.path.join(opt['path']['models'], model_prefix + str(epoch + 1))
            torch.save({
                'epoch': epoch + 1,
                'step': step + 1,
                'step_img': {x: step_img[x] + 1 for x in _modes},
                'model_state_dict': netG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'lr_schedulerD_state_dict': lr_schedulerD.state_dict(),
                'lr_schedulerG_state_dict': lr_schedulerG.state_dict(),
            }, save_path_model)
            model_prefix_g = 'G_state_'
            save_path_model = os.path.join(opt['path']['models'], model_prefix_g + str(epoch + 1) + '.pt')
            torch.save(netG.state_dict(), save_path_model)
            model_prefix_d = 'D_state_'
            save_path_model = os.path.join(opt['path']['models'], model_prefix_d + str(epoch + 1) + '.pt')
            torch.save(netD.state_dict(), save_path_model)
        toc = time.time()
        logger.info('This epoch take time {:.2f}'.format(toc - tic))
    writer.close()
    logger.info('Reach the maximal epochs! Finish training')





def main():
    iniData = sio.loadmat("CofAll_11_new")
    A = iniData['A']
    B = iniData['B']
    A = A[:, 0:30]
    B = B[:, 0:30]
    netG = TRGNet(30, opt['model']['dictNum'], opt['model']['ker_num'], 11, opt['model']['theta_num'], A, B, writer, unpaired=False, path=opt['path']['experiments_root']).cuda()  # rain_generate network
    # ========================================================================
    # Using smaller lr for rain kernel
    all_params = netG.parameters()
    coef_params = []
    for pname, p in netG.named_parameters():
        if any([pname.endswith(k) for k in ['coef','coef1','coef2','coef3']]):
            logger.info('pname={}'.format(pname))
            coef_params += [p]
    params_id = list(map(id, coef_params))
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    optimizerG = optim.Adam([
        { 'params': other_params },
        { 'params': coef_params, 'lr': 0.1 * opt['train']['lrG']}
    ], lr=opt['train']['lrG'])
    # ===========================================================================
    schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, opt['train']['milestone'], gamma=0.2)

    netD = PatchDiscriminator(input_nc=3, path=opt['path']['experiments_root']).cuda()
    optimizerD = optim.Adam(netD.parameters(), lr=opt['train']['lrD'])
    schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, opt['train']['milestone'], gamma=0.2)


    train_dataset = Dataset(opt['datasets']['data_path'], opt['datasets']['gt_path'], opt['datasets']['patchSize'], opt['datasets']['batchSize'] * 3000, path=opt['path']['experiments_root'])
    val_dataset = Dataset(opt['datasets']['data_path'], opt['datasets']['gt_path'])
    datasets = {'train': train_dataset, 'val': val_dataset}
    # train model
    train_model(netG, netD, datasets, optimizerG, schedulerG, optimizerD, schedulerD)


if __name__ == '__main__':
    main()

