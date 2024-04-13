import os
import torch.nn.parallel
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
from tqdm import tqdm
import datetime

from Discriminators import Discriminator
from Datasets import Dataset
from TRGNet import *
from options import parse_options, copy_opt_file
from logger import Logger

root_path = osp.abspath(osp.join(__file__, osp.pardir))
opt, args = parse_options(root_path)


_modes = ['train', 'val']

try:
    os.makedirs(opt['path']['models'])
except OSError:
    pass
try:
    os.makedirs(opt['path']['val'])
except OSError:
    pass
try:
    os.makedirs(opt['path']['log'])
except OSError:
    pass


os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu_id']

copy_opt_file(__file__, opt['path']['experiments_root'])
copy_opt_file(args.opt, opt['path']['experiments_root'])

writer = SummaryWriter(opt['path']['log'])
log = Logger(opt['path']['experiments_root'])
logger = log.get_log()
logger.info('torch version:{}'.format(torch.__version__))
tic0 = time.time()
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
    if opt['model']['use_rot_tv']:
        logger.info('*'*100)
        logger.info('Use rotTV!')
        lambda_tv = opt['train']['lambda_tv']
        logger.info('lambda_tv={}'.format(lambda_tv))
    else:
        logger.info('*'*100)
        logger.info('Not use rotTV!')
        lambda_tv = 0
        logger.info('lambda_tv={}'.format(lambda_tv))
    for epoch in range(opt['train']['epoch'], opt['train']['niter']):
        if opt['train']['epoch'] != 0:
            for _ in range(int(opt['train']['epoch']*3000)):
                step += 1
            for _ in range(int(opt['train']['epoch']*10)):
                step_val += 1
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
            input_spa = torch.cat((input_pair, input_pair - gt_spa), 1)

            # ==========================================================================
            # (1) Update Discriminator
            # ==========================================================================
            # train with real
            netD.train()
            netD.zero_grad()
            d_out_real, dr1, dr2 = netD(input_spa)
            if opt['train']['adv_loss'] == 'wgan-gp':
                d_loss_real = - torch.mean(d_out_real)
            elif opt['train']['adv_loss'] == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            fake, R, fake_temp, R_temp, theta = netG(gt_spa, input_pair, ifshow, step, logger)
            fake = torch.cat((fake, R), 1)
            fake_temp = torch.cat((fake_temp, R_temp), 1)
            
            d_out_fake, df1, df2 = netD(fake.detach())
            if opt['train']['adv_loss'] == 'wgan-gp':
                d_loss_fake = d_out_fake.mean()
            elif opt['train']['adv_loss'] == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            
            if epoch < opt['train']['PreTrainstep']: 
                d_out_fake_temp, df1, df2 = netD(fake_temp.detach())
                if opt['train']['adv_loss'] == 'wgan-gp':
                    d_loss_fake_temp = d_out_fake_temp.mean()
                elif opt['train']['adv_loss'] == 'hinge':
                    d_loss_fake_temp = torch.nn.ReLU()(1.0 + d_out_fake_temp).mean()
                errD = d_loss_real + 0.5 * d_loss_fake + 0.5 * d_loss_fake_temp 
                errD.backward()
                optimizerD.step()

                if opt['train']['adv_loss'] == 'wgan-gp':
                    # Compute gradient penalty
                    alpha = torch.rand(input_spa.size(0), 1, 1, 1).cuda().expand_as(input_spa)
                    interpolated = Variable(alpha * input_spa.data + (1 - alpha) * fake.data, requires_grad=True)
                    out, _, _ = netD(interpolated)

                    grad = torch.autograd.grad(outputs=out,
                                               inputs=interpolated,
                                               grad_outputs=torch.ones(out.size()).cuda(),
                                               retain_graph=True,
                                               create_graph=True,
                                               only_inputs=True)[0]

                    grad = grad.view(grad.size(0), -1)
                    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                    d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
                    # Backward + Optimize
                    errD = opt['train']['lambda_gp'] * d_loss_gp
                    errD.backward()
                    optimizerD.step()

                    alpha = torch.rand(input_spa.size(0), 1, 1, 1).cuda().expand_as(input_spa)
                    interpolated = Variable(alpha * input_spa.data + (1 - alpha) * fake_temp.data,
                                            requires_grad=True)
                    out, _, _ = netD(interpolated)

                    grad = torch.autograd.grad(outputs=out,
                                               inputs=interpolated,
                                               grad_outputs=torch.ones(out.size()).cuda(),
                                               retain_graph=True,
                                               create_graph=True,
                                               only_inputs=True)[0]

                    grad = grad.view(grad.size(0), -1)
                    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                    d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
                    # Backward + Optimize
                    errD = opt['train']['lambda_gp'] * d_loss_gp
                    errD.backward()
                    optimizerD.step()
            else:
                errD = d_loss_real + d_loss_fake
                errD.backward()
                optimizerD.step()
                if opt['train']['adv_loss'] == 'wgan-gp':
                    # Compute gradient penalty
                    alpha = torch.rand(input_spa.size(0), 1, 1, 1).cuda().expand_as(input_spa)
                    interpolated = Variable(alpha * input_spa.data + (1 - alpha) * fake.data, requires_grad=True)
                    out, _, _ = netD(interpolated)

                    grad = torch.autograd.grad(outputs=out,
                                               inputs=interpolated,
                                               grad_outputs=torch.ones(out.size()).cuda(),
                                               retain_graph=True,
                                               create_graph=True,
                                               only_inputs=True)[0]

                    grad = grad.view(grad.size(0), -1)
                    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                    d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
                    # Backward + Optimize
                    errD = opt['train']['lambda_gp'] * d_loss_gp
                    errD.backward()
                    optimizerD.step()

            # ==========================================================================
            # (2) Update Gen network
            # ==========================================================================
            if step % opt['train']['n_dis'] == 0:
                netG.train()
                netG.zero_grad()
                g_out_fake, _, _ = netD(fake)
                if opt['train']['adv_loss'] == 'wgan-gp':
                    g_loss_fake = - g_out_fake.mean()
                elif opt['train']['adv_loss'] == 'hinge':
                    g_loss_fake = - g_out_fake.mean()

                TV_theta_loss = rotTV()
                R_TV_theta_loss = TV_theta_loss(R, theta[:, -1].unsqueeze(1))
                R_temp_TV_theta_loss = TV_theta_loss(R_temp, theta[:, -1].unsqueeze(1))

                errG = 0.5 * g_loss_fake + lambda_tv * R_TV_theta_loss
                if epoch < opt['train']['PreTrainstep']: 
                    g_out_fake_temp, _, _ = netD(fake_temp)
                    if opt['train']['adv_loss'] == 'wgan-gp':
                        g_loss_fake_temp = - g_out_fake_temp.mean()
                    elif opt['train']['adv_loss'] == 'hinge':
                        g_loss_fake_temp = - g_out_fake_temp.mean()

                    Resloss = torch.mean(torch.abs(fake - fake_temp))
                    errG = 0.5 * g_loss_fake + 0.5 * g_loss_fake_temp + 0.1 * Resloss + lambda_tv * R_temp_TV_theta_loss + lambda_tv * R_TV_theta_loss
                errG.backward()
                optimizerG.step()

            if ii % 1000 == 0:
                template = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>5d}/{:0>5d}, DLoss={:5.2e}, GLoss={:5.2e}'
                logger.info(template.format(epoch + 1, opt['train']['niter'], phase, ii, num_iter_epoch[phase], errD.item(),
                                      g_loss_fake.item()))
                writer.add_scalar('TotalDloss',
                                  d_loss_real.item() + d_loss_fake.item() + opt['train']['lambda_gp'] * d_loss_gp.item(), step)
                writer.add_scalar('Dloss', errD.item(), step)
                writer.add_scalar('drloss', d_loss_real.item(), step)
                writer.add_scalar('dfloss', d_loss_fake.item(), step)
                writer.add_scalar('gploss', opt['train']['lambda_gp'] * d_loss_gp.item(), step)
                writer.add_scalar('gfloss', g_loss_fake.item(), step)
                writer.add_scalar('R_TV_theta_loss', R_TV_theta_loss.item(), step)
                ifshow = 1
                toc = time.time()
            else:
                ifshow = 0
            step += 1
        logger.info('-' * 100)
        logger.info('-' * 100)

        # adjust the learning rate
        lr_schedulerG.step()
        lr_schedulerD.step()
        # save model
        if (epoch+1) % 10 == 0:
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
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - tic0)))
    logger.info(f'End of training. Time consumed: {consumed_time}')




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
    # ========================================================================
    schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, opt['train']['milestone'], gamma=0.2)

    netD = Discriminator(conv_dim=opt['train']['ndf'], in_dim=6, path=opt['path']['experiments_root']).cuda()
    optimizerD = optim.Adam(netD.parameters(), lr=opt['train']['lrD'])
    schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, opt['train']['milestone'], gamma=0.2)
    if opt['train']['epoch']:  # fr
        checkpoint = torch.load(os.path.join(opt['path']['models'], 'model_' + str(opt['train']['epoch'])))
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])

        schedulerD.load_state_dict(checkpoint['lr_schedulerD_state_dict'])
        schedulerG.load_state_dict(checkpoint['lr_schedulerG_state_dict'])

        netD.load_state_dict(torch.load(os.path.join(opt['path']['models'], 'D_state_' + str(opt['train']['epoch']) + '.pt')))
        netG.load_state_dict(torch.load(os.path.join(opt['path']['models'], 'G_state_' + str(opt['train']['epoch']) + '.pt')))
        logger.info('loaded checkpoints, epoch{:d}'.format(checkpoint['epoch']))
        

    train_dataset = Dataset(opt['datasets']['data_path'], opt['datasets']['gt_path'], opt['datasets']['patchSize'], opt['datasets']['batchSize'] * 3000, path=opt['path']['experiments_root'])
    val_dataset = Dataset(opt['datasets']['data_path'], opt['datasets']['gt_path'])
    
    datasets = {'train': train_dataset, 'val': val_dataset}
    # train model
    train_model(netG, netD, datasets, optimizerG, schedulerG, optimizerD, schedulerD)


if __name__ == '__main__':
    main()

