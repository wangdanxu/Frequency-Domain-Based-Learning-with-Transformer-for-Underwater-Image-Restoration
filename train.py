from __future__ import absolute_import, division, print_function

import torch

import dataset as dataset
from vgg import *
import math
from measure_ssim_psnr import *
import shutil
from tqdm import tqdm
from get_uiqm import *
from test import test
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from options import opt, device
from models import *
from misc import *
import torch.fft as fft
from ssim import *
from timm.scheduler.cosine_lr import CosineLRScheduler


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':

    scale = 2
    print("Underwater Image Super-Resolution [SCALE]: ", scale)

    netG = CC_Module(scale)
    # print('underwater dehaze network ', netG)
    netG.to(device)

    mse_loss = nn.MSELoss()
    ssim_loss = SSIMLoss(11)
    L1 = nn.L1Loss()

    vgg = Vgg16(requires_grad=False).to(device)

    optim_g = optim.Adam(netG.parameters(),
                         lr=opt.learning_rate_g,
                         betas=(opt.beta1, opt.beta2),
                         weight_decay=opt.wd_g)

    lr_scheduler = CosineLRScheduler(
        optimizer=optim_g,
        t_initial=int(opt.end_epoch),
        t_mul=1.,
        lr_min=opt.MIN_LR,  # 5e-6
        warmup_lr_init=opt.WARMUP_LR,  # 5e-7
        warmup_t=0,  # 0
        cycle_limit=1,
        t_in_epochs=False,
    )

    dataset = dataset.Dataset_Load(data_path=opt.data_path,
                                   scale=scale,
                                   transform=dataset.ToTensor()
                                   )
    batches = int(dataset.len / opt.batch_size)

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    if not os.path.exists(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)

    # models_loaded = getLatestCheckpointName()
    # latest_checkpoint_G = models_loaded
    #
    # print('loading model for generator ', latest_checkpoint_G)
    latest_checkpoint_G = None
    if latest_checkpoint_G == None:
        start_epoch = 1
        print('No checkpoints found for netG and netD! retraining')

    else:
        checkpoint_g = torch.load(os.path.join(opt.checkpoints_dir, latest_checkpoint_G))
        start_epoch = checkpoint_g['epoch'] + 1
        netG.load_state_dict(checkpoint_g['model_state_dict'])
        optim_g.load_state_dict(checkpoint_g['optimizer_state_dict'])
        for param_group in optim_g.param_groups:
            param_group['lr'] = opt.learning_rate_g

        print('Restoring model from checkpoint ' + str(start_epoch))

    netG.train()
    from tensorboardX import SummaryWriter  # tensorboard

    writer = SummaryWriter('./result/')

    for epoch in range(start_epoch, opt.end_epoch + 1):

        # bar = Bar('Training', max=batches)

        opt.total_mse_loss = 0.0
        opt.total_vgg_loss = 0.0
        opt.total_ssim_loss = 0.0
        opt.total_fft_loss = 0.0
        opt.total_G_loss = 0.0

        for i_batch, sample_batched in enumerate(dataloader):
            hazy_batch = sample_batched['hazy']
            clean_batch = sample_batched['clean']

            hazy_batch = hazy_batch.to(device)
            clean_batch = clean_batch.to(device)

            optim_g.zero_grad()

            pred_batch = netG(hazy_batch)
            batch_mse_loss = torch.mul(opt.lambda_mse, mse_loss(pred_batch, clean_batch))
            batch_mse_loss.backward(retain_graph=True)

            batch_ssim_loss = torch.mul(opt.lambda_ssim, ssim_loss(pred_batch, clean_batch))
            batch_ssim_loss.backward(retain_graph=True)

            # fft_clean = fft.fft2(clean_batch).float()
            # fft_pred = fft.fft2(pred_batch).float()
            # batch_fft_loss = torch.mul(opt.lambda_fft_mse, L1(fft_pred, fft_clean))
            # batch_fft_loss.backward(retain_graph=True)

            clean_vgg_feats = vgg(normalize_batch(clean_batch))
            pred_vgg_feats = vgg(normalize_batch(pred_batch))
            batch_vgg_loss = torch.mul(opt.lambda_vgg, mse_loss(pred_vgg_feats.relu2_2, clean_vgg_feats.relu2_2))
            batch_vgg_loss.backward()


            opt.batch_mse_loss = batch_mse_loss.item()
            opt.total_mse_loss += opt.batch_mse_loss

            opt.batch_ssim_loss = batch_ssim_loss.item()
            opt.total_ssim_loss += opt.batch_ssim_loss

            opt.batch_vgg_loss = batch_vgg_loss.item()
            opt.total_vgg_loss += opt.batch_vgg_loss

            # opt.batch_fft_loss = batch_fft_loss.item()
            # opt.total_fft_loss += opt.batch_fft_loss

            # opt.batch_G_loss = opt.batch_mse_loss + opt.batch_vgg_loss + opt.batch_ssim_loss+ opt.batch_fft_loss
            opt.batch_G_loss = opt.batch_mse_loss + opt.batch_vgg_loss + opt.batch_ssim_loss
            opt.total_G_loss += opt.batch_G_loss

            optim_g.step()

            lr_scheduler.step(epoch)

            # bar.suffix = f' Epoch : {epoch} | ({i_batch+1}/{batches}) | ETA: {bar.eta_td} | g_mse: {opt.batch_mse_loss} | g_vgg: {opt.batch_vgg_loss}'
            # print('\r Epoch : ' + str(epoch) + ' | (' + str(i_batch+1) + '/' + str(batches) + ') | mse: ' + str(opt.batch_mse_loss) + ' | vgg: ' + str(opt.batch_vgg_loss) + ' | ssim: ' + str(opt.batch_ssim_loss)+ ' | g_fft: ' + str(opt.batch_fft_loss), end='', flush=True)
            print('\r Epoch : ' + str(epoch) + ' | (' + str(i_batch + 1) + '/' + str(batches) + ') | mse: ' + str(
                opt.batch_mse_loss) + ' | vgg: ' + str(opt.batch_vgg_loss) + ' | ssim: ' + str(opt.batch_ssim_loss),
                  end='', flush=True)
        # bar.next()
        writer.add_scalar('mse', opt.total_mse_loss, epoch)  # tensorboard
        # print('\nFinished ep. %d, lr = %.6f, total_mse = %.6f, total_vgg = %.6f, total_fft = %.6f,total_ssim = %.6f' % (epoch, get_lr(optim_g), opt.total_mse_loss, opt.total_vgg_loss,opt.total_fft_loss, opt.total_ssim_loss))
        print('\nFinished ep. %d, lr = %.6f, total_mse = %.6f, total_vgg = %.6f, total_ssim = %.6f' % (
        epoch, get_lr(optim_g), opt.total_mse_loss, opt.total_vgg_loss, opt.total_ssim_loss))
        # print('training epoch %d, %d / %d patches are finished, g_mse = %.6f' % (
        # epoch, i_batch, batches, opt.batch_mse_loss))

        torch.save({'epoch': epoch,
                    'model_state_dict': netG.state_dict(),
                    'optimizer_state_dict': optim_g.state_dict(),
                    'mse_loss': opt.total_mse_loss,
                    'vgg_loss': opt.total_vgg_loss,
                    'ssim_loss': opt.total_ssim_loss,
                    # 'fft_loss': opt.total_fft_loss,
                    'opt': opt,
                    'total_loss': opt.total_G_loss}, os.path.join(opt.checkpoints_dir, 'netG_' + str(epoch) + '.pt'))
        test(netG, epoch)
        ### compute SSIM and PSNR
        SSIM_measures, PSNR_measures = SSIMs_PSNRs("./UFO-120/TEST/hr/",
                                                   './facades/' + str(scale) + 'netG_' + str(epoch) + '/')
        print("SSIM on {0} samples".format(len(SSIM_measures)) + "\n")
        print("Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)) + "\n")
        writer.add_scalar('SSIM_mean', np.mean(SSIM_measures), epoch)
        writer.add_scalar('SSIM_std', np.std(SSIM_measures), epoch)
        print("PSNR on {0} samples".format(len(PSNR_measures)) + "\n")
        print("Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)) + "\n")
        writer.add_scalar('PSNR_mean', np.mean(PSNR_measures), epoch)
        writer.add_scalar('PSNR_std', np.std(PSNR_measures), epoch)

        gen_uqims = measure_UIQMs('./facades/' + str(scale) + 'netG_' + str(epoch) + '/')
        writer.add_scalar('gen_mean', np.mean(gen_uqims), epoch)
        writer.add_scalar('gen_std', np.std(gen_uqims), epoch)