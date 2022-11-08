import argparse
import torch
import os

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default='UFO-120/train_val/')
parser.add_argument('--scale', type=int, default=2)

parser.add_argument('--checkpoints_dir', default='./ckpts/')
parser.add_argument('--batch_size', type=int, default=1)

parser.add_argument('--learning_rate_g', type=float, default=1e-03)

parser.add_argument('--TRAIN.LR_SCHEDULER.NAME',default='cosine')
parser.add_argument('--TRAIN.WARMUP_EPOCHS',type=int,default=1) #swin transformer 是20
parser.add_argument('--MIN_LR',type=float,default=5e-6)
parser.add_argument('--WARMUP_LR',type=float,default=5e-7)

parser.add_argument('--end_epoch', type=int, default=20)
parser.add_argument('--img_extension', default='.png')

parser.add_argument('--beta1', type=float ,default=0.7)
parser.add_argument('--beta2', type=float ,default=0.999)
parser.add_argument('--wd_g', type=float ,default=0.05)
parser.add_argument('--wd_d', type=float ,default=0.00000)

parser.add_argument('--batch_mse_loss', type=float, default=0.0)
parser.add_argument('--total_mse_loss', type=float, default=0.0)

parser.add_argument('--batch_vgg_loss', type=float, default=0.0)
parser.add_argument('--total_vgg_loss', type=float, default=0.0)

parser.add_argument('--batch_ssim_loss', type=float, default=0.0)
parser.add_argument('--total_ssim_loss', type=float, default=0.0)

parser.add_argument('--batch_fft_loss', type=float, default=0.0)
parser.add_argument('--total_fft_loss', type=float, default=0.0)

parser.add_argument('--batch_G_loss', type=float, default=0.0)
parser.add_argument('--total_G_loss', type=float, default=0.0)

parser.add_argument('--lambda_mse', type=float, default=1)
# parser.add_argument('--lambda_fft_mse', type=float, default=0.4)
parser.add_argument('--lambda_vgg', type=float, default=0.02)
parser.add_argument('--lambda_ssim', type=float, default=0.5)


parser.add_argument('--testing_start', type=int, default=1)
parser.add_argument('--testing_end', type=int, default=1)
parser.add_argument('--testing_dir_inp', default="./UFO-120/11/")
parser.add_argument('--testing_dir_gt', default="./UFO-120/TEST/hr/")

opt = parser.parse_args()
# print(opt)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

if not os.path.exists(opt.checkpoints_dir):
    os.makedirs(opt.checkpoints_dir)