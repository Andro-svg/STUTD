import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import re
from einops import rearrange
from utils import train_one_epoch
import random
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torch.nn as nn
from model import SwinTransformerDecoder
import torch.optim.lr_scheduler as lr_scheduler
import math
import torch.nn.functional as F
from scipy.ndimage import convolve

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
# os.environ["WANDB_API_KEY"] = 'b2f259aee83528c8c8dbb6e85360ce182825956b'

def extract_num(filename):
    # 从文件名中提取数字部分
    return int(re.search(r'\d+', filename).group())


class Transform:
    def __init__(self):
        pass

    def flip(self, x, is_hflip, is_vflip):
        if is_hflip:
            x = x.flip(dims=[-1])
        if is_vflip:
            x = x.flip(dims=[-2])
        return x

    def normalize(self, x):
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x))

    def shapesize(self, x, imagesize):
        x = x.to(torch.float32)
        x = F.interpolate(x, size=(imagesize, imagesize))  # , mode='bilinear', align_corners=False
        return x

    def rotate(self, x, is_rotate):
        if is_rotate:
            x = rearrange(x, 'c d h w  -> c d w h')
        return x

    def __call__(self, image, imagesize):
        is_hflip = np.random.rand() < 0.5
        is_vflip = np.random.rand() < 0.5
        is_rotate = np.random.rand() < 0.5
        image = self.shapesize(image, imagesize)
        image = self.flip(image, is_hflip, is_vflip)
        image = self.rotate(image, is_rotate)
        return image


def compute_means(padded_img, kernel):
    return convolve(padded_img, kernel, mode='constant', cval=0.0)


class CustomDataset(Dataset):
    def __init__(self, root_dir, num_images_per_sample, transform=Transform, imagesize=None, per=None):
        self.root_dir = root_dir
        self.num_images_per_sample = num_images_per_sample
        self.transform = transform
        self.subfolders = sorted(os.listdir(root_dir), key=lambda x: int(x))
        self.imagesize = imagesize
        self.per = per

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, idx):
        folder_path = os.path.join(self.root_dir, self.subfolders[idx])
        images = []
        start_index_list = list(
            range(1, len(os.listdir(folder_path)) - self.num_images_per_sample, self.num_images_per_sample))
        start_index_list.append(len(os.listdir(folder_path)) - self.num_images_per_sample)
        image_names = sorted(os.listdir(folder_path), key=lambda x: extract_num(x))

        start_index = random.choice(start_index_list)
        for i in range(start_index, start_index + self.num_images_per_sample):
            image_path = os.path.join(folder_path, image_names[i])
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
            images.append(image)
            concatenated_image = np.stack(images, axis=0)
        concatenated_image = torch.tensor(np.expand_dims(concatenated_image, axis=0))
        if self.transform:
            concatenated_image = self.transform(concatenated_image, self.imagesize)

        return [concatenated_image]


def load_pretrained_parameters(model, pretrained_state_dict):
    model_dict = model.state_dict()
    new_pretrained_state_dict = {k.replace('module.', ''): v for k, v in pretrained_state_dict.items()}

    for key, value in new_pretrained_state_dict.items():
        if key in model_dict and value.shape == model_dict[key].shape:
            model_dict[key] = value
    model.load_state_dict(model_dict)


def main(args):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    print(current_directory)
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    train_dataset = CustomDataset(root_dir=args.data_path,
                                  num_images_per_sample=args.num_images_per_sample,
                                  transform=Transform(), imagesize=args.image_size, per=args.per)
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # 使用DataLoader加载数据
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = SwinTransformerDecoder(pretrained=None, pretrained2d=None, patch_size1=(2, 4, 4), patch_size2=(2, 4, 4),
                                   patch_size3=(2, 4, 4), in_chans=1, embed_dim=48,
                                   depths1=[4], num_heads1=[3], depths2=[4, 2], num_heads2=[3, 6],
                                   depths3=[4, 2, 2], num_heads3=[3, 6, 12],
                                   window_size1=(2, 7, 7), window_size2=(2, 7, 7), window_size3=(2, 7, 7),
                                   mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                                   attn_drop_rate=0., drop_path_rate=0.2, norm_layer=nn.LayerNorm, patch_norm=False,
                                   frozen_stages=-1, use_checkpoint=False, hidden_channels=None, out_channels=1,
                                   scale_factor=2, x1=None, x2=None,
                                   gate_head=4, device=device)

    for weight in model.parameters():
        if weight.requires_grad:
            # 对需要更新的参数进行初始化
            if len(weight.shape) < 2:
                torch.nn.init.kaiming_normal_(weight.unsqueeze(0))
            else:
                torch.nn.init.kaiming_normal_(weight)

    # 加载预训练参数到模型的编码器部分
    pretrained_encoder_state_dict = torch.load(args.weights)
    load_pretrained_parameters(model, pretrained_encoder_state_dict)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_loss = float('inf')
    writer = SummaryWriter("logs_without_vae_data3")

    for epoch in range(args.epochs):
        ori_img, pred, train_loss, loss_re, loss_kl, loss_vae, loss_ms_ssim = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            weights=args.loss_weights)
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        print('learning rate', optimizer.param_groups[0]['lr'])
        writer.add_scalar('train_loss', train_loss, epoch)
        print('train_loss:', train_loss)
        writer.add_scalar('train_loss_re', loss_re, epoch)
        print('train_loss_re:', loss_re)
        writer.add_scalar('train_loss_kl', loss_kl, epoch)
        print('train_loss_kl:', loss_kl)
        writer.add_scalar('train_loss_vae', loss_vae, epoch)
        print('train_loss_vae:', loss_vae)
        writer.add_scalar('train_loss_ms_ssim', loss_ms_ssim, epoch)
        print('train_loss_ms_ssim:', loss_ms_ssim)
        scheduler.step()

        if epoch % 100 == 0:
            torch.save(model.state_dict(), ''.join(["./weights/epoch_", str(epoch), ".pth"]))

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), ''.join(["./weights/best_model_", str(epoch), ".pth"]))
            print(f"Epoch {epoch}: New best model saved with train_loss {train_loss:.4f}")

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15000)
    parser.add_argument('--batch-size', type=int, default=2)#8
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--data-path', type=str,
                        default="./data")
    parser.add_argument('--num_images_per_sample', type=int, default=4)#16
    parser.add_argument('--weights', type=str, default='./weights/best_model.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--image_size', type=int, default=256,
                        help='image size')
    parser.add_argument('--per', type=float, default=0.5,
                        help='for orthogonal space projection')
    parser.add_argument('--loss_weights', default=[1.0, 10.0, 1.0, 0.1],
                        help='weights for loss items')
    parser.add_argument('--lrf', type=float, default=0.01)
    opt = parser.parse_args()

    main(opt)
