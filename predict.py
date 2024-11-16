import os
import torch.nn.functional as F
import re
import torch
from model import SwinTransformerDecoder
import torch.nn as nn
import numpy as np
from pathlib import Path
import cv2
import argparse
import scipy.io
import time


def extract_num(filename):
    return int(re.search(r'\d+', filename).group())


def shapesize(x, Height, Width):
    x = x.to(torch.float32)
    x = F.interpolate(x, size=(Height, Width))
    return x

parser = argparse.ArgumentParser()
parser.add_argument('--image-path', type=str, default=r'./data',
                    help='image path')
parser.add_argument('--save-path', type=str, default=r'./result',
                    help='image path')
parser.add_argument('--num_images_per_sample', type=int, default=16)
args = parser.parse_args()

model_weight_path = "./weights/best_model.pth"

os.makedirs(args.save_path, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_folder = Path(args.image_path)
save_folder = Path(args.save_path)
in_files = [fn for fn in os.listdir(img_folder) if fn.endswith('.bmp')]
image_names = sorted(os.listdir(args.image_path), key=lambda x: extract_num(x))
start_list = list(range(1, len(in_files) - args.num_images_per_sample, args.num_images_per_sample))
start_list.append(len(in_files) - args.num_images_per_sample + 1)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)


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

model_weight = torch.load(model_weight_path, map_location=device)
model = model.to(device)
new_state_dict = {k.replace('module.', ''): v for k, v in model_weight.items()}
model.load_state_dict(new_state_dict)
model.eval()

start_time = time.time()
with torch.no_grad():
    for i in start_list:
        images = []
        for j in range(args.num_images_per_sample):
            image_path = os.path.join(args.image_path, image_names[i + j - 1])
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
            H, W = image.shape
            images.append(image)
            concatenated_image = np.stack(images, axis=0)
        data = torch.tensor(np.expand_dims(concatenated_image, axis=0))
        data = shapesize(data, 256, 256)
        data = data.unsqueeze(dim=0)
        data = data.float()
        ori_img, pred, _, _ = model(data.to(device))
        ori_img = ori_img.squeeze(dim=0)
        pred = pred.squeeze(dim=0)
        ori_img = shapesize(ori_img, H, W)
        pred = shapesize(pred, H, W)
        tar_map = abs(ori_img - pred)

        pred = pred.cpu()
        for j in range(args.num_images_per_sample):
            ori_img = data[0, 0, j, :, :]
            BKG = pred[0, j, :, :]
            BKG = (BKG - torch.min(BKG)) / (torch.max(BKG) - torch.min(BKG))
            tar = tar_map[0, j, :, :]
            tar = tar.cpu()
            tar = np.array((tar - torch.min(tar)) / (torch.max(tar) - torch.min(tar)))
            idx = np.argsort(np.abs(tar).ravel())[::-1]
            tar.ravel()[idx[int(0.00004 * H * W):(H * W)]] = 0
            image_name = image_names[i + j - 1]
            result_name = str(extract_num(image_name)) + '_tar.jpg'
            final_out_path = save_folder / result_name
            cv2.imwrite(str(final_out_path), np.array(255 * tar))
            mat_name = str(extract_num(image_name)) + '_tar.mat'
            mat_path = save_folder / mat_name
            E = tar
            mat_data = {'E': E}
            scipy.io.savemat(mat_path, mat_data)

all_time = time.time() - start_time
print(all_time / (len(in_files) * args.num_images_per_sample))
