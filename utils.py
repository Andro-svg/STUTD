import cv2 as cv
import sys
import pickle
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
from scipy.ndimage import convolve


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class MS_SSIM(torch.nn.Module):
    def __init__(self, size_average=True, max_val=1.0):
        super(MS_SSIM, self).__init__()
        self.size_average = size_average
        self.channel = 1
        self.max_val = max_val

    def _ssim(self, img1, img2, size_average=True):
        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = create_window(window_size, sigma, self.channel).cuda()
        img1 = img1.float().cuda()
        img2 = img2.float().cuda()
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=self.channel) - mu1_mu2

        C1 = (0.01 * self.max_val) ** 2
        C2 = (0.03 * self.max_val) ** 2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
        mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=4):
        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda())
        msssim = Variable(torch.Tensor(levels, ).cuda())
        mcs = Variable(torch.Tensor(levels, ).cuda())
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2

        value = (torch.prod(mcs[0:levels - 1] ** weight[0:levels - 1]) * (msssim[levels - 1] ** weight[levels - 1]))
        return value

    def forward(self, img1, img2):
        return self.ms_ssim(img1, img2)


def compute_means(padded_img, kernel):
    return convolve(padded_img, kernel, mode='constant', cval=0.0)


def WLoss(ori, pred, W):
    E = torch.abs(ori - pred)
    wloss = torch.log(torch.norm(W * W * E, p='fro'))
    return wloss


def calculate_LCM(x):
    B, C, D, H, W = x.shape
    x = x.cpu().detach().numpy()
    k = 7
    for m in range(B):
        for n in range(D):
            xtmp = x[m, 0, n, :, :]
            xtmp = 255 * (xtmp - xtmp.min()) / (xtmp.max() - xtmp.min())
            xtmp = cv.GaussianBlur(xtmp, (3, 3), 1)
            padded_img = np.pad(xtmp, ((k, k), (k, k)), mode='constant', constant_values=0)
            kernel_center = np.ones((5, 5)) / 25
            kernel_side = np.ones((5, 5)) / 25
            offsets = [
                (-6, -6), (-6, -2), (-6, 4),
                (-2, 4), (4, 4), (4, -2),
                (4, -6), (-2, -6)
            ]
            mean0 = compute_means(padded_img, kernel_center)[k:-k, k:-k]
            means = np.zeros((H, W, 8))
            for i, (row_offset, col_offset) in enumerate(offsets):
                means[:, :, i] = compute_means(
                    padded_img[k + row_offset:k + row_offset + H, k + col_offset:k + col_offset + W],
                    kernel_side)
            max_means = np.max(means, axis=2)
            LCM = np.where(mean0 > 1.1 * max_means, (mean0 - max_means) * (mean0 / max_means), 0)
            LCM = (LCM - LCM.min()) / (LCM.max() - LCM.min())
            x[m, 0, n, :, :] = LCM

    image = torch.tensor(x)
    return image


def train_one_epoch(model, optimizer, data_loader, device, epoch, weights):
    model.train()
    optimizer.zero_grad()
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        ori, pred, loss_kl, loss_vae = model(data[0].to(device))

        LCM = calculate_LCM(ori)
        ori = ori / (LCM.to(device) + 0.0001)
        ori = (ori - torch.min(ori)) / (torch.max(ori) - torch.min(ori))

        loss_ms_ssim = torch.zeros(1).to(device)
        b, c, d, h, w = ori.shape
        ms_ssim = MS_SSIM()
        for i in range(d):
            tmp = ms_ssim(ori[:, :, i, :, :], pred[:, :, i, :, :])
            loss_ms_ssim += 1 - tmp

        E = torch.abs(ori-pred)
        W = torch.max(E) - E
        loss_re = WLoss(ori, pred, W.to(device))

        loss = torch.sum(loss_kl) + torch.sum(loss_vae) + 5 * loss_re + 10 * loss_ms_ssim
        loss.backward()

        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, loss.item())

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return ori, pred, loss.item(), loss_re.item(), torch.sum(loss_kl).item(), torch.sum(loss_vae).item(), loss_ms_ssim.item()
