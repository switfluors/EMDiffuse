import os
import random

import torch
import torch.multiprocessing
import torch.utils.data as data
from PIL import Image, ImageFilter
from PIL import ImageOps
from tifffile import imread
from torchvision import transforms


def find_max_number(folder_path):
    max_number = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            filename = filename[:-4]
        if not filename.isdigit():
            continue
        filename = int(filename)

        number = int(filename)
        max_number = max(max_number, number)
    return max_number


def find_max_folder_number(folder_path):
    max_number = 0
    for folder_name in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, folder_name)):
            if not folder_name.isdigit():
                continue
            folder_name = int(folder_name)
            number = int(folder_name)
            max_number = max(max_number, number)
    return max_number


def pil_loader(path):
    return Image.open(path).convert('L')


def pil_loader_noL(path):
    return Image.open(path)


def invert(tensor):
    return 1 - tensor


import os
import random
import numpy as np
from torch.utils import data
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import tifffile
import torch


def pil_loader(path):
    """Custom loader that reads uint16 TIFF and converts to PIL Image scaled to [0, 255] (as float32)"""
    img = tifffile.imread(path).astype(np.float32) / 65535.0  # Normalize to [0, 1]
    img = (img * 255).clip(0, 255).astype(np.uint8)  # Rescale to [0, 255] for PIL compatibility
    return Image.fromarray(img)


class EMDiffusenDataset(data.Dataset):  # Denoise and super-resolution Dataset
    def __init__(self, data_root, data_len=-1, norm=True, percent=False, phase='train', image_size=[128, 16],
                 loader=pil_loader):
        self.data_root = data_root
        self.phase = phase
        self.img_paths, self.gt_paths = self.read_dataset(self.data_root)
        print(f"[EMDiffusenDataset] Loaded {len(self.img_paths)} image pairs from {data_root}")

        # PIL-based transforms after .tif loading & float scaling
        self.tfs = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0,1] float32 tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1,1]
        ])

        self.loader = loader
        self.norm = norm
        self.image_size = image_size
        print(f"[EMDiffusenDataset] Phase: {self.phase}, Normalization: {self.norm}, Resize to: {image_size}")

    def __getitem__(self, index):
        ret = {}
        file_name = self.img_paths[index]
        gt_file_name = self.gt_paths[index]

        # img_before = np.array(self.loader(file_name))  # Load original input image (PIL)
        # img_after = self.tfs(self.loader(file_name)).numpy()  # After transforms
        #
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 4))
        # plt.subplot(1, 2, 1)
        # plt.title("Original input")
        # plt.imshow(img_before, cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.title("Transformed input")
        # plt.imshow(img_after.squeeze(), cmap='gray')
        # plt.show()

        # Load as PIL images (uint8 scaled from float)
        img = self.loader(file_name)
        gt = self.loader(gt_file_name)

        print(f"[EMDiffusenDataset][{index}] Loaded input image: {file_name}, GT image: {gt_file_name}")
        print(f"[EMDiffusenDataset][{index}] Raw input size: {img.size}, GT size: {gt.size}")

        # Optional augmentations (flip, rotate, blur)
        # if self.phase == 'train':
        #     img, gt = self.aug(img, gt)

        # Apply resizing and normalization transforms
        img = self.tfs(img)
        gt = self.tfs(gt)

        print(f"[EMDiffusenDataset][{index}] Tensor input shape: {img.shape}, GT shape: {gt.shape}")

        ret['gt_image'] = gt
        ret['cond_image'] = img
        ret['path'] = '_'.join(file_name.split(os.sep)[-3:])  # e.g., 0_Spec__4w_04_123.tif

        return ret

    def __len__(self):
        return len(self.img_paths)

    # def aug(self, img, gt):
    #     if random.random() < 0.5:
    #         img = ImageOps.flip(img)
    #         gt = ImageOps.flip(gt)
    #     if random.random() < 0.5:
    #         img = img.rotate(90)
    #         gt = gt.rotate(90)
    #     if random.random() < 0.5:
    #         img = img.filter(ImageFilter.GaussianBlur(radius=3))
    #     return img, gt

    def read_dataset(self, data_root):
        img_paths = []
        gt_paths = []
        for cell_num in os.listdir(data_root):
            if cell_num == '.DS_Store':
                continue
            cell_path = os.path.join(data_root, cell_num)
            for noise_level in os.listdir(cell_path):
                for img in sorted(os.listdir(os.path.join(cell_path, noise_level))):
                    if 'tif' in img:
                        img_path = os.path.join(cell_path, noise_level, img)
                        gt_path = img_path.replace('wf', 'gt')

                        print(f"Input: {img_path}")
                        print(f"GT:    {gt_path}")

                        img_paths.append(img_path)
                        gt_paths.append(gt_path)

        print(f"Total image pairs collected: {len(img_paths)}")
        return img_paths, gt_paths


class vEMDiffuseTrainingDatasetPatches(
    data.Dataset):  # Dataset for vEMDiffuse training using the downloaded subvolume
    def __init__(self, data_root, data_len=-1, norm=True, percent=False, phase='train', image_size=[256, 256],
                 method='vEMDiffuse-i',
                 z_times=6, loader=pil_loader_noL):
        self.data_root = data_root
        self.phase = phase
        self.z_times = z_times
        self.gt_paths, self.max_nums = self.read_dataset(self.data_root)
        self.percent = percent
        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.loader = loader
        self.norm = norm
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        # sub_volume_index = random.randint(0, len(self.gt_paths) - 1)

        gt_file_name = self.gt_paths[index]
        upper_bound = self.z_times // 2
        lower_bound = self.z_times // 2 if self.z_times % 2 == 0 else self.z_times // 2 + 1
        file_index = random.randint(upper_bound, self.max_nums[index] - lower_bound)
        gt_file_name = os.path.join(gt_file_name, str(file_index) + '.tif')
        img_up = self.loader(os.path.join(os.path.dirname(gt_file_name), str(file_index - upper_bound) + '.tif'))
        img_below = self.loader(os.path.join(os.path.dirname(gt_file_name), str(file_index + lower_bound) + '.tif'))
        gts = []
        for i in range(file_index - upper_bound + 1, file_index + lower_bound):
            gts.append(self.loader(os.path.join(os.path.dirname(gt_file_name), str(i) + '.tif')))
        img_up, img_below, gts = self.aug(img_up, img_below, gts)
        img_up = self.tfs(img_up)
        img_below = self.tfs(img_below)
        out_gts = []
        for i in range(len(gts)):
            gt_tmp = self.tfs(gts[i])
            out_gts.append(gt_tmp)
        img = torch.cat([img_below, img_up], dim=0)
        gt = torch.cat(out_gts, dim=0)
        ret['gt_image'] = gt
        ret['cond_image'] = img
        ret['path'] = '_'.join(gt_file_name.split(os.sep)[-3:])
        return ret

    def __len__(self):
        return len(self.gt_paths)

    def aug(self, img_up, img_below, gts):
        if random.random() < 0.5:
            img_up = img_up.filter(ImageFilter.GaussianBlur(radius=3))
        if random.random() < 0.5:
            img_below = img_below.filter(ImageFilter.GaussianBlur(radius=3))
        if random.random() < 0.5:
            img_up = ImageOps.flip(img_up)
            img_below = ImageOps.flip(img_below)
            for i in range(len(gts)):
                gt_tmp = ImageOps.flip(gts[i])
                gts[i] = gt_tmp
        if random.random() < 0.5:
            img_up = img_up.rotate(90)
            img_below = img_below.rotate(90)
            for i in range(len(gts)):
                gt_tmp = gts[i].rotate(90)
                gts[i] = gt_tmp
        return img_up, img_below, gts

    def read_dataset(self, data_root):
        gt_paths = []  # subfolder
        max_nums = []  # z_depth in subfolder
        for cell_num in os.listdir(data_root):
            cell_path = os.path.join(data_root, cell_num)
            gt_paths.append(cell_path)
            max_nums.append(find_max_number(os.path.join(cell_path)))
        return gt_paths, max_nums


class vEMDiffuseTrainingDatasetVolume(data.Dataset):  # Dataset for vEMDiffuse training using the whole isotropic volume
    def __init__(self, data_root, data_len=-1, norm=True, percent=False, phase='train', image_size=[256, 256],
                 method='vEMDiffuse-i',
                 z_times=6, loader=pil_loader_noL):
        self.data_root = data_root
        self.phase = phase
        self.z_times = z_times
        self.method = method
        self.percent = percent
        self.tfs = transforms.Compose([
            # transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.depth = find_max_number(self.data_root)
        print('the number of layers: ', self.depth + 1)
        self.height, self.width = imread(os.path.join(data_root, str(self.depth) + '.tif')).shape
        self.loader = loader
        self.norm = norm
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        x_index = random.randint(0, self.width - self.image_size[0])
        y_index = random.randint(0, self.height - self.image_size[1])
        upper_bound = self.z_times // 2
        lower_bound = self.z_times // 2 if self.z_times % 2 == 0 else self.z_times // 2 + 1
        z_index = random.randint(upper_bound, self.depth - lower_bound - 1)

        img_up = Image.fromarray(imread(os.path.join(self.data_root, str(z_index - upper_bound) + '.tif'))[
                                 y_index:y_index + self.image_size[1],
                                 x_index:x_index + self.image_size[0]])
        img_below = Image.fromarray(imread(os.path.join(self.data_root, str(z_index + lower_bound) + '.tif'))[
                                    y_index:y_index + self.image_size[1],
                                    x_index:x_index + self.image_size[0]])
        gts = []
        for i in range(z_index - upper_bound + 1, z_index + lower_bound):
            gts.append(Image.fromarray(
                imread(os.path.join(self.data_root, str(i) + '.tif'))[y_index:y_index + self.image_size[1],
                x_index:x_index + self.image_size[0]]))
        img_up, img_below, gts = self.aug(img_up, img_below, gts)
        img_up = self.tfs(img_up)
        img_below = self.tfs(img_below)
        out_gts = []
        for i in range(len(gts)):
            gt_tmp = self.tfs(gts[i])
            out_gts.append(gt_tmp)
        img = torch.cat([img_below, img_up], dim=0)
        gt = torch.cat(out_gts, dim=0)
        ret['gt_image'] = gt
        ret['cond_image'] = img
        ret['path'] = str(y_index) + '_' + str(x_index) + '_' + str(z_index) + '.tif'
        return ret

    def __len__(self):
        if self.phase == 'train':
            return 20000

    def aug(self, img_up, img_below, gts):
        if random.random() < 0.5:
            img_up = img_up.filter(ImageFilter.GaussianBlur(radius=3))
        if random.random() < 0.5:
            img_below = img_below.filter(ImageFilter.GaussianBlur(radius=3))
        if random.random() < 0.5:
            img_up = ImageOps.flip(img_up)
            img_below = ImageOps.flip(img_below)
            for i in range(len(gts)):
                gt_tmp = ImageOps.flip(gts[i])
                gts[i] = gt_tmp
        if random.random() < 0.5:
            img_up = img_up.rotate(90)
            img_below = img_below.rotate(90)
            for i in range(len(gts)):
                gt_tmp = gts[i].rotate(90)
                gts[i] = gt_tmp
        return img_up, img_below, gts


class vEMDiffuseTestIsotropic(
    data.Dataset):  # dataset for testing vEMDiffuse-i. Given an isotropic volume, first downsample it and then reconstruct it.
    def __init__(self, data_root, data_len=-1, norm=True, percent=False, phase='train', image_size=[256, 256],
                 z_times=6,
                 loader=pil_loader):
        self.data_root = data_root
        self.phase = phase
        self.z_times = z_times
        self.gt_paths = self.read_dataset(self.data_root)
        self.percent = percent

        self.tfs = transforms.Compose([
            # transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.loader = loader
        self.norm = norm
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        gt_file_name = self.gt_paths[index]
        file_index = int(gt_file_name.split(os.sep)[-2])
        upper_bound = self.z_times // 2
        lower_bound = self.z_times // 2 if self.z_times % 2 == 0 else self.z_times // 2 + 1
        img_up = self.loader(os.path.join(self.data_root, str(file_index - upper_bound), gt_file_name.split('/')[-1]))
        img_below = self.loader(
            os.path.join(self.data_root, str(file_index + lower_bound), gt_file_name.split(os.sep)[-1]))
        gts = []
        for i in range(file_index - upper_bound + 1, file_index + lower_bound):
            gts.append(self.loader(os.path.join(self.data_root, str(i), gt_file_name.split(os.sep)[-1])))
        img_up = self.tfs(img_up)
        img_below = self.tfs(img_below)
        out_gts = []
        for i in range(len(gts)):
            gt_tmp = self.tfs(gts[i])
            out_gts.append(gt_tmp)
        img = torch.cat([img_below, img_up], dim=0)
        gt = torch.cat(out_gts, dim=0)
        ret['gt_image'] = gt
        ret['cond_image'] = img
        ret['path'] = '_'.join(gt_file_name.split(os.sep)[-3:])
        return ret

    def __len__(self):
        return len(self.gt_paths)

    def read_dataset(self, data_root):
        import os
        z_depth = find_max_folder_number(data_root)
        upper_bound = self.z_times // 2
        lower_bound = self.z_times // 2 if self.z_times % 2 == 0 else self.z_times // 2 + 1
        gt_paths = []
        for i in range(upper_bound, z_depth - lower_bound, self.z_times):  # Downsample
            for file in os.listdir(os.path.join(data_root, str(i))):
                gt_paths.append(os.path.join(data_root, str(i), file))
                # below_paths.append(os.path.join(data_root, str(i + 1), file))
        return gt_paths


class vEMDiffuseTestAnIsotropic(
    data.Dataset):  # dataset for testing vEMDiffuse-i. Given an anisotropic volume, reconstruct it.
    def __init__(self, data_root, data_len=-1, norm=True, percent=False, phase='train', image_size=[256, 256],
                 z_times=6,
                 loader=pil_loader):
        self.data_root = data_root
        self.phase = phase
        self.z_times = z_times
        self.gt_paths, self.below_paths = self.read_dataset(self.data_root)
        self.percent = percent

        self.tfs = transforms.Compose([
            # transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.loader = loader
        self.norm = norm
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        gt_file_name = self.gt_paths[index]
        img_up = self.loader(gt_file_name)
        img_below = self.loader(self.below_paths[index])
        gts = []
        for i in range(self.z_times - 1):
            gts.append(self.loader(gt_file_name))
        img_up = self.tfs(img_up)
        img_below = self.tfs(img_below)
        out_gts = []
        for i in range(len(gts)):
            gt_tmp = self.tfs(gts[i])
            out_gts.append(gt_tmp)
        img = torch.cat([img_below, img_up], dim=0)
        gt = torch.cat(out_gts, dim=0)
        ret['gt_image'] = gt
        ret['cond_image'] = img
        ret['path'] = '_'.join(gt_file_name.split(os.sep)[-3:])
        return ret

    def __len__(self):
        return len(self.gt_paths)

    def read_dataset(self, data_root):
        import os
        z_depth = find_max_folder_number(data_root)
        gt_paths = []
        below_paths = []
        for i in range(0, z_depth):
            for file in os.listdir(os.path.join(data_root, str(i))):
                gt_paths.append(os.path.join(data_root, str(i), file))
                below_paths.append(os.path.join(data_root, str(i + 1), file))
                # below_paths.append(os.path.join(data_root, str(i + 1), file))
        return gt_paths, below_paths
