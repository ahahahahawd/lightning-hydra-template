from os import path
import os
import pathlib
import torch
import numpy as np


def is_image_file(filename):
    """Return True if the filename is a valid image file."""
    return any(filename.lower().endswith(extension) for extension in [
        '.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'
        ])


# def get_file_list(file_dir: pathlib.Path):
#     """
#     Get all image files in a directory.
    
#     :param file_dir: The directory to get files from.
#     :param file_dir: image folder
#     :return: image path str list
#     """
    
#     all_image_paths = sorted(list(file_dir.glob('*')))
#     all_image_paths_ = [str(path) for path in all_image_paths if is_image_file(path)]
#     return all_image_paths_


def get_image_path_list(path_dir):
    """get all image paths in a directory."""
    files = sorted(os.listdir(path_dir))
    image_path_list = [os.path.join(path_dir, x) for x in files if is_image_file(x)]
    return image_path_list


def get_long_path_by_short_name(name=None):
    path_dict = {
        'mini': '/data1/wangding/datasets/snow/mini/',
        'all': '/data1/wangding/datasets/snow/all/',
        'L': '/data1/wangding/datasets/snow/media/jdway/GameSSD/overlapping/test/Snow100K-L/',
        'M': '/data1/wangding/datasets/snow/media/jdway/GameSSD/overlapping/test/Snow100K-M/',
        'S': '/data1/wangding/datasets/snow/media/jdway/GameSSD/overlapping/test/Snow100K-S/',
        'real': '/data1/wangding/datasets/snow/realistic/',

    }
    if name in path_dict:
        path = path_dict[name]
        return path
    else:
        return name
    
    
# operation_seed_counter = 0
# def get_generator():
#     global operation_seed_counter
#     operation_seed_counter += 1
#     g_cuda_generator = torch.Generator(device="cuda")
#     g_cuda_generator.manual_seed(operation_seed_counter)
#     return g_cuda_generator

# class AugmentNoise(object):
#     def __init__(self, style):
#         if style.startswith('gauss'):
#             self.params = [float(p) / 255.0 for p in style.replace('gauss', '', 1).split('_')]
#             if len(self.params) == 1:
#                 self.style = "gauss_fix"
#             elif len(self.params) == 2:
#                 self.style = "gauss_range"
#         elif style.startswith('poisson'):  
#             self.params = [float(p) for p in style.replace('poisson', '', 1).split('_')]
#             if len(self.params) == 1:
#                 self.style = "poisson_fix"
#             elif len(self.params) == 2:
#                 self.style = "poisson_range"
#     def add_train_noise(self, x):
#         shape = x.shape
#         if self.style == "gauss_fix":
#             std = self.params[0]
#             std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
#             noise = torch.cuda.FloatTensor(shape, device=x.device)
#             torch.normal(mean=0.0, std=std, generator=get_generator(), out=noise)
#             return x + noise
#         elif self.style == "gauss_range":
#             min_std, max_std = self.params
#             std = torch.rand(size=(shape[0], 1, 1, 1), device=x.device) * (max_std - min_std) + min_std
#             noise = torch.cuda.FloatTensor(shape, device=x.device)
#             torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
#             return x + noise
#         elif self.style == "poisson_fix":
#             lam = self.params[0]
#             lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
#             noised = torch.poisson(lam * x, generator=get_generator()) / lam
#             return noised
#         elif self.style == "poisson_range":
#             min_lam, max_lam = self.params
#             lam = torch.rand(size=(shape[0], 1, 1, 1), device=x.device) * (max_lam - min_lam) + min_lam
#             noised = torch.poisson(lam * x, generator=get_generator()) / lam
#             return noised
#     def add_valid_noise(self, x):
#         shape = x.shape
#         if self.style == "gauss_fix":
#             std = self.params[0]
#             return np.array(x + np.random.normal(size=shape) * std, dtype=np.float32)
#         elif self.style == "gauss_range":
#             min_std, max_std = self.params
#             std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
#             return np.array(x + np.random.normal(size=shape) * std, dtype=np.float32)
#         elif self.style == "poisson_fix":
#             lam = self.params[0]
#             return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
#         elif self.style == "poisson_range":
#             min_lam, max_lam = self.params
#             lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
#             return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)


# def space_to_depth(x, block_size):
#     n, c, h, w = x.size()
#     unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
#     return unfolded_x.view(n, c * block_size**2, h // block_size,
#                            w // block_size)

# def generate_mask_pair(img):
#     # This function generates random masks with shape (N x C x H/2 x W/2)
#     #import pdb
#     #pdb.set_trace()
#     n, c, h, w = img.shape
#     mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
#                         dtype=torch.bool,
#                         device=img.device)
#     mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
#                         dtype=torch.bool,
#                         device=img.device)
#     idx_pair = torch.tensor(
#         [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
#         dtype=torch.int64,
#         device=img.device)
#     rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
#                          dtype=torch.int64,
#                          device=img.device)
#     torch.randint(low=0,
#                   high=8,
#                   size=(n * h // 2 * w // 2, ),
#                   generator=get_generator(),
#                   out=rd_idx)
#     rd_pair_idx = idx_pair[rd_idx]
#     rd_pair_idx += torch.arange(start=0,
#                                 end=n * h // 2 * w // 2 * 4,
#                                 step=4,
#                                 dtype=torch.int64,
#                                 device=img.device).reshape(-1, 1)
#     mask1[rd_pair_idx[:, 0]] = 1
#     mask2[rd_pair_idx[:, 1]] = 1
#     return mask1, mask2

# def generate_subimages(img, mask):
#     # This function generates paired subimages given random masks
#     n, c, h, w = img.shape
#     subimage = torch.zeros(n,
#                            c,
#                            h // 2,
#                            w // 2,
#                            dtype=img.dtype,
#                            layout=img.layout,
#                            device=img.device)
#     for i in range(c):
#         img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
#         img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
#         subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
#             n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
#     return subimage
