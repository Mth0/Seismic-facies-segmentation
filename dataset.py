import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, ConcatDataset
import torch
import random
from torchvision.transforms import Normalize
from torchvision.transforms import v2
import matplotlib.colors as mcolors

class SDataset(Dataset):
    def __init__(self, image_path=None, mask_path=None, sample=None, transform=None, device="cpu"):
        """
        Args:
            image_path (str): path to .npy file containing all images
            mask_path (str, optional): path to .npy file containing all segmentation masks
            sample (list[tuple[Tensor, Tensor]], optional): lista de pares (img, mask)
            transform (callable, optional): transform applied to both image and mask
            device (str): The device where the computations are going to be made.
        """
        if sample is not None:
            imgs, masks = zip(*sample)
            imgs_t = torch.stack(imgs).float()
            masks_t = torch.stack(masks).long()
        else:
            # ---- Load from disk ----
            imgs = np.load(image_path, mmap_mode=None)
            masks = np.load(mask_path, mmap_mode=None) if mask_path is not None else None
            imgs_t = torch.from_numpy(imgs).float()
            masks_t = torch.from_numpy(masks).long() if masks is not None else None

        # ---- Normalize shape ----
        if imgs_t.ndim == 3:
            imgs_t = imgs_t.unsqueeze(1)  # (N, 1, H, W)
        elif imgs_t.ndim == 4 and imgs_t.shape[-1] <= 4:
            imgs_t = imgs_t.permute(0, 3, 1, 2)

        self.images = imgs_t.to(device, non_blocking=True)
        self.masks = masks_t.to(device, non_blocking=True) if masks_t is not None else None
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx] if self.masks is not None else None
    
        if self.transform:
            # --- TRAINING/VALIDATION PATH ---
            if mask is not None:
                transformed_dict = self.transform({"image": img, "mask": mask})
                img = transformed_dict['image']
                mask = transformed_dict['mask']
                
                return img, mask
            
            # --- INFERENCE PATH (mask is None) ---
            else:
                transformed_dict = self.transform({"image": img})
                img = transformed_dict['image']
                return img 
    
        # Fallback (if self.transform is None)
        if mask is not None:
            return img, mask
        else:
            return img

    @staticmethod
    def count_classes(masks, num_classes = 6):
        """
        Count how many pixels there's in each class.
        Args:
            masks (numpy.ndarray): Array containg the masks of each image
            num_classes (int): Total number of classes in the dataset
        Returns:
            counts (numpy.array): An array containing the total number of
                pixels in each class.
            (float): The percentage of each class in the dataset
        """
         mask_pixels = masks.flatten()
        
         counts = np.zeros((num_classes,))
         for i in range(num_classes):
             counts[i] += (mask_pixels == i).sum()
         return counts, counts / counts.sum()

def vis_first_samples(dataset, num_samples=10):
    """
    Visualize the first samples from dataset.
    Args:
        dataset (SDataset): A dataset object containg the data
        num_samples (int): How many samples are going to be plotted
    """
    class_colors = ["#d73027", "#1a9850", "#9e81d0", "#a6611a", "#fdae61", "#fee08b"]
    cmap = mcolors.ListedColormap(class_colors)

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        if isinstance(sample, tuple):
            image, mask = sample
        else:
            image, mask = sample, None

        # Converts shape: (C, H, W) → (H, W, C)
        img_np = image.numpy()
        if img_np.ndim == 3:
            if img_np.shape[0] in [1, 3]:  # C,H,W → H,W,C
                img_np = img_np.transpose(1, 2, 0)
        else:
            img_np = img_np

        # Plots image
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(img_np.squeeze(), cmap='gray' if img_np.shape[-1] == 1 else None)
        plt.title(f"Image {i}")
        plt.axis('off')

        # Plots masks (if exists)
        if mask is not None:
            mask_np = mask.numpy().squeeze()
            plt.subplot(1, 2, 2)
            plt.imshow(mask_np, cmap=cmap, vmin=0, vmax=len(class_colors)-1)
            plt.title(f"Mask {i}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------
# 1) Plot: Majority class couting
# ----------------------------------------------------------------------
def plot_majority_class_counts(masks: torch.Tensor):
    """
    Plot how many masks has a class as its majority class.
    Args:
        masks (torch.tensor): A tensor containg the masks of
            each image
    """

    n = masks.shape[0]
    flat = masks.view(n, -1)
    majors = torch.mode(flat, dim=1).values.cpu().numpy()
    num_classes = int(masks.max().item()) + 1

    counts = np.bincount(majors, minlength=num_classes)

    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(num_classes), counts)
    plt.xticks(np.arange(num_classes))
    plt.xlabel("class label")
    plt.ylabel("number of masks with this majority")
    plt.title("Counts of majority-class across masks")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 2) Plot: Avarage percentage of each class
# ----------------------------------------------------------------------
def plot_mean_class_percentage(masks: torch.Tensor):
    """
    Plot average percentage of each class in the dataset.
    Args:
        masks (torch.tensor): A tensor containg the masks of
            each image
    """

    n, H, W = masks.shape
    num_classes = int(masks.max().item()) + 1
    flat = masks.view(n, -1)

    mean_perc = []
    for cls in range(num_classes):
        cls_pixels = (flat == cls).float().mean(dim=1)  # % em cada mask
        mean_perc.append(cls_pixels.mean().item())      # média entre masks

    mean_perc = np.array(mean_perc)

    plt.figure(figsize=(7, 4))
    plt.bar(np.arange(num_classes), mean_perc * 100)
    plt.xticks(np.arange(num_classes))
    plt.ylabel("Average percentage (%)")
    plt.xlabel("Class")
    plt.title("Mean pixel percentage per class across all masks")
    plt.tight_layout()
    plt.show()



def construct_normalizer(imgs, inplace=False):
    """
    Construct a torchvision.transforms.Normalize object
    based on a set of images.
    Args:
        imgs (numpy.ndarray): The array containg the images
            to be considered during the Normalize construction.
        inplace (bool): bool to make this operation in-place.
    Returns:
        
    """
    flat = imgs.flatten()
    mean = flat.mean()
    std = flat.std()

    return v2.Normalize([mean], [std], inplace=inplace)