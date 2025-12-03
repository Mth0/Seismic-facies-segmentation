import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import torch
import random
from torchvision.transforms import InterpolationMode, Normalize, v2
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from dataset import *

class PercentileClip:
    """
    Normalize an image by clipping pixel values based on percentile ranges.

    This transform computes the `lower` and `upper` percentiles of the input
    image and rescales all pixel values to the [0, 1] range using those
    percentile thresholds. Values below the lower percentile become 0, values
    above the upper percentile become 1.

    Args
        lower (float, default=5) : Lower percentile used for clipping.
        upper (float, default=95) : Upper percentile used for clipping.
        cache (bool, default=True) : Placeholder flag for optional caching of percentile values.

    Returns
        sample (dict): Updated sample dictionary with the normalized image.
    """

    def __init__(self, lower=5, upper=95, cache=True):
        self.lower = lower
        self.upper = upper
        self.cache = cache
        self._clip_vals = {}

    def __call__(self, sample):
        img = sample["image"] 
        
        p_low = np.percentile(img, self.lower)
        p_high = np.percentile(img, self.upper)

        img = np.clip((img - p_low) / (p_high - p_low + 1e-8), 0, 1)
        sample["image"] = img
        
        return sample

def make_fast_transform(normalizer=None, img_size=224, augment=False):
    """
    Given a size of image (square), constructs a list of
    transformations to be applied.

    Args:
        normalizer (torchvision.transforms.v2.Normalize): A normalizer
        object.
        img_size (int): The image size input for the transformations.
        augment (bool): A flag indicating if augmentation transformations
        will be considered.
    Returns:
        torchvision.transforms.v2.Compose: A object containing the transform
        pipeline.
    """
    transforms = []

    transforms.append(PercentileClip(lower=5, upper=95))

    if augment:
        transforms += [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=(-5, 5), interpolation=InterpolationMode.BILINEAR),
            v2.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.8, 1.2),
                ratio=(1.0, 1.0),
                interpolation=InterpolationMode.BILINEAR,
            ),
            v2.RandomApply(
                [v2.ElasticTransform(alpha=50,
                                     sigma=5,
                                     interpolation=InterpolationMode.BILINEAR),],
                p=0.3,
            ),
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.2, contrast=0.2)],
                p=0.3,
            ),
            v2.RandomApply(
                [v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))],
                p=0.2,
            ),
            v2.RandomErasing(p=0.3,
                             scale=(0.02, 0.33),
                             ratio=(0.3, 3.3))
        ]
    else:
        transforms.append(
            v2.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR)
        )

    transforms.append(v2.ToDtype(torch.float32, scale=False))
    if normalizer is not None:
        transforms.append(normalizer)

    return v2.Compose(transforms)





def generate_centered_augmentations(
    dataset,
    target_classes,
    crop_size=112,
    resize_to=224,
    min_fraction_img=0.2,
    num_crops_per_img=2,
    target_per_class=3000,
):
    """
    Generates balanced augmentations for underrepresented classes
    by selecting images with a minimum percentage of the target class
    and creating crops centered on the class region.
    Args:
        dataset (Dataset): returns (img, mask).
        target_classes (list[int]): classes to balance.
        crop_size (int): crop size.
        resize_to (int): final crop size.
        min_fraction_img (float): min fraction a class should have to be considered.
        num_crops_per_img (int): number of crops per selected image.
        target_per_class (int): number of generated images per class.

    Returns:
        all_aug (list[(Tensor, Tensor)]): a list (img, mask) with the augmented images
        and its masks.
    """
    all_aug = []
    stats = {cls: 0 for cls in target_classes}

    for cls in target_classes:
        total_generated = 0
        # Filter images that contains the target class
        candidate_images = []
        for img, mask in tqdm(dataset):
            img = img if img.ndim == 3 else img.unsqueeze(0)
            mask = mask if mask.ndim == 3 else mask.unsqueeze(0)
            frac = (mask == cls).float().mean().item()
            if frac >= min_fraction_img:
                candidate_images.append((img, mask))

        if len(candidate_images) == 0:
            print(f"No images with class {cls} >= {min_fraction_img*100:.2f}%")
            continue

        # Generates crops centered on the class region
        while total_generated < target_per_class:
            img, mask = random.choice(candidate_images)
            _, H, W = mask.shape

            for _ in range(num_crops_per_img):
                # Finds pixels from the target class
                ys, xs = torch.where(mask[0] == cls)
                if len(xs) == 0:
                    continue
                # Chooses a random central pixel
                center_y = random.choice(ys).item()
                center_x = random.choice(xs).item()

                # Calculates the top-left of the central pixel
                top = max(0, min(H - crop_size, center_y - crop_size // 2))
                left = max(0, min(W - crop_size, center_x - crop_size // 2))

                crop_m = mask[:, top:top+crop_size, left:left+crop_size]
                crop_i = img[:, top:top+crop_size, left:left+crop_size]

                # Redimensions
                crop_i = F.interpolate(crop_i.unsqueeze(0), size=(resize_to, resize_to),
                                       mode='bilinear', align_corners=False).squeeze(0)
                crop_m = F.interpolate(crop_m.unsqueeze(0).float(), size=(resize_to, resize_to),
                                       mode='nearest').long().squeeze(0)

                all_aug.append((crop_i, crop_m))
                total_generated += 1
                stats[cls] += 1

                if total_generated >= target_per_class:
                    break

        print(f"Class {cls}: {stats[cls]} crops generated")

    return all_aug


def create_dataset_train(train_data, normalizer):
    """
    Creates a new dataset object containg more images
    with rare classes occurence.
    Args:
        train_data (np.ndarray): The images numpy array.
        normalizer (torchvision.transforms.v2.Normalize): A normalizer
        object.
    Returns:
        train_data_all (torch.utils.data.ConcatDataset): A list of concatenated
        datasets containing the augmented datasets and the original one.
    """
    # Generates balanced augmentations for each rare class
    aug_class_2 = generate_centered_augmentations(train_data, target_classes=[2], target_per_class=3000, crop_size = 180, num_crops_per_img = 4, min_fraction_img=0.2)
    aug_class_4 = generate_centered_augmentations(train_data, target_classes=[4], target_per_class=5000, crop_size = 112, num_crops_per_img = 4, min_fraction_img=0.05)


    # Applies the transform on the crops
    aug_ds_2 = SDataset(sample = aug_class_2, transform= make_fast_transform(normalizer, img_size=224, augment=True))
    aug_ds_4 = SDataset(sample = aug_class_4, transform= make_fast_transform(normalizer, img_size=224, augment=True))

    # Concat everything
    train_data_all = ConcatDataset([train_data, aug_ds_2, aug_ds_4])

    print(f"Total count of generated samples: {len(train_data_all)-len(train_data)}")

    return train_data_all



def classify_strats(label, RARE_CLASS_1=2, RARE_CLASS_2=4, device="cuda"):
    """
    Given a array of labeled images and two rare classes, it separates
    the images into four groups:
        1. Images that contains none of the two rare classes;
        2. Images that contains the first rare class, but not the second;
        3. Images that contains the second rare class, but not the first;
        4. Images that contains both rare classes.

    Args:
        label (numpy.ndarray): Images with its pixel-wise labels.
        RARE_CLASS_1 (int): The id of the first rare class.
        RARE_CLASS_2 (int): The id of the second rare class.
        device (str): The device where the computations are going to be made.
    Returns:
        stratify_keys_np (np.ndarray): An array indicating to which group
        each image is.
    """
    temp_label = torch.tensor(label).to(device)

    print("Generating stratification keys...")

    stratify_keys = []
    for i in range(len(label)):
        mask = label[i].squeeze() 

        has_rare_1 = (mask == RARE_CLASS_1).any()
        has_rare_2 = (mask == RARE_CLASS_2).any()

        key = 0
        if has_rare_1:
            key += 1
        if has_rare_2:
            key += 2

        stratify_keys.append(key)

    stratify_keys_np = np.array(stratify_keys)

    print(f"Key 0 (no rare): {np.sum(stratify_keys_np == 0)}")
    print(f"Key 1 (only {RARE_CLASS_1}): {np.sum(stratify_keys_np == 1)}")
    print(f"Key 2 (only {RARE_CLASS_2}): {np.sum(stratify_keys_np == 2)}")
    print(f"Key 3 (both): {np.sum(stratify_keys_np == 3)}")
    print("-" * 30)

    return stratify_keys_np


def mean_std_cut(data):
    """
    Given a dataset, computes its mean and
    standard deviation, returning a normalizer object.
    Args:
        data (np.ndarray): the array containing the dataset.
    Returns:
        normalizer (torchvision.transforms.v2.Normalize): A normalizer
        object.
    """
    cut_data = np.empty_like(data)
    percentil_clip = PercentileClip(5, 95)
    for i, image in enumerate(data):
        res = percentil_clip({"image": image, "mask": None})
        cut_data[i] = res["image"]

    normalizer = construct_normalizer(cut_data, inplace=False)
    return normalizer


def create_stratified_split(input_data_path, output_data_path,
                            train_size=0.8, random_state=None,
                            normalizer=None, device="cuda"):
    """
    Given an input data and its labels, splits the dataset based
    on a stratification.
    Args:
        input_data_path (str): File path to the input data.
        output_data_path (str): File path to the label data.
        train_size (float): Size of the training dataset after the split.
        random_state (int): The random state used for the splitting. 'None' gives
        a pseudo-random splitting.
        device (str): The device where the computations are going to be made.
    Returns:
        train_data (SDataset): The training dataset after the split.
        val_data (SDataset): The validation dataset after the split.
        normalizer (torchvision.transforms.v2.Normalize): A normalizer
        object.
    """
    train_data = np.load(input_data_path)
    label_data = np.load(output_data_path)

    stratify_keys = classify_strats(label_data,
                                    RARE_CLASS_1=2,
                                    RARE_CLASS_2=4,
                                    device=device)

    img_train, img_val, label_train, label_val = train_test_split(train_data,
                                                              label_data,
                                                              train_size=0.8,
                                                              random_state=random_state,
                                                              stratify=stratify_keys)

    paths = []
    if random_state is not None:
        paths = [
            f"./seismic_data/img_train_{random_state}.npy",
            f"./seismic_data/label_train_{random_state}.npy",
            f"./seismic_data/img_val_{random_state}.npy",
            f"./seismic_data/label_val_{random_state}.npy"]
    else:
        paths = [
            "./seismic_data/img_train.npy",
            "./seismic_data/label_train.npy",
            "./seismic_data/img_val.npy",
            "./seismic_data/label_val.npy"]

    np.save(paths[0], img_train)
    np.save(paths[1], label_train)
    np.save(paths[2], img_val)
    np.save(paths[3], label_val)

    print("Constructing normalizer....")
    if normalizer == None:
        normalizer = mean_std_cut(img_train)
    print(f"Normalizer with mean {normalizer.mean} and standard devation {normalizer.std} constructed")

    print('-' * 30)
    print("Constructing datasets....")

    train_data = SDataset(paths[0],
                          paths[1],
                          transform = make_fast_transform(normalizer=normalizer,
                                                          img_size=224, augment=True))

    val_data = SDataset(paths[2],
                        paths[3],
                        transform = make_fast_transform(normalizer=normalizer,
                                                        img_size=224, augment=False))

    print("Datasets created")
    print('-' * 30)

    return train_data, val_data, normalizer
