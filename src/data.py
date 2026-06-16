import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

EMOTIONS = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
            4: "Sad", 5: "Surprise", 6: "Neutral"}
NUM_CLASSES = 7
IMG_SIZE = 48

USAGE_TO_SPLIT = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}

FER_MEAN = (0.5077,)
FER_STD = (0.2550,)


def load_fer_arrays(csv_path):
    df = pd.read_csv(csv_path)
    images = np.stack(
        [np.asarray(p.split(), dtype=np.uint8) for p in df["pixels"]]
    ).reshape(-1, IMG_SIZE, IMG_SIZE)
    labels = df["emotion"].to_numpy(dtype=np.int64)
    usage = df["Usage"].to_numpy()
    return images, labels, usage


def split_arrays(images, labels, usage):
    splits = {}
    for usage_name, split_name in USAGE_TO_SPLIT.items():
        mask = usage == usage_name
        splits[split_name] = (images[mask], labels[mask])
    return splits


def compute_mean_std(images):
    x = images.astype(np.float32) / 255.0
    return float(x.mean()), float(x.std())


class FER2013Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx], mode="L")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


def build_transforms(augment=False):
    tfms = []
    if augment:
        tfms += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
    tfms += [transforms.ToTensor(), transforms.Normalize(FER_MEAN, FER_STD)]
    return transforms.Compose(tfms)


def class_weights(labels, num_classes=NUM_CLASSES):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(labels, num_classes=NUM_CLASSES):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    per_class_w = 1.0 / counts
    sample_w = per_class_w[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True,
    )


def get_dataloaders(csv_path, batch_size=64, augment=True, num_workers=2,
                    use_weighted_sampler=False):
    images, labels, usage = load_fer_arrays(csv_path)
    splits = split_arrays(images, labels, usage)

    train_ds = FER2013Dataset(*splits["train"], transform=build_transforms(augment))
    val_ds   = FER2013Dataset(*splits["val"], transform=build_transforms(False))
    test_ds  = FER2013Dataset(*splits["test"], transform=build_transforms(False))

    if use_weighted_sampler:
        sampler = make_weighted_sampler(splits["train"][1])
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
