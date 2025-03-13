import torch
from augment import get_augment, get_test_augment
import os
import tqdm
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Custom dataset class for loading images.
    """
    def __init__(self, root_dir: str, classes: int, types: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes
        assert types in [
            "train",
            "val",
        ], f"Invalid type {types}, please choose from train, val"
        self.image_path = []
        for cls in tqdm.tqdm(range(self.classes), desc=f"Loading {types} dataset"):
            cls_dir = os.path.join(root_dir, types, str(cls))
            for img in os.listdir(cls_dir):
                self.image_path.append((os.path.join(cls_dir, img), cls))

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img_path, label = self.image_path[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class ImageTestDataset(Dataset):
    """
    Custom dataset class for loading test images.
    """
    def __init__(self, root_dir: str, classes: int, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes
        self.image_path = []
        test_dir = os.path.join(root_dir, "test")
        for img in os.listdir(test_dir):
            self.image_path.append((os.path.join(test_dir, img), img))

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img_path, img_name = self.image_path[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name


def get_train_dataset(
    root_dir="kaggle/input/hw1-dataset/data/", num_workers=0 if os.name == "nt" else 4
):
    """
    For Windows, num_workers must be set to 0 for multiprocessing settings.
    Multiple workers are only available on Unix-based systems.
    """
    transform = get_augment()

    train_dataset = ImageDataset(
        root_dir=root_dir, classes=100, types="train", transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=72, shuffle=True, num_workers=num_workers
    )
    return train_loader


def get_val_dataset(
    root_dir="kaggle/input/hw1-dataset/data/", num_workers=0 if os.name == "nt" else 4
):
    transform = get_test_augment()
    val_dataset = ImageDataset(
        root_dir=root_dir, classes=100, types="val", transform=transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=72, shuffle=False, num_workers=num_workers
    )
    return val_loader


def get_test_dataset(
    root_dir="kaggle/input/hw1-dataset/data/", num_workers=0 if os.name == "nt" else 4
):
    transform = get_test_augment()
    test_dataset = ImageTestDataset(root_dir=root_dir, classes=100, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=72, shuffle=False, num_workers=num_workers
    )
    return test_loader
