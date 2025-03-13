import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class ResizeImages:
    """
    Resize images to a fixed size. This is used in the data augmentation pipeline.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, img_tensor):
        _, height, width = img_tensor.shape  # [C, H, W]
        new_size = min(width, height)
        img_tensor = TF.center_crop(img_tensor, new_size)

        return TF.resize(
            img_tensor,
            [self.size, self.size],
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )


def get_augment(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Get the data augmentation pipeline for training.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to(device)),
            ResizeImages(224),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10,
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1
                    )
                ],
                p=0.5,
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_test_augment(
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
):
    """
    Get the data augmentation pipeline for testing.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to(device)),
            ResizeImages(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transform
