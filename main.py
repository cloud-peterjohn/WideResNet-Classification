"""
Student ID: 113550901
Note: This code was run on Kaggle with NVIDIA P100 GPU.
"""

from train import train_resnet
from test import test
import torch


def main():
    """
    Train the ResNet model and test it.
    """
    assert torch.cuda.is_available(), "CUDA is not available!"
    train_resnet()
    test()


if __name__ == "__main__":
    main()
# 一句话总结这个项目（英文）：
# 