import os
import time
import torch
from model import ResNetModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import get_train_dataset, get_val_dataset
import warnings
from utils import freeze_layers, unfreeze_layers, full_training_configurations

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def train_resnet():
    """
    Train the ResNet model by a two-phase training strategy.
    """

    train_loader = get_train_dataset()
    val_loader = get_val_dataset()

    model = ResNetModel()

    # Phase 1: Train only the fully connected (FC) layer
    time_first_phrase_begin = time.time()
    if os.path.exists("kaggle/working/checkpoints/resnet_fc.pth"):
        model.model.load_state_dict(
            torch.load("kaggle/working/checkpoints/resnet_fc.pth")
        )
        print("Load pre-trained model from checkpoints/resnet_fc.pth.")
    else:
        freeze_layers(model.model)
        trainer = pl.Trainer(
            max_epochs=10, precision="16-mixed", devices=1, accelerator="gpu"
        )
        trainer.fit(model, train_loader, val_loader)
        time_first_phrase_end = time.time()
        if not os.path.exists("kaggle/working/checkpoints"):
            os.makedirs("kaggle/working/checkpoints")
        torch.save(model.model.state_dict(), "kaggle/working/checkpoints/resnet_fc.pth")
        print(
            f"First training phrase finished in {(time_first_phrase_end - time_first_phrase_begin) / 3600.0} hours."
        )

    # Phase 2: Train all the parameters of the model
    unfreeze_layers(model.model)
    checkpoint_callback = ModelCheckpoint(
        dirpath="kaggle/working/checkpoints/",
        filename="resnet-{epoch:03d}-{val_acc:.4f}",
        save_top_k=-1,
        save_last=True,
        every_n_epochs=15,
    )
    model.configure_optimizers = full_training_configurations.__get__(
        model, ResNetModel
    )
    trainer = pl.Trainer(
        max_epochs=125,
        callbacks=[checkpoint_callback],
        precision="16-mixed",
        gradient_clip_val=0.8,
        gradient_clip_algorithm="norm",
        devices=1,
        accelerator="gpu",
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    torch.save(model.model.state_dict(), "kaggle/working/checkpoints/resnet_final.pth")
    time_second_phrase_end = time.time()
    print(
        f"Total training time cost: {(time_second_phrase_end - time_first_phrase_begin) / 3600.0} hours."
    )
