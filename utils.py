import torch


def freeze_layers(model):
    """
    Freeze all the layers except the fully connected layer for phase 1 training.
    """
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False


def unfreeze_layers(model):
    """
    Unfreeze all the layers for phase 2 training.
    """
    for param in model.parameters():
        param.requires_grad = True


def full_training_configurations(self):
    """
    Optimizer and scheduler configurations for phrase 2 training.
    """
    # optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
    optimizer = torch.optim.NAdam(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=5
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=45, eta_min=5e-6
            ),
        ],
        milestones=[5],
    )
    return [optimizer], [scheduler]
