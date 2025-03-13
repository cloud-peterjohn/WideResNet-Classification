import os
import torch
import pandas as pd
from model import ResNetModel
from dataset import get_test_dataset
import tqdm


def test_resnet(checkpoint_path):
    """
    Test the ResNet model with the given checkpoint path.
    """
    test_loader = get_test_dataset()

    assert checkpoint_path.endswith(
        ".ckpt"
    ), 'Invalid checkpoint path, must end with ".ckpt"!'
    model = ResNetModel.load_from_checkpoint(
        os.path.join("/kaggle/working/checkpoints", checkpoint_path)
    )
    model.eval()
    model.cuda()

    predictions = []
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            inputs, img_name = batch
            inputs = inputs.to("cuda")
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for name, pred in zip(img_name, preds.cpu().numpy()):
                name = name.split(".")[0]
                predictions.append((name, pred))

    if not os.path.exists("/kaggle/working/results"):
        os.makedirs("/kaggle/working/results")
    df = pd.DataFrame(predictions, columns=["image_name", "pred_label"])
    df.to_csv(
        f"/kaggle/working/results/predictions_with_{checkpoint_path}.csv", index=False
    )
    print(
        f"Test predictions saved to predictions_with_{checkpoint_path}.csv successfully."
    )


def test(checkpoint_dir="/kaggle/working/checkpoints"):
    """
    Test all the checkpoints in the given directory.
    """
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".ckpt"):
            try:
                test_resnet(file)
            except Exception as e:
                print(f"Error occurred when testing {file}: {e}")
