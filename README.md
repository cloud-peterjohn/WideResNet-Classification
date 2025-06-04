# ResNet-Classification
## Introduction
This project trains a Wide ResNet model on specified dataset (100 classes) using PyTorch Lightning and tests it on the test set.
| **Metric**          | **Value/Info**                                                                                                    |
|------------------------|--------------------------------------------------------------------------------------------------------------|
| **Model \#Params**     | 63.94M                                                                                                       |
| **Device**             | P100 on Kaggle                                                                                               |
| **Dataset**            | [Click Here](https://drive.google.com/file/d/1fx4Z6xl5b6r4UFkBrn5l0oPEIagZxQ5u/view?usp=drive_link)            |
| **Test Accuracy**      | 0.94                                                                                                         |

## Installation
Install
```python
conda create --name resnet python=3.12
conda activate resnet
pip install requirements.txt
```
Run
```python
python main.py
```
