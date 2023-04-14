# ResNet on CIFAR-10 within budget - Mini Project 1

## Prerequisites
The following Python packages are required:

- torch
- torchsummary
- numpy
- tqdm
- multiprocessing

You can install them manually or use the following command in your Python notebook:

```
! pip install torch torchsummary numpy tqdm multiprocessing
```
## Overview
This project implements ResNet on the CIFAR-10 dataset while staying within a specified budget. The `resnet_model.py` file defines the core model, while the `DataCollector.py` file is used to load the CIFAR-10 dataset and apply various data augmentation techniques. Additionally, the `train.ipynb` file is used for training the model on the train dataset.

## Results
The model achieved a training accuracy of 99.98% and a testing accuracy of 92.75%. We tested the model using the `test.ipynb` file, which is specifically designed for testing on the test dataset. The following table summarizes the results:

| Type |	Accuracy| 
| ---| ---------
| Training  |99.98%| 
| Testing	|92.75%| 