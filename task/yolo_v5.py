import torch
from torch import nn


class YoloV5(nn.Module):
    def __init__(self,):
        super().__init__()
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", _verbose=False)

    def forward(self, x):
        return self.model(x)