import torch
import numpy as np
import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_channel, out_channel, downsample = None, stride = 1):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size = 1, stride = 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size = 3, stride = stride, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * 4, kernel_size = 1, stride = 1, padding = 0)
        self.bn3 = nn.BatchNorm2d(out_channel * 4)

        self.relu = nn.ReLU()
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample:
            identity = self.downsample(identity)
        print("merging")
        x += identity
        x = self.relu(x)
        return x
#layer 1 stride 1
#layer 2 3 4 stride 2  
class resnet(nn.Module):
    def __init__(self, block, layers, img_channel, num_classes):
        super(resnet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(img_channel, 64, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(block, layers[0], 64, 1)
        self.layer2 = self._make_layer(block, layers[1], 128, 2)
        self.layer3 = self._make_layer(block, layers[2], 256, 2)
        self.layer4 = self._make_layer(block, layers[3], 512, 2)
        self.avgPool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgPool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_blocks, out_channel, stride):
        downsample = None
        layers = []
        print(num_blocks)
        if stride != 1 or self.in_channel != out_channel * 4: 
            downsample = nn.Sequential(nn.Conv2d(self.in_channel, out_channel * 4, kernel_size = 1, stride = stride),
                                        nn.BatchNorm2d(out_channel * 4)
                                        )
        layers.append(block(self.in_channel, out_channel, downsample, stride))
        self.in_channel = out_channel * 4

        for i in range(num_blocks - 1):
            layers.append(block(self.in_channel, out_channel))
        return nn.Sequential(*layers)

def resnet50(img_channels = 3, num_classes = 1000):
    return resnet(block, [3, 4, 6, 3], img_channels, num_classes)

model = resnet50()

inp = torch.randn(2, 3, 224, 224)
print(model(inp).shape)




