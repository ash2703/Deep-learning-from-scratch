import torch
import torch.nn as nn
from torchviz import make_dot

class block(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1, padding = 0, downsample = None):
        super(block, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channel, out_channel // 4, kernel_size = 1, stride = 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(out_channel // 4)
        self.conv2 = nn.Conv2d(out_channel // 4, out_channel // 4, kernel_size = 3, stride = stride, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channel // 4)
        self.conv3 = nn.Conv2d(out_channel // 4, out_channel, kernel_size = 1, stride = 1, padding = 0)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        original = x
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("con1 bn", x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print("con2 ", x.shape)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # print("con3 ", x.shape)
        # print(self.downsample)
        if self.downsample:
            # print(x.shape)
            original = self.downsample(original)
        #     print("downsampling ", x.shape)
        # print("merging")
        x += original
        # print(x.shape)
        x = self.relu(x)
        print("1 block done", x.shape)
        return x
    
class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(block, 64, 1, layers[0])
        self.layer2 = self._make_layer(block, 128, 2, layers[1])
        self.layer3 = self._make_layer(block, 256, 2, layers[2])
        self.layer4 = self._make_layer(block, 512, 2, layers[3])

        self.avgPool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(2048, 1000)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.layer1(x)
        print("layer 1 shape: ", x.shape)
        x = self.layer2(x)
        print("layer 2 shape: ", x.shape)
        x = self.layer3(x)
        print("layer 3")
        x = self.layer4(x)
        print("layer 4 ", x.shape)

        x = self.avgPool(x)
        print(x.shape)
        x = x.reshape(x.shape[0], -1)
        print(x.shape)
        x = self.linear(x)

        return x


        

    def _make_layer(self, block, in_channel, stride, num_blocks):
        layers = []
        print(self.in_channel)
        downsample = nn.Sequential(nn.Conv2d(self.in_channel, in_channel * 4, kernel_size = 1, stride = stride),
                                   nn.BatchNorm2d(in_channel * 4))
        

        layers.append(block(self.in_channel, in_channel * 4, stride = stride, downsample = downsample))
        
        for i in range(num_blocks - 1):
            layers.append(block(in_channel * 4, in_channel * 4))
        self.in_channel = in_channel * 4
        return nn.Sequential(*layers)

resnet50 = ResNet(block, [3, 4, 6, 3])
print(resnet50)

inp = torch.randn(2, 3, 224, 224)
out = resnet50(inp)
make_dot(out).render("attached", format="png")



