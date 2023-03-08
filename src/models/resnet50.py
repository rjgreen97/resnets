import torch
import torch.nn as nn

from src.models.bottleneck_block import Bottleneck


class ResNet50(nn.Module):

    """
    A residual network with 50 layers that takes in a color 32x32 image and outputs a 10 dimensional
    vector. The network is composed of an inital downsampling layer, followed by 4 bottleneck blocks,
    followed finally by a collection of fully connected layers.
    """

    def __init__(self, block, layer_list, num_classes=10):
        super(ResNet50, self).__init__()
        self.in_feature_maps = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(
            block=block, num_blocks=layer_list[0], feature_maps=64, stride=1
        )
        self.layer2 = self._make_layer(
            block=block, num_blocks=layer_list[1], feature_maps=128, stride=1
        )
        self.layer3 = self._make_layer(
            block=block, num_blocks=layer_list[2], feature_maps=256, stride=2
        )
        self.layer4 = self._make_layer(
            block=block, num_blocks=layer_list[3], feature_maps=512, stride=1
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features=512 * block.expansion, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

    def _make_layer(self, block, num_blocks, feature_maps, stride=1):
        skip_connection = None
        layers = []

        if stride != 1 or self.in_feature_maps != feature_maps * block.expansion:
            skip_connection = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_feature_maps,
                    out_channels=feature_maps * block.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(num_features=feature_maps * block.expansion),
            )

        layers.append(
            block(
                in_channels=self.in_feature_maps,
                out_channels=feature_maps,
                skip_connection=skip_connection,
                stride=stride,
            )
        )

        self.in_feature_maps = feature_maps * block.expansion
        for _i in range(1, num_blocks):
            layers.append(
                block(in_channels=self.in_feature_maps, out_channels=feature_maps)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]

        assert x.shape == (batch_size, 3, 32, 32)
        x = self.conv1(x)
        assert x.shape == (batch_size, 64, 8, 8)
        x = self.layer1(x)
        assert x.shape == (batch_size, 256, 8, 8)
        x = self.layer2(x)
        assert x.shape == (batch_size, 512, 8, 8)
        x = self.layer3(x)
        assert x.shape == (batch_size, 1024, 4, 4)
        x = self.layer4(x)
        assert x.shape == (batch_size, 2048, 4, 4)
        x = self.avgpool(x)
        assert x.shape == (batch_size, 2048, 1, 1)
        x = torch.flatten(x, start_dim=1)
        assert x.shape == (batch_size, 2048)
        x = self.fc1(x)
        assert x.shape == (batch_size, 512)
        x = self.fc2(x)
        assert x.shape == (batch_size, 128)
        x = self.fc3(x)
        assert x.shape == (batch_size, 10)

        return x


if __name__ == "__main__":
    model = ResNet50(Bottleneck, [3, 4, 6, 3])
    x = torch.FloatTensor(256, 3, 32, 32)
    model.forward(x)
