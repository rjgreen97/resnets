import torch
import torch.functional as F
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, skip_connection=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels=(out_channels * self.expansion),
                kernel_size=1,
                stride=stride,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=(out_channels * self.expansion)),
        )
        self.skip_connection = skip_connection
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.skip_connection:
            residual = self.skip_connection(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet101(nn.Module):
    def __init__(self, block, layer_list, num_classes=10):
        super(ResNet101, self).__init__()
        self.in_feature_maps = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(
            block=block, feature_maps=64, num_blocks=layer_list[0], stride=1
        )
        self.layer2 = self._make_layer(
            block=block, feature_maps=128, num_blocks=layer_list[1], stride=2
        )
        self.layer3 = self._make_layer(
            block=block, feature_maps=256, num_blocks=layer_list[2], stride=2
        )
        self.layer4 = self._make_layer(
            block=block, feature_maps=512, num_blocks=layer_list[3], stride=2
        )
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(
            in_features=(512 * block.expansion), out_features=num_classes
        )

    def _make_layer(self, block, feature_maps, num_blocks, stride=1):
        skip_connection = None
        layer_list = []

        if stride != 1 or self.in_feature_maps != (feature_maps * block.expansion):

            skip_connection = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_feature_maps,
                    out_channels=feature_maps,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(num_features=(feature_maps * block.expansion)),
            )

        layer_list.append(
            block(
                in_channels=self.in_feature_maps,
                out_channels=feature_maps,
                stride=stride,
                skip_connection=skip_connection,
            )
        )

        self.in_feature_maps = feature_maps * block.expansion
        for i in range(1, num_blocks):
            layer_list.append(
                block(in_channels=self.in_feature_maps, out_channels=feature_maps)
            )

        return nn.Sequential(*layer_list)

    def forward(self, x):
        batch_size = x.shape[0]

        assert x.shape == (batch_size, 3, 32, 32)
        x = self.conv1(x)
        assert x.shape == (batch_size, 64, 8, 8)
        x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = torch.flatten(x)
        # x = self.relu(x)
        # x = self.fc1(x)

        # return x


if __name__ == "__main__":
    model = ResNet101(Bottleneck, [3, 4, 23, 3])
    x = torch.FloatTensor(256, 3, 32, 32)
    model.forward(x)
