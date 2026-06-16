import torch.nn as nn


class TinyCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 12 * 12, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DeeperCNN(nn.Module):

    def __init__(self, num_classes=7, p_drop=0.3):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout(p_drop),
            )
        self.features = nn.Sequential(
            block(1, 32),    # 48 -> 24
            block(32, 64),   # 24 -> 12
            block(64, 128),  # 12 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return nn.functional.relu(out)


class SmallResNet(nn.Module):

    def __init__(self, num_classes=7, p_drop=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(BasicBlock(64, 64),      BasicBlock(64, 64))      # 48
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2),  BasicBlock(128, 128))    # -> 24
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))    # -> 12
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))
