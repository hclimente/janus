import torch.nn as nn


class SiameseNet(nn.Module):

    def __init__(self):
        super(SiameseNet, self).__init__()

        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1),
                                     nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 3, padding=1),
                                     nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 64, 3, padding=1),
                                     nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 8 * 8, 256),
                                nn.Dropout(0.5),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.Dropout(0.5))

    def forward(self, x1, x2):
        output1 = self.embedding(x1)
        output2 = self.embedding(x2)
        return output1, output2

    def embedding(self, x):
        output = self.convnet(x)
        output = output.view(-1, 64 * 8 * 8)
        output = self.fc(output)
        return output
