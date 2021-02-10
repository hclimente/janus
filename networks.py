import torch.nn as nn


class SiameseNet(nn.Module):

    def __init__(self):
        super(SiameseNet, self).__init__()

        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5),
                                     nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5),
                                     nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(16384, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256))

        self.predict = nn.Sequential(nn.PReLU(),
                                     nn.Linear(256, 2))

    def forward(self, x1, x2):
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)
        return output1, output2

    def forward_one(self, x):
        output = self.embedding(x)
        output = self.predict(output)
        return output

    def embedding(self, x):
        output = self.convnet(x)
        output = output.view(1, -1)
        output = self.fc(output)
        return output
