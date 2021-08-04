import torch.nn as nn


# convolutional siamese network
class CSN(nn.Module):
    def __init__(
        self,
        p_dropout=0,
        embedding_dim=256,
        feature_extractor=None,
        idx_cutoff=19,
        **kwargs
    ):
        super(CSN, self).__init__(**kwargs)

        if feature_extractor:
            self.convnet = nn.Sequential(
                feature_extractor.features[:idx_cutoff],
                nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(256),
                nn.PReLU(),
                nn.MaxPool2d(2, stride=2),
            )

            # freeze weights
            # for name, p in self.convnet.named_parameters():
            #    idx_param = int(name.split('.')[0])
            #    if idx_param <= idx_cutoff:
            #        p.requires_grad = False
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(32),
                nn.PReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.PReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.PReLU(),
                nn.MaxPool2d(2, stride=2),
            )

        self.fc = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(64 * 8 * 8, 256),
            nn.PReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x1, x2):
        output1 = self.embedding(x1)
        output2 = self.embedding(x2)
        return output1, output2

    def embedding(self, x):
        output = self.convnet(x)
        output = output.view(-1, 64 * 8 * 8)
        output = self.fc(output)
        return output


# fully connected siamese network
class FCSN(nn.Module):
    def __init__(self, p_dropout=0, embedding_dim=256, **kwargs):
        super(FCSN, self).__init__(**kwargs)

        self.fc = nn.Sequential(
            nn.Linear(517, 1024),
            nn.PReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(1024, 256),
            nn.PReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x1, x2):
        output1 = self.fc(x1)
        output2 = self.fc(x2)
        return output1, output2

    def embedding(self, x):
        return self.fc(x)
