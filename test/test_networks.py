import torch
import torchvision.models as models

from janus.networks import SiameseNet

img1 = torch.rand((1, 3, 64, 64))
img2 = torch.rand((1, 3, 64, 64))


def test_forward():

    net = SiameseNet()

    output1, output2 = net(img1, img2)

    assert type(output1) is torch.Tensor
    assert output1.shape == (1, 256)

    assert type(output2) is torch.Tensor
    assert output2.shape == (1, 256)


def test_embedding():

    net = SiameseNet(embedding_dim=512)

    output1, output2 = net(img1, img2)

    assert output1.shape == (1, 512)
    assert torch.all(output1 == net.embedding(img1))
    assert torch.all(output2 == net.embedding(img2))


def test_vgg():

    vgg19 = models.vgg19(pretrained=True)
    net = SiameseNet(feature_extractor=vgg19)

    output1, output2 = net(img1, img2)

    assert torch.all(output1 == net.embedding(img1))
    assert torch.all(output2 == net.embedding(img2))

