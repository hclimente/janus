import torch

from networks import SiameseNet


def test_forward():

    img1 = torch.rand((1, 3, 64, 64))
    img2 = torch.rand((1, 3, 64, 64))
    net = SiameseNet()

    output1, output2 = net(img1, img2)

    assert type(output1) is torch.Tensor
    assert output1.shape == (1,256)

    assert type(output2) is torch.Tensor
    assert output2.shape == (1,256)


def test_embedding():

    img1 = torch.rand((1, 3, 64, 64))
    img2 = torch.rand((1, 3, 64, 64))
    net = SiameseNet()

    output1, output2 = net(img1, img2)

    assert torch.all(output1 == net.embedding(img1))
    assert torch.all(output2 == net.embedding(img2))