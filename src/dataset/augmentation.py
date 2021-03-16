import torch
from torchvision.transforms.functional import hflip
from torchvision.transforms import (
    Compose as C,
    ToTensor as TT,
    ColorJitter as CJ
)


class Compose(C):
    """This is an extension of torchvision.transforms.Compose so that it can be applied to 'image' and 'gt'.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        super(Compose, self).__init__(transforms)

    def __call__(self, img, gt):
        for t in self.transforms:
            img, gt = t(img, gt)
        return img, gt


class ToTensor(TT):
    """This is an extension of torchvision.transforms.ToTensor so that it can be applied to 'image' and 'gt'.
    """

    def __call__(self, img, gt):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return super(ToTensor, self).__call__(img), gt


class RandomColorJitter(CJ):
    def __init__(self, p: float = 0.5, brightness=0.3, contrast=0.3, saturation=1.5, hue=0.1):
        super(RandomColorJitter, self).__init__(brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, img, gt):
        if torch.rand(1) < self.p:
            img = super(RandomColorJitter, self).__call__(img)
        return img, gt


class RandomFlip(torch.nn.Module):
    def __init__(self, input_size: int, p: float = 0.5):
        """initialize
        Args:
            input_size (int): image size
            p (float, optional): probability of executing. Defaults to 0.5.
        """
        self.input_size = input_size
        self.p = p

    def __call__(self, img, gt):
        if torch.rand(1) < self.p:
            img = hflip(img)
            gt[:, 0], gt[:, 2] = self.input_size - gt[:, 2], self.input_size - gt[:, 0]
        return img, gt
