import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from random import choice
from dataset.pascalvoc import PascalVOCV2


class DataLoader(DataLoader):
    def __init__(self, batch_size, key, is_train=True, **kargs):
        if 'yolov2-voc' == key:
            if is_train:
                paths = [
                    './data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
                    './data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
                ]
                tfs = transforms.Compose([
                    transforms.ToTensor()])
            else:
                paths = [
                    './data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
                ]
                tfs = transforms.Compose([
                    transforms.ToTensor()])
            sizes = kargs['sizes']
            input_h, input_w = choice(sizes)
            dataset = PascalVOCV2(
                data_list_paths=paths,
                input_h=input_h,
                input_w=input_w,
                transforms=tfs)

        super(DataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=os.cpu_count(),
            collate_fn=dataset.collate_fn
        )
