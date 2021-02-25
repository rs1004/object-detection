import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DL
from random import choice
from pathlib import Path
from dataset.pascalvoc import PascalVOCV2


class DataLoader(DL):
    def __init__(self, batch_size, key, is_train=True, **cfg):
        self.batch_size = batch_size
        self.key = key
        self.is_train = is_train
        self.num_workers = os.cpu_count()
        self.__dict__.update(cfg)
        self._init_d = self.__dict__.copy()

        if 'yolov2-voc' == self.key:
            if self.is_train:
                paths = [
                    Path(self.data_dir) / 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
                    Path(self.data_dir) / 'VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
                ]
                tfs = transforms.Compose([
                    transforms.ToTensor()])
            else:
                paths = [
                    Path(self.data_dir) / 'VOCdevkit/VOC2007/ImageSets/Main/test.txt'
                ]
                tfs = transforms.Compose([
                    transforms.ToTensor()])
            input_h, input_w = choice(self.sizes[0 if self.is_train else 1])
            self.dataset = PascalVOCV2(
                data_list_paths=paths,
                input_h=input_h,
                input_w=input_w,
                transforms=tfs)

        super(DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.is_train,
            num_workers=self.num_workers,
            collate_fn=self.dataset.collate_fn
        )

    def reset(self):
        self = self.__class__(**self._init_d)


if __name__ == '__main__':
    import json
    with open('src/config.json', 'r') as f:
        cfg = json.load(f)['yolov2-voc']['common']
    cfg['sizes'] = [[416, 416]]
    dataloader = DataLoader(32, 'yolov2-voc', is_train=True, **cfg)

    for image, gt, mask in dataloader:
        print(image.shape)
        print(gt.shape)
        print(mask.shape)
        break
