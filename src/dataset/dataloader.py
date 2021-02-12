import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from random import choice
from pathlib import Path
from dataset.pascalvoc import PascalVOCV2


class DataLoader(DataLoader):
    def __init__(self, batch_size, key, is_train=True, **cfg):
        self.__dict__.update(cfg)
        self.batch_size = batch_size
        self.is_train = is_train
        self.num_workers = os.cpu_count()

        if 'yolov2-voc' == key:
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
            input_h, input_w = choice(self.sizes)
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


if __name__ == '__main__':
    import json
    with open('src/config.json', 'r') as f:
        cfg = json.load(f)['yolov2-voc']['common']
    cfg['sizes'] = [[416, 416]]
    dataloader = DataLoader(32, 'yolov2-voc', is_train=True, **cfg)

    for i in dataloader:
        print(i)
        break
