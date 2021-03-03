import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DL
from random import choice
from functools import partial
from pathlib import Path
from dataset.pascalvoc import PascalVOC
from dataset.coco import Coco
from dataset.maskwearing import MaskWearing
from torchvision.ops import box_convert, box_iou


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
            self.dataset = PascalVOC(
                data_list_paths=paths,
                input_h=input_h,
                input_w=input_w,
                transforms=tfs)
            anchors = torch.tensor(self.anchors)
            collate_fn = partial(collate_yolov2, anchors=anchors, input_h=input_h, input_w=input_w)

        elif 'yolov2-coco' == self.key:
            data_dir = Path(self.data_dir) / 'coco'
            if self.is_train:
                data_name = 'train2014'
                tfs = transforms.Compose([
                    transforms.ToTensor()])
            else:
                data_name = 'val2014'
                tfs = transforms.Compose([
                    transforms.ToTensor()])
            input_h, input_w = choice(self.sizes[0 if self.is_train else 1])
            self.dataset = Coco(
                data_dir=data_dir / data_name,
                input_h=input_h,
                input_w=input_w,
                transforms=tfs)
            anchors = torch.tensor(self.anchors)
            collate_fn = partial(collate_yolov2, anchors=anchors, input_h=input_h, input_w=input_w)

        elif 'yolov2-maskw' == self.key:
            data_dir = Path(self.data_dir) / 'maskWearing'
            if self.is_train:
                data_name = 'train'
                tfs = transforms.Compose([
                    transforms.ToTensor()])
            else:
                data_name = 'valid'
                tfs = transforms.Compose([
                    transforms.ToTensor()])
            input_h, input_w = choice(self.sizes[0 if self.is_train else 1])
            self.dataset = MaskWearing(
                data_dir=data_dir / data_name,
                input_h=input_h,
                input_w=input_w,
                transforms=tfs)
            anchors = torch.tensor(self.anchors)
            collate_fn = partial(collate_yolov2, anchors=anchors, input_h=input_h, input_w=input_w)

        else:
            raise NotImplementedError(f'{self.key} is not expected')

        super(DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.is_train,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

    def reset(self):
        self = self.__class__(**self._init_d)


def collate_yolov2(batch, anchors, input_h, input_w):
    grid_h = input_h // 32
    grid_w = input_w // 32

    images = []
    gts = []
    masks = []
    for image, anno in batch:
        gt = torch.zeros((grid_h * grid_w * len(anchors), anno.shape[1]))
        mask = torch.zeros((grid_h * grid_w * len(anchors)))

        if len(anno) > 0:
            anno[:, :4] /= 32
            cx, cy = torch.meshgrid(torch.arange(0.5, grid_w), torch.arange(0.5, grid_h))
            cx = cx.t().contiguous().view(-1, 1)  # transpose because anchors to be organized in H x W order
            cy = cy.t().contiguous().view(-1, 1)

            centers = torch.cat([cx, cy], dim=1).float()

            all_anchors = torch.cat([
                centers.view(-1, 1, 2).expand(-1, len(anchors), 2),
                anchors.view(1, -1, 2).expand(grid_h * grid_w, -1, 2)
            ], dim=2).view(-1, 4)
            all_anchors = box_convert(all_anchors, in_fmt='cxcywh', out_fmt='xyxy')

            indices = box_iou(anno[:, :4], all_anchors).max(dim=1).indices
            gt[indices] = anno
            mask[indices] = 1.

        images.append(image)
        gts.append(gt)
        masks.append(mask)

    images = torch.stack(images, dim=0)
    gts = torch.stack(gts, dim=0)
    masks = torch.stack(masks, dim=0)

    return images, gts, masks


if __name__ == '__main__':
    import json
    with open('src/config.json', 'r') as f:
        cfg = json.load(f)['yolov2-maskw']
    cfg = dict(cfg['common'], **cfg['dataloader'])
    dataloader = DataLoader(32, 'yolov2-maskw', is_train=True, **cfg)

    for image, gt, mask in dataloader:
        print(image.shape)
        print(gt.shape)
        print(mask.shape)
        break
