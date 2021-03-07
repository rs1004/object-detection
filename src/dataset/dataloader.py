import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DL
from functools import partial
from pathlib import Path
from dataset import ImageAnnotationSet
from torchvision.ops import box_convert, box_iou


class DataLoader(DL):
    def __init__(self, model_name, is_train, **cfg):
        self.__dict__.update(cfg)

        if is_train:
            data_dir = Path(self.data_dir) / 'train'
            tfs = transforms.Compose([
                transforms.ToTensor()])
        else:
            data_dir = Path(self.data_dir) / 'validation'
            tfs = transforms.Compose([
                transforms.ToTensor()])

        dataset = ImageAnnotationSet(
            data_dir=data_dir,
            input_size=self.input_size,
            transforms=tfs
        )
        if model_name == 'yolov2':
            anchors = torch.tensor(self.anchors)
            collate_fn = partial(collate_yolov2, anchors=anchors, input_size=self.input_size)

        else:
            raise NotImplementedError(f'{self.key} is not expected')

        super(DataLoader, self).__init__(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=is_train,
            num_workers=os.cpu_count(),
            collate_fn=collate_fn
        )


def collate_yolov2(batch, anchors, input_size):
    grid_h = input_size // 32
    grid_w = input_size // 32
    anchors *= input_size / 32

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
    cfg = {
        'data_dir': '/home/sato/work/yolo/data/mask_wearing',
        'input_size': 416,
        'batch_size': 1,
        'anchors': [
            [0.09583878, 0.18330716],
            [0.16641968, 0.29788929],
            [0.04672656, 0.08603257],
            [0.26187747, 0.44968904],
            [0.46062868, 0.75800859]
        ]
    }
    dataloader = DataLoader('yolov2', is_train=True, **cfg)

    for image, gt, mask in dataloader:
        print(image.shape)
        print(gt.shape)
        print(mask.shape)
        break
