import torch
import json
from torch.utils.data import Dataset
from PIL import Image


class MaskWearing(Dataset):
    def __init__(self, data_dir, input_h, input_w, transforms=None):
        super(MaskWearing, self).__init__()
        self.data_list = self._get_data_list(data_dir)
        self.input_h, self.input_w = input_h, input_w
        self.labels = ['People', 'mask', 'no-mask']
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, anno_path = self.data_list[idx]

        # get data
        image = Image.open(image_path).resize((self.input_h, self.input_w))

        with open(anno_path, 'r') as f:
            anno_d = json.load(f)

        org_w = anno_d['width']
        org_h = anno_d['height']
        x_scale = self.input_w / org_w
        y_scale = self.input_h / org_h

        anno = []
        for obj in anno_d['object']:
            name = obj['category']
            class_id = self.labels.index(name)

            xmin = obj['bbox'][0] * x_scale
            ymin = obj['bbox'][1] * y_scale
            xmax = (xmin + obj['bbox'][2]) * x_scale
            ymax = (ymin + obj['bbox'][3]) * y_scale

            iscrowd = obj['iscrowd']

            anno.append(torch.tensor([xmin, ymin, xmax, ymax, class_id, 0, iscrowd]))

        if len(anno) > 0:
            anno = torch.stack(anno)
        else:
            anno = torch.empty(size=(0, 7))

        # transforms
        if self.transforms is not None:
            image = self.transforms(image)

        return image, anno

    def _get_data_list(self, data_dir):
        image_paths = sorted(data_dir.glob('*.jpg'))
        anno_paths = sorted(data_dir.glob('*.json'))

        data_list = [(ip.resolve().as_posix(), ap.resolve().as_posix()) for ip, ap in zip(image_paths, anno_paths)]

        return data_list
