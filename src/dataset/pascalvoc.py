import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from torchvision.ops import box_convert, box_iou
from PIL import Image


class PascalVOC(Dataset):
    def __init__(self, data_list_paths, input_h, input_w, transform=None):
        super(PascalVOC, self).__init__()
        self.data_list = self._get_data_list(data_list_paths)
        self.input_h, self.input_w = input_h, input_w
        self.labels = [
            'aeroplane',
            'bicycle',
            'bird',
            'boat',
            'bottle',
            'bus',
            'car',
            'cat',
            'chair',
            'cow',
            'diningtable',
            'dog',
            'horse',
            'motorbike',
            'person',
            'pottedplant',
            'sheep',
            'sofa',
            'train',
            'tvmonitor'
        ]
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, anno_path = self.data_list[idx]

        # get data
        image = Image.open(image_path).resize((self.input_h, self.input_w))

        anno = []

        root = ET.parse(anno_path).getroot()
        org_w = int(root.find('size').find('width').text)
        org_h = int(root.find('size').find('height').text)
        x_scale = self.input_w / org_w
        y_scale = self.input_h / org_h

        for obj in root.iter('object'):
            name = obj.find('name').text
            class_id = self.labels.index(name)

            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text) * x_scale
            ymin = int(bbox.find('ymin').text) * y_scale
            xmax = int(bbox.find('xmax').text) * x_scale
            ymax = int(bbox.find('ymax').text) * y_scale

            difficult = int(obj.find('difficult').text)

            anno.append(torch.tensor([xmin, ymin, xmax, ymax, class_id, difficult, 0]))

        if len(anno) > 0:
            anno = torch.stack(anno)
        else:
            anno = torch.empty(size=(0, 7))

        # transform
        if self.transform is not None:
            image = self.transform(image)

        return image, anno

    def _get_data_list(self, data_list_paths):
        data_list = []
        for data_list_path in data_list_paths:
            image_dir = data_list_path.resolve().parents[2] / 'JPEGImages'
            anno_dir = data_list_path.resolve().parents[2] / 'Annotations'
            with open(data_list_path, 'r') as f:
                file_names = f.read().split('\n')[:-1]
            data_list += [
                (image_dir / f'{file_name}.jpg', anno_dir / f'{file_name}.xml')
                for file_name in file_names]

        return data_list


class PascalVOCV2(PascalVOC):
    def __init__(self, data_list_paths, input_h, input_w, transforms=None):
        super(PascalVOCV2, self).__init__(data_list_paths, input_h, input_w, transforms)
        self.anchors = torch.tensor([
            [1.3221, 1.73145],
            [3.19275, 4.00944],
            [5.05587, 8.09892],
            [9.47112, 4.84053],
            [11.2364, 10.0071]
        ])
        self.grid_unit = 32
        self.grid_h = self.input_h // self.grid_unit
        self.grid_w = self.input_w // self.grid_unit

        self.collate_fn = None

    def __getitem__(self, idx):
        image, anno = super(PascalVOCV2, self).__getitem__(idx)

        gt = torch.zeros((self.grid_h * self.grid_w * len(self.anchors), anno.shape[1]))
        mask = torch.zeros((self.grid_h * self.grid_w * len(self.anchors)))

        if len(anno) > 0:
            anno[:, :4] /= self.grid_unit
            cx, cy = torch.meshgrid(torch.arange(0.5, self.grid_w), torch.arange(0.5, self.grid_h))
            cx = cx.t().contiguous().view(-1, 1)  # transpose because anchors to be organized in H x W order
            cy = cy.t().contiguous().view(-1, 1)

            centers = torch.cat([cx, cy], axis=1).float()

            all_anchors = torch.cat([
                centers.view(-1, 1, 2).expand(-1, len(self.anchors), 2),
                self.anchors.view(1, -1, 2).expand(self.grid_h * self.grid_w, -1, 2)
            ], axis=2).view(-1, 4)
            all_anchors = box_convert(all_anchors, in_fmt='cxcywh', out_fmt='xyxy')

            indices = box_iou(anno[:, :4], all_anchors).max(dim=1).indices
            gt[indices] = anno
            mask[indices] = 1.

        return image, gt, mask
