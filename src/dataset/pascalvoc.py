import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from torchvision.ops import box_convert
from PIL import Image
from function import calc_iou


class PascalVOC(Dataset):
    def __init__(self, data_list_paths, input_h, input_w, transform):
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
            label_id = self.labels.index(name)

            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text) * x_scale
            ymin = int(bbox.find('ymin').text) * y_scale
            xmax = int(bbox.find('xmax').text) * x_scale
            ymax = int(bbox.find('ymax').text) * y_scale

            coord = torch.tensor([xmin, ymin, xmax, ymax])
            coord = box_convert(coord, in_fmt='xyxy', out_fmt='cxcywh')

            anno.append([label_id, coord])

        # transform
        for t in self.transform:
            image = t(image)

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
        super(PascalVOCV2, self).__init__(data_list_paths, input_h, input_w)
        self.anchors = [
            [1.3221, 1.73145],
            [3.19275, 4.00944],
            [5.05587, 8.09892],
            [9.47112, 4.84053],
            [11.2364, 10.0071]
        ]
        self.grid_unit = 32
        self.grid_h = self.input_h // self.grid_unit
        self.grid_w = self.input_w // self.grid_unit

    def __getitem__(self, idx):
        image, anno = super(PascalVOCV2, self).__getitem__(idx)

        gt = torch.zeros((self.grid_h, self.grid_w, len(self.anchors), 5 + 1))
        mask = torch.zeros((self.grid_h, self.grid_w, len(self.anchors)))

        for label_id, coord in anno:
            coord /= self.grid_unit

            # select anchor (an highest iou anchor is selected)
            cx, cy, w, h = coord

            anchors = torch.cat(
                [torch.tensor([cx, cy]).repeat(len(self.anchors), 1) // 1 + 0.5,
                 self.anchors],
                axis=1)
            anchor_id = calc_iou(coord, anchors).argmax()

            # set annotation
            w_id = cx.int()
            h_id = cy.int()

            gt[h_id, w_id, anchor_id, :5] += torch.tensor([cx, cy, w, h, 1.])
            gt[h_id, w_id, anchor_id, 5] += label_id
            mask[h_id, w_id, anchor_id] = 1.

        gt = gt.view(-1, 5 + 1)  # (h * w * num_anchors, [cx, cy, w, h, conf, class_id])
        mask = mask.view(-1)

        return image, gt, mask

    def collate_fn(self, batch):
        images = []
        gts = []
        masks = []
        for image, gt, mask in batch:
            images.append(image)
            gts.append(gt)
            masks.append(mask)

        images = torch.stack(images, dim=0)
        gts = torch.stack(gts, dim=0)
        masks = torch.stack(masks, dim=0)

        return images, gts, masks
