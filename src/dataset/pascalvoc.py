import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image


class PascalVOC(Dataset):
    def __init__(self, data_list_paths, input_h, input_w, transforms=None):
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
        self.transforms = transforms

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

        # transforms
        if self.transforms is not None:
            image = self.transforms(image)

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
