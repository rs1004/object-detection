import torch
import json
from torch.utils.data import Dataset
from PIL import Image
from pathlib import PosixPath


class ImageAnnotationSet(Dataset):
    def __init__(self, data_dir, input_size, transforms=None):
        super(ImageAnnotationSet, self).__init__()
        self.data_list = self._get_data_list(data_dir)
        self.labels = self._get_labels(data_dir)
        self.input_size = input_size
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> tuple:
        """
        Args:
            idx (int): index of data list

        Returns:
            tuple: (Image, Annotation)
                * Image:      PIL.Image
                * Annotation: torch.tensor (n, 6)
        """
        image_path, anno_path = self.data_list[idx]

        # get data
        image = Image.open(image_path).resize((self.input_size, self.input_size))

        with open(anno_path, 'r') as f:
            anno_d = json.load(f)

        org_w = anno_d['image']['width']
        org_h = anno_d['image']['height']
        x_scale = self.input_size / org_w
        y_scale = self.input_size / org_h

        annotation = []
        for anno in anno_d['annotation']:
            name = anno['category']
            class_id = self.labels.index(name)

            xmin = anno['bbox'][0] * x_scale
            ymin = anno['bbox'][1] * y_scale
            xmax = anno['bbox'][2] * x_scale
            ymax = anno['bbox'][3] * y_scale

            void = anno['void']

            annotation.append(torch.tensor([xmin, ymin, xmax, ymax, class_id, void]))

        if len(annotation) > 0:
            annotation = torch.stack(annotation)
        else:
            annotation = torch.empty(size=(0, 6))

        # transforms
        if self.transforms is not None:
            image, annotation = self.transforms(image, annotation)

        return image, annotation

    def _get_data_list(self, data_dir: PosixPath) -> list:
        """
        Args:
            data_dir (PosixPath): data directory path

        Returns:
            list: [(image_path, anno_path), ...]
        """
        image_paths = sorted(data_dir.glob('*.jpg'))
        anno_paths = sorted(data_dir.glob('*.json'))

        data_list = [(ip.resolve().as_posix(), ap.resolve().as_posix()) for ip, ap in zip(image_paths, anno_paths)]

        return data_list

    def _get_labels(self, data_dir: PosixPath) -> list:
        """
        Args:
            data_dir (PosixPath): data directory path

        Returns:
            list: label list
        """
        with open(data_dir.parent / 'labels', 'r') as f:
            labels = f.read().split('\n')
        return labels


if __name__ == '__main__':
    from pathlib import Path

    data_dir = Path('/home/sato/work/yolo/data/pascal_voc/validation')
    ds = ImageAnnotationSet(data_dir, 416, None)
    print(ds.data_list[0])
    for _ in ds:
        print(_)
        break
