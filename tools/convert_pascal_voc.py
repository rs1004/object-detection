"""
Set as below:

somewhere
    ├ VOC2007
    ├ VOC2012
    └ convert_pascal_voc.py
"""

from pathlib import Path
from shutil import copy
from tqdm import tqdm
import xml.etree.ElementTree as ET
import json

dst_dir = Path('/home/sato/work/yolo/data/pascal_voc')

labels = [
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

paths_all = {
    'train': [
        Path(__file__).parent / 'VOC2007/ImageSets/Main/trainval.txt',
        Path(__file__).parent / 'VOC2012/ImageSets/Main/trainval.txt'
    ],
    'validation': [
        Path(__file__).parent / 'VOC2007/ImageSets/Main/test.txt',
    ]
}


for name, paths in paths_all.items():
    dst_dir_ = dst_dir / name
    dst_dir_.mkdir(parents=True, exist_ok=True)
    for path in paths:
        img_dir = path.parents[2] / 'JPEGImages'
        anno_dir = path.parents[2] / 'Annotations'
        with open(path, 'r') as f:
            d = f.read().split('\n')
        if len(d[-1]) == 0:
            d = d[:-1]
        for file_name in tqdm(d):
            img_path = img_dir / f'{file_name}.jpg'
            anno_path = anno_dir / f'{file_name}.xml'

            copy(img_path, dst_dir_)

            anno_d = {
                'image': {},
                'annotation': []
            }

            root = ET.parse(anno_path).getroot()
            for t in ['height', 'width']:
                anno_d['image'][t] = int(root.find('size').find(t).text)

            for obj in root.iter('object'):
                category = obj.find('name').text

                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                difficult = int(obj.find('difficult').text)

                anno_d['annotation'].append({
                    'category': category,
                    'bbox': [xmin, ymin, xmax, ymax],
                    'void': difficult
                })

            with open(dst_dir_ / f'{file_name}.json', 'w') as f:
                json.dump(anno_d, f, indent=4)
