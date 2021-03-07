"""
Set as below:

somewhere
    ├ test
    ├ train
    ├ valid
    └ convert_maskw.py
"""

from pathlib import Path
from shutil import copy
from tqdm import tqdm
import json

dst_dir = Path('/home/sato/work/yolo/data/maskw')

for t in ['train', 'valid']:
    data_dir = Path(__file__).parent / t
    dst_dir_ = dst_dir / t.replace('valid', 'validation')
    dst_dir_.mkdir(parents=True, exist_ok=True)
    path = Path(__file__).parent / t / '_annotations.coco.json'

    with open(path, 'r') as f:
        d = json.load(f)

    categories = {c['id']: c['name'] for c in d['categories']}

    anno_d = {}
    for anno in d['images']:
        anno_d[anno['id']] = {
            'file_name': anno['file_name'],
            'image': {
                'height': anno['height'],
                'width': anno['width']
            },
            'annotation': []
        }

    for anno in d['annotations']:
        anno_d[anno['image_id']]['annotation'].append({
            'category': categories[anno['category_id']],
            'bbox': [
                anno['bbox'][0],
                anno['bbox'][1],
                anno['bbox'][2] + anno['bbox'][0],
                anno['bbox'][3] + anno['bbox'][1]
            ],
            'void': anno['iscrowd']
        })

    for _, anno in tqdm(anno_d.items()):
        file_name = anno['file_name']
        del anno['file_name']
        copy(data_dir / file_name, dst_dir_)
        with open(dst_dir_ / (file_name[:-4] + '.json'), 'w') as f:
            json.dump(anno, f, indent=4)
