"""
Set as below:

somewhere
    ├ annotations
    ├ train2014
    ├ val2014
    └ convert_coco.py
"""

from pathlib import Path
from shutil import copy
from tqdm import tqdm
import json

dst_dir = Path('/home/sato/work/yolo/data/coco')

for t in ['train', 'val']:
    data_dir = Path(__file__).parent / f'{t}2014'
    dst_dir_ = dst_dir / t.replace('val', 'validation')
    dst_dir_.mkdir(parents=True, exist_ok=True)
    path = Path(__file__).parent / 'annotations' / f'instances_{t}2014.json'

    with open(path, 'r') as f:
        d = json.load(f)

    categories = {c['id']: c['name'] for c in d['categories']}

    anno_d = {}
    for anno in d['images']:
        anno_d[anno['file_name']] = {
            'image': {
                'height': anno['height'],
                'width': anno['width']
            },
            'annotation': []
        }

    for anno in d['annotations']:
        file_name = f'COCO_{t}2014_{anno["image_id"]:012}.jpg'
        anno_d[file_name]['annotation'].append({
            'category': categories[anno['category_id']],
            'bbox': [
                anno['bbox'][0],
                anno['bbox'][1],
                anno['bbox'][2] + anno['bbox'][0],
                anno['bbox'][3] + anno['bbox'][1]
            ],
            'void': anno['iscrowd']
        })

    for file_name, anno in tqdm(anno_d.items()):
        copy(data_dir / file_name, dst_dir_)
        with open(dst_dir_ / (file_name[:-4] + '.json'), 'w') as f:
            json.dump(anno, f, indent=4)
