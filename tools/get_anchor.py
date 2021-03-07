from pathlib import Path
import json
import numpy as np
from sklearn.cluster import KMeans

input_size = 416
data_dir = Path('/home/sato/work/yolo/data/mask_wearing')

whs = []
for p in data_dir.glob('**/*.json'):
    with open(p, 'r') as f:
        d = json.load(f)
    H, W = d['image']['height'], d['image']['width']
    for anno in d['annotation']:
        xmin, ymin, xmax, ymax = anno['bbox']
        whs.append([(xmax - xmin) / W, (ymax - ymin) / H])

whs = np.array(whs)
kmeans = KMeans(n_clusters=5, random_state=0).fit(whs)
print(kmeans.cluster_centers_ * input_size / 32)
