import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from torchvision.ops import box_convert, batched_nms, box_iou
from sklearn.metrics import average_precision_score


class Evaluate:
    def __init__(self, model, dataloader, **cfg):
        self.model = model
        self.dataloader = dataloader

        self.__dict__.update(cfg)

    def run(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        labels = self.dataloader.dataset.labels
        results = []
        with torch.no_grad():
            for images, gts, _ in tqdm(self.dataloader, total=len(self.dataloader)):
                # to GPU device
                images = images.to(device)

                # forward
                outputs = self.model(images)

                # calculate average precision
                outputs = outputs.to('cpu')

                f = partial(self._calc_ap, labels=labels)
                with Pool() as p:
                    results += p.starmap(f, zip(outputs, gts))

        results = {label: np.concatenate([r[label] for r in results]) for label in labels}

        aps = []
        for name, res in results.items():
            ap = average_precision_score(res[:, 0], res[:, 1])
            aps.append(ap)
            print(name, ap)
        print('mAP', sum(aps) / len(aps))

    def _calc_ap(self, output, gt, labels):
        boxes = box_convert(output[:, :4], in_fmt='cxcywh', out_fmt='xyxy')
        confs = output[:, 4]
        class_ids = output[:, 5:].max(dim=-1).indices

        # sort
        boxes = boxes[confs.sort(descending=True).indices]
        class_ids = class_ids[confs.sort(descending=True).indices]
        confs = confs[confs.sort(descending=True).indices]

        # leave valid bboxes
        boxes = boxes[confs > self.conf_thresh]
        class_ids = class_ids[confs > self.conf_thresh]
        confs = confs[confs > self.conf_thresh]

        ids = batched_nms(boxes, confs, class_ids, iou_threshold=self.iou_thresh)
        boxes = boxes[ids]
        confs = confs[ids]
        class_ids = class_ids[ids]

        result = {label: np.empty(shape=(0, 2)) for label in labels}
        for i in range(len(labels)):
            gt_box = box_convert(gt[gt[:, 5] == i][:, :4], in_fmt='cxcywh', out_fmt='xyxy')
            box = boxes[class_ids == i]
            conf = confs[class_ids == i]

            r = []
            if len(box) > 0:
                if len(gt_box) > 0:
                    iou = box_iou(box, gt_box)
                    chosen = set()
                    for j in range(len(iou)):
                        max_iou, index = iou[j].max(axis=0)
                        if max_iou > self.correct_thresh and index not in chosen:
                            r.append(np.array([[1, conf[j].numpy()]]))
                            chosen.add(index)
                        else:
                            r.append(np.array([[0, conf[j].numpy()]]))
                else:
                    r.append(np.stack([[0, c] for c in conf.numpy()]))
            if len(r) > 0:
                result[labels[i]] = np.concatenate(r)

        return result
