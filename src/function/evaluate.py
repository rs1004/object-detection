import torch
import numpy as np
import json
from tqdm import tqdm
from torchvision.ops import batched_nms, box_iou
from collections import Counter


class Evaluate:
    def __init__(self, model, dataloader, **cfg):
        self.model = model
        self.dataloader = dataloader

        self.__dict__.update(cfg)

    def run(self):
        correct_thresholds = [0.5]
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        labels = self.dataloader.dataset.labels
        metric_fn = MeanAveragePrecision(num_classes=len(labels), conf_thresh=self.conf_thresh, iou_thresh=self.iou_thresh)
        with torch.no_grad():
            for images, gts, masks in tqdm(self.dataloader, total=len(self.dataloader)):
                # to GPU device
                images = images.to(device)

                # forward
                outputs = self.model(images)

                # calculate average precision
                outputs = outputs.to('cpu')

                for output, gt, mask in zip(outputs, gts, masks):
                    metric_fn.stack(pred=output, gt=gt[mask == 1])

        result = metric_fn.value(
            correct_thresholds=correct_thresholds,
            recall_thresholds=self.recall_thresholds,
            mpolicy=self.mpolicy
        )

        with open('evaluate.json', 'w') as f:
            json.dump(result, f, indent=4)


class MeanAveragePrecision:
    def __init__(self, num_classes, conf_thresh, iou_thresh):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.interm = []
        self.count = []

    def stack(self, pred, gt):
        boxes = pred[:, :4]
        class_ids = pred[:, 5:].max(dim=-1).indices
        confs = pred[:, 4]

        gt_boxes = gt[:, :4]
        gt_class_ids = gt[:, 4]

        # leave valid bboxes
        boxes = boxes[confs > self.conf_thresh]
        class_ids = class_ids[confs > self.conf_thresh]
        confs = confs[confs > self.conf_thresh]

        ids = batched_nms(boxes, confs, class_ids, iou_threshold=self.iou_thresh)
        boxes = boxes[ids]
        confs = confs[ids]
        class_ids = class_ids[ids]

        # void
        void_id = torch.where((gt[:, 5:] == 1).any(dim=1))[0]
        count = gt_class_ids[void_id]

        # assign bboxes
        if len(boxes) > 0:
            ious, bbox_ids = box_iou(boxes, gt_boxes).max(dim=1)
            ious *= torch.tensor([bbox_ids[i] not in bbox_ids[:i] for i in range(len(bbox_ids))])
            ious *= class_ids == gt_class_ids[bbox_ids]
            interm = torch.stack([ious, confs, class_ids, bbox_ids], dim=1)
            interm = interm[(interm[:, 3] != void_id.reshape(-1, 1)).all(dim=0), :3]  # void id でない行のみを残す.
        else:
            interm = torch.empty(size=(0, 3))

        self.interm.append(interm.numpy().copy())
        self.count.append(count.numpy().copy())

    def value(self, correct_thresholds, recall_thresholds=None, mpolicy='greedy'):
        if isinstance(correct_thresholds, float):
            correct_thresholds = [correct_thresholds]

        # shape data
        interm = np.concatenate(self.interm)
        count = np.concatenate(self.count)

        ious, _, class_ids = interm[np.argsort(interm[:, 1])[::-1]].T
        interm = np.stack([ious > ct for ct in correct_thresholds] + [class_ids], axis=1)
        count = Counter(count.tolist())

        # calculate
        result = {}
        aps = np.empty(shape=(0, len(correct_thresholds)))
        for i in range(self.num_classes):
            interm_ = interm[interm[:, -1] == i]

            pres = np.vstack([
                np.zeros((1, len(correct_thresholds))),
                np.cumsum(interm_, axis=0) / np.arange(1, len(interm_) + 1).reshape(-1, 1),
                np.zeros((1, len(correct_thresholds)))
            ])
            recs = np.vstack([
                np.zeros((1, len(correct_thresholds))),
                np.cumsum(interm_, axis=0) / count[i],
                np.ones((1, len(correct_thresholds)))
            ])

            aps_a_class = self._calc_ap(pres, recs, recall_thresholds, mpolicy)
            for j, (ct, ap) in enumerate(zip(correct_thresholds, aps_a_class)):
                if ct not in result:
                    result[ct] = {}
                result[ct][i] = {
                    'ap': ap,
                    'precision': pres[:, j],
                    'recall': recs[:, j]
                }
            aps = np.concatenate([aps, aps_a_class.reshape(1, -1)], axis=0)

        result['mAP'] = aps.mean(axis=0).mean()

        return result

    def _calc_ap(self, pres, recs, recall_thresholds, mpolicy):
        if mpolicy == 'greedy':
            pass
        elif mpolicy == 'soft':
            pass
