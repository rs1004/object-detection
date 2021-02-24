import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torchvision.ops import batched_nms, box_iou
from collections import Counter


class Evaluate:
    def __init__(self, model, dataloader, **cfg):
        self.model = model
        self.dataloader = dataloader

        self.__dict__.update(cfg)

    def run(self):
        save_dir = Path(self.result_dir) / 'evaluate'
        save_dir.mkdir(exist_ok=True, parents=True)

        if self.mpolicy == 'greedy':
            correct_thresholds = [0.5]
            recall_thresholds = None
        elif self.mpolicy == 'soft':
            correct_thresholds = np.arange(0.5, 1.0, 0.05)
            recall_thresholds = np.arange(0., 1.01, 0.01)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        labels = self.dataloader.dataset.labels
        metric_fn = MeanAveragePrecision(num_classes=len(labels), nms_thresh=self.nms_thresh)
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
            recall_thresholds=recall_thresholds,
            mpolicy=self.mpolicy
        )

        with open(save_dir / 'evaluate.json', 'w') as f:
            json.dump(result, f, indent=4)

        s_df = pd.DataFrame([[result[ct][i]['ap'] for ct in correct_thresholds] for i in range(len(labels))])
        s_df.index = labels
        s_df.columns = [f'th = {ct:.02f}' for ct in correct_thresholds]

        s_df.loc['AP'] = s_df.mean(axis=0)
        s_df['mean'] = s_df.mean(axis=1)

        with open(save_dir / 'summary.txt', 'w') as f:
            f.write(s_df.applymap('{:.03f}'.format).to_markdown(tablefmt='grid'))


class MeanAveragePrecision:
    def __init__(self, num_classes, nms_thresh):
        self.num_classes = num_classes
        self.nms_thresh = nms_thresh

        self.interm = []
        self.count = []

    def stack(self, pred, gt):
        boxes = pred[:, :4]
        class_ids = pred[:, 5:].max(dim=-1).indices
        confs = pred[:, 4]

        gt_boxes = gt[:, :4]
        gt_class_ids = gt[:, 4]

        # leave valid bboxes
        ids = batched_nms(boxes, confs, class_ids, iou_threshold=self.nms_thresh)
        boxes = boxes[ids]
        confs = confs[ids]
        class_ids = class_ids[ids]

        # void
        void_id = torch.where((gt[:, 5:] == 1).any(dim=1))[0]
        non_void_id = torch.where((gt[:, 5:] == 0).all(dim=1))[0]
        count = gt_class_ids[non_void_id]

        # assign bboxes
        if len(boxes) > 0:
            ious, bbox_ids = box_iou(boxes, gt_boxes).max(dim=1)
            ious *= torch.tensor([bbox_ids[i] not in bbox_ids[:i] for i in range(len(bbox_ids))])
            ious *= class_ids == gt_class_ids[bbox_ids]
            interm = torch.stack([ious, confs, class_ids, bbox_ids], dim=1)
            interm = interm[(interm[:, 3] != void_id.reshape(-1, 1)).all(dim=0), :3]  # void id でない行のみを残す.
        else:
            interm = torch.empty(size=(0, 3))

        self.interm.append(interm.clone())
        self.count.append(count.clone())

    def value(self, correct_thresholds, recall_thresholds=None, mpolicy='greedy'):
        # shape data
        interm = torch.cat(self.interm)
        count = torch.cat(self.count)

        ious, _, class_ids = interm[interm[:, 1].argsort(descending=True)].T
        interm = torch.stack([ious > ct for ct in correct_thresholds] + [class_ids], dim=1)
        count = Counter(count.tolist())

        # calculate
        result = {}
        aps = torch.empty(size=(0, len(correct_thresholds)))
        for i in range(self.num_classes):
            interm_ = interm[interm[:, -1] == i, :-1]

            pres = interm_.cumsum(dim=0) / torch.arange(1, len(interm_) + 1).reshape(-1, 1)
            recs = interm_.cumsum(dim=0) / max(count[i], 1)

            aps_a_class = self._calc_ap(pres, recs, recall_thresholds, mpolicy)
            for _, (ct, ap) in enumerate(zip(correct_thresholds, aps_a_class)):
                if ct not in result:
                    result[ct] = {}
                result[ct][i] = {
                    'ap': float(ap),
                    'precision': pres[:, _].tolist(),
                    'recall': recs[:, _].tolist()
                }
            aps = torch.cat([aps, aps_a_class.reshape(1, -1)])

        result['mAP'] = float(aps.mean(axis=0).mean())

        return result

    def _calc_ap(self, pres, recs, recall_thresholds, mpolicy):
        if mpolicy == 'greedy':
            pres = torch.vstack([
                torch.zeros(size=(1, pres.shape[1])),
                pres,
                torch.zeros(size=(1, pres.shape[1]))
            ])
            recs = torch.vstack([
                torch.zeros(size=(1, recs.shape[1])),
                recs,
                torch.ones(size=(1, recs.shape[1]))
            ])

            ad_pres = pres.flip(dims=(0,)).cummax(dim=0).values.flip(dims=(0,))
            average_precision = (ad_pres[1:] * (recs[1:] - recs[:-1])).sum(dim=0)

        elif mpolicy == 'soft':
            average_precision = torch.stack([
                torch.where(recs >= rt, pres, torch.zeros_like(pres)).max(dim=0).values for rt in recall_thresholds
            ]).mean(dim=0)

        return average_precision
