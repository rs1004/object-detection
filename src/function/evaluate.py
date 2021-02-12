import torch
import numpy as np
from tqdm import tqdm
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
        result = {label: [] for label in labels}
        count = 0
        with torch.no_grad():
            for images, gts, _ in tqdm(self.dataloader, total=len(self.dataloader)):
                # to GPU device
                images = images.to(device)

                # forward
                outputs = self.model(images)

                # restore
                outputs = outputs.to('cpu')
                for b in range(len(outputs)):
                    gt = gts[b]

                    boxes = box_convert(outputs[b, :, :4], in_fmt='cxcywh', out_fmt='xyxy')
                    confs = outputs[b, :, 4]
                    class_ids = outputs[b, :, 5:].max(dim=-1).indices

                    # leave valid bboxes
                    boxes = boxes[confs > self.conf_thresh]
                    class_ids = class_ids[confs > self.conf_thresh]
                    confs = confs[confs > self.conf_thresh]

                    ids = batched_nms(boxes, confs, class_ids, iou_threshold=self.iou_thresh)
                    boxes = boxes[ids]
                    confs = confs[ids]
                    class_ids = class_ids[ids]

                    for i in range(len(labels)):
                        gt_box = box_convert(gt[gt[:, 5] == i][:, :4], in_fmt='cxcywh', out_fmt='xyxy')
                        box = boxes[class_ids == i]
                        conf = confs[class_ids == i]

                        if len(box) > 0:
                            if len(gt_box) > 0:
                                iou = box_iou(box, gt_box)
                                chosen = set()
                                for j in range(len(iou)):
                                    max_iou, index = iou[j].max(axis=0)
                                    if max_iou > self.correct_thresh and index not in chosen:
                                        result[labels[i]].append(np.array([1, conf[j].numpy()]))
                                        chosen.add(index)
                                    else:
                                        result[labels[i]].append(np.array([0, conf[j].numpy()]))
                            else:
                                result[labels[i]].append(np.stack([np.zeros_like(conf.numpy()), conf.numpy()], axis=1))

                count += 1
                if count > 10:
                    break

        for name, res in result.items():
            if len(res) > 0:
                res = np.stack(res, axis=0)
                ap = average_precision_score(res[:, 0], res[:, 1])
            else:
                ap = 0
            print(name, ap)
