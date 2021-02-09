import torch.nn as nn
import torch.nn.functional as F
from model import Model, Conv2dBNLeaky, Route, Reorg
from function import calc_iou


class YoloV2(Model):
    def __init__(self):
        super(YoloV2, self).__init__()

        self.features = nn.Sequential(
            Conv2dBNLeaky(c_in=3, c_out=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            Conv2dBNLeaky(c_in=32, c_out=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            Conv2dBNLeaky(c_in=64, c_out=128, kernel_size=3, stride=1, padding=1),
            Conv2dBNLeaky(c_in=128, c_out=64, kernel_size=1, stride=1),
            Conv2dBNLeaky(c_in=64, c_out=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            Conv2dBNLeaky(c_in=128, c_out=256, kernel_size=3, stride=1, padding=1),
            Conv2dBNLeaky(c_in=256, c_out=128, kernel_size=1, stride=1),
            Conv2dBNLeaky(c_in=128, c_out=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            Conv2dBNLeaky(c_in=256, c_out=512, kernel_size=3, stride=1, padding=1),
            Conv2dBNLeaky(c_in=512, c_out=256, kernel_size=1, stride=1),
            Conv2dBNLeaky(c_in=256, c_out=512, kernel_size=3, stride=1, padding=1),
            Conv2dBNLeaky(c_in=512, c_out=256, kernel_size=1, stride=1),
            Conv2dBNLeaky(c_in=256, c_out=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            Conv2dBNLeaky(c_in=512, c_out=1024, kernel_size=3, stride=1, padding=1),
            Conv2dBNLeaky(c_in=1024, c_out=512, kernel_size=1, stride=1),
            Conv2dBNLeaky(c_in=512, c_out=1024, kernel_size=3, stride=1, padding=1),
            Conv2dBNLeaky(c_in=1024, c_out=512, kernel_size=1, stride=1),
            Conv2dBNLeaky(c_in=512, c_out=1024, kernel_size=3, stride=1, padding=1),
            Conv2dBNLeaky(c_in=1024, c_out=1024, kernel_size=3, stride=1, padding=1),
            Conv2dBNLeaky(c_in=1024, c_out=1024, kernel_size=3, stride=1, padding=1),

            Route(layers=[-9]),
            Conv2dBNLeaky(c_in=512, c_out=64, kernel_size=1, stride=1),
            Reorg(stride=2),

            Route(layers=[-1, -4]),
            Conv2dBNLeaky(c_in=1280, c_out=1024, kernel_size=3, stride=1, padding=1)
        )
        self.detector = nn.Conv2d(in_channels=1024, out_channels=125, kernel_size=1, stride=1)

        self.route_queue = {}
        for i, m in enumerate(self.features):
            if isinstance(m, Route):
                for lnum in m.layers:
                    self.route_queue[lnum + i] = None

    def forward(self, x):
        for i, m in enumerate(self.features):
            if i in self.route_queue:
                self.route_queue[i] = m(x)
            if isinstance(m, Route):
                xs = [self.route_queue.pop(lnum + i) for lnum in m.layers]
                x = m(xs)
            else:
                x = m(x)
        out = self.detector(x)
        return out

    def loss(self, outputs, gts, masks, coefs: tuple):
        l_coord, l_obj, l_noobj, l_class = coefs
        b = outputs.shape[0]
        loss_xy = loss_wh = loss_obj = loss_noobj = loss_c = 0

        for i in range(b):
            ids = torch.nonzero(masks[i]).squeeze()
            non_ids = torch.nonzero(1 - masks[i]).squeeze()

            # localization loss
            loss_xy += F.mse_loss(outputs[i, ids, 0:2], gts[i, ids, 0:2], reduction='sum')
            loss_wh += F.mse_loss(outputs[i, ids, 2:4].sqrt(), gts[i, ids, 2:4].sqrt(), reduction='sum')

            # confidence loss
            max_iou_i = calc_iou(outputs[i, :, 0:4], gts[i, ids, 0:4]).max(axis=1).values
            loss_obj += F.mse_loss(max_iou_i[ids], gts[i, ids, 4], reduction='sum')
            loss_noobj += F.mse_loss(max_iou_i[non_ids], gts[i, non_ids, 4], reduction='sum')

            # class loss
            loss_c += F.cross_entropy(outputs[i, ids, 5:], gts[i, ids, 5].long(), reduction='sum')

        # sum up
        loss = 1/b * (l_coord * (loss_xy + loss_wh) + l_obj * loss_obj + l_noobj * loss_noobj + l_class * loss_c)
        return loss

    def get_paramaters(self):
        return self.parameters()


if __name__ == '__main__':
    import torch
    model = YoloV2()
    print(model)
    x = torch.rand((2, 3, 416, 416))
    print(model(x))
