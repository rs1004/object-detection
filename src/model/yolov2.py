import torch.nn as nn
from model import Conv2dBNLeaky, Route, Reorg


class YoloV2(nn.Module):
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


if __name__ == '__main__':
    import torch
    model = YoloV2()
    print(model)
    x = torch.rand((2, 3, 416, 416))
    print(model(x))
