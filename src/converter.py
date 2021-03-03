import torch
import torch.nn as nn
import numpy as np
from functools import partial
from model import Conv2dBNLeaky


class Darknet2PT:
    def __init__(self, weights_path, model, save_path, is_transfer=False):
        self.weights_path = weights_path
        self.model = model
        self.save_path = save_path
        self.is_transfer = is_transfer

    def run(self):
        with open(self.weights_path, 'rb') as wf:
            get_parameter = partial(self._get_parameter, wf=wf)

            # read head
            wf.read(16)

            # read body
            for m in self.model.features:
                if isinstance(m, Conv2dBNLeaky):
                    m.bn.bias = get_parameter((m.bn.num_features, ))
                    m.bn.weight = get_parameter((m.bn.num_features, ))
                    m.bn.running_mean = get_parameter((m.bn.num_features, ), return_tensor=True)
                    m.bn.running_var = get_parameter((m.bn.num_features, ), return_tensor=True)
                    m.conv.weight = get_parameter((m.conv.out_channels, m.conv.in_channels, m.conv.kernel_size[0], m.conv.kernel_size[0]))

            if not self.is_transfer:
                m = self.model.detector
                self.model.detector.bias = get_parameter((m.out_channels, ))
                self.model.detector.weight = get_parameter((m.out_channels, m.in_channels, m.kernel_size[0], m.kernel_size[0]))

                remain = len(wf.read())
                assert remain == 0, f'weight buffer remained: {remain}'

        torch.save(self.model.state_dict(), self.save_path)

    def _get_parameter(self, shape, wf, return_tensor=False):
        param = np.ndarray(
            shape=shape,
            dtype='float32',
            buffer=wf.read(np.product(shape) * 4))
        t = torch.as_tensor(param.copy())
        if return_tensor:
            return t
        else:
            return nn.Parameter(t)


if __name__ == '__main__':
    from model.yolov2 import YoloV2

    # ------ set params ------ #
    anchors = torch.tensor([
        [1.2459042,  2.38299313],
        [2.16345581, 3.87256071],
        [0.60744533, 1.11842341],
        [3.4044071, 5.84595754],
        [5.98817284, 9.85411166]
    ])
    num_classes = 3
    is_transfer = True
    # ------------------------ #

    model = YoloV2(anchors, num_classes)

    converter = Darknet2PT('./assets/weights/yolov2.weights', model, './assets/weights/yolov2-maskw-00000.pt', is_transfer)
    converter.run()

    model = YoloV2(anchors, num_classes)
    model.load_state_dict(torch.load('./assets/weights/yolov2-maskw-00000.pt'))

    print(model.features[0].conv.weight)
