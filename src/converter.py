import torch
import torch.nn as nn
import numpy as np
from functools import partial
from model import Conv2dBNLeaky


class Darknet2PT:
    def __init__(self, weights_path, model, save_path):
        self.weights_path = weights_path
        self.model = model
        self.save_path = save_path

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

    model = YoloV2()

    converter = Darknet2PT('./assets/weights/yolo-voc.weights', model, './assets/weights/yolov2-voc-00000.pt')
    converter.run()

    model = YoloV2()
    model.load_state_dict(torch.load('./assets/weights/yolov2-voc-00000.pt'))

    print(model.features[0].conv.weight)
