import torch
import torch.nn as nn


class Conv2dBNLeaky(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding=0):
        super(Conv2dBNLeaky, self).__init__()

        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.activation(x)
        return out


class Route(nn.Module):
    def __init__(self, layers):
        super(Route, self).__init__()
        self.layers = layers

    def forward(self, xs):
        return torch.cat(xs, axis=1)


class Reorg(nn.Module):
    def __init__(self, stride):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        b, c, h, w = x.data.size()
        ws = self.stride
        hs = self.stride
        x = x.view(b, c, int(h / hs), hs, int(w / ws), ws).transpose(3, 4).contiguous()
        x = x.view(b, c, int(h / hs * w / ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(b, c, hs * ws, int(h / hs), int(w / ws)).transpose(1, 2).contiguous()
        x = x.view(b, hs * ws * c, int(h / hs), int(w / ws))
        return x


class Region(nn.Module):
    def __init__(self, anchors, num_classes):
        super(Region, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes

    def forward(self, x):
        b, _, h, w = x.shape

        # (b, c, h, w) => (b, h * w * num_anchors, coord + num_classes)
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h * w * len(self.anchors), 5 + self.num_classes)

        # activate
        x[:, :, 0:2] = torch.sigmoid(x[:, :, 0:2])
        x[:, :, 2:4] = torch.exp(x[:, :, 2:4])
        x[:, :, 4:5] = torch.sigmoid(x[:, :, 4:5])
        x[:, :, 5:] = torch.softmax(x[:, :, 5:], dim=2)

        # restore
        cx, cy = torch.meshgrid(torch.arange(w), torch.arange(h))
        cx = cx.t().contiguous().view(-1, 1)  # transpose because anchors to be organized in H x W order
        cy = cy.t().contiguous().view(-1, 1)

        centers = torch.cat([cx, cy], axis=1).float()
        anchors = torch.as_tensor(self.anchors)

        all_anchors = torch.cat([
            centers.view(-1, 1, 2).expand(-1, len(self.anchors), 2),
            anchors.view(1, -1, 2).expand(h * w, -1, 2)
        ], axis=2).view(-1, 4)  # (h * w * num_anchors, [cx, cy, w, h])

        x[:, :, 0:2] += all_anchors[:, 0:2]
        x[:, :, 2:4] *= all_anchors[:, 2:4]

        return x
