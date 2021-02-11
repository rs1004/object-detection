from torchvision.ops import box_convert, box_iou


def calc_iou(t, s):
    if len(t.shape) == 1:
        t = t.unsqueeze(0)
    if len(s.shape) == 1:
        s = s.unsqueeze(0)

    t = box_convert(t, in_fmt='cxcywh', out_fmt='xyxy')
    s = box_convert(s, in_fmt='cxcywh', out_fmt='xyxy')

    iou = box_iou(t, s)
    return iou
