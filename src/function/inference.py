import torch
import seaborn as sns
from torchvision.ops import nms
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path, PosixPath
from tqdm import tqdm


class Inference:
    def __init__(self, model, dataloader, **cfg):
        self.model = model
        self.dataloader = dataloader

        self.__dict__.update(cfg)

    def run(self):
        colors = [tuple([int(i * 255) for i in c]) for c in sns.color_palette('hls', n_colors=len(self.dataloader.dataset.labels))]
        save_dir = Path(self.result_dir) / 'inference'
        save_dir.mkdir(exist_ok=True, parents=True)
        num_output = min(self.num_output, len(self.dataloader) * self.batch_size)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        count = 0
        with torch.no_grad():
            for images, gts, masks in tqdm(self.dataloader, total=len(self.dataloader)):
                # to GPU device
                images = images.to(device)

                # forward
                outputs = self.model(images)

                # draw bbox and save image
                images = images.to('cpu')
                outputs = outputs.to('cpu')

                outputs[:, :, 0:4] = outputs[:, :, 0:4] * self.restore_scale
                gts[:, :, 0:4] = gts[:, :, 0:4] * self.restore_scale

                saved = []
                for i, image, output, gt, mask in zip(range(count, len(images) + count), images, outputs, gts, masks):
                    saved.append(self._draw_and_save(i, image, output, gt[mask == 1], colors, save_dir, self.draw_gt_box))

                count += sum(saved)

                if count >= num_output:
                    break

        print('Finished Inference')

    def _draw_and_save(self, i: int, image: torch.Tensor, output: torch.Tensor,
                       gt: torch.Tensor, colors: list, save_dir: PosixPath, draw_gt_box: bool = False) -> int:
        """ draw bbox on image and save image

        Args:
            i (int): image number to save
            image (torch.Tensor): image tensor
            output (torch.Tensor): [n, coord + 1 + num_classes] (coord: xyxy, normalized)
            gt (torch.Tensor): [m, coord + class_id + 1] (coord: xyxy, normalized)
            colors (list): RGB tuple list
            save_dir (PosixPath): image save directory
            draw_gt_box (bool, optional): if True, gt bbox will be drawn. Defaults to False.

        Returns:
            int: 1 (no special meaning)
        """
        image = Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype('uint8'))

        output = output[output[:, 4].sort(descending=True).indices]

        boxes = output[:, 0:4]
        class_ids = output[:, 5:].max(dim=-1).indices
        confs = output[:, 4]

        # leave valid bboxes
        boxes = boxes[confs > self.conf_thresh]
        class_ids = class_ids[confs > self.conf_thresh]
        confs = confs[confs > self.conf_thresh]

        ids = nms(boxes, confs, iou_threshold=self.nms_thresh)
        boxes = boxes[ids]
        class_ids = class_ids[ids]
        confs = confs[ids]

        # draw & save
        draw = ImageDraw.Draw(image)

        if draw_gt_box:
            gt_color = (255, 255, 255)
            space = 20
            for gt_box in gt[:, 0:4]:
                xmin, ymin, xmax, ymax = gt_box.data.numpy()
                for x in range(int(xmin), int(xmax) + 1, space):
                    draw.line((x, ymin, x + 5, ymin), fill=gt_color)
                    draw.line((x, ymax, x + 5, ymax), fill=gt_color)

                for y in range(int(ymin), int(ymax) + 1, space):
                    draw.line((xmin, y, xmin, y + 5), fill=gt_color)
                    draw.line((xmax, y, xmax, y + 5), fill=gt_color)

        font = ImageFont.truetype((Path(__file__).parent / self.font_name).as_posix(), self.font_size)
        for box, conf, class_id in zip(boxes, confs, class_ids):
            xmin, ymin, xmax, ymax = box.data.numpy()
            color = colors[class_id.data]
            text = f'{self.dataloader.dataset.labels[class_id.data]}: {round(float(conf.data), 3)}'

            draw.rectangle((xmin, ymin, xmax, ymax), outline=color)
            draw.text((xmin, ymin), text, fill=color, font=font)

        image.save(save_dir / f'{i+1:05}.png')

        return 1
