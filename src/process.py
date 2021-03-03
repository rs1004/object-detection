import argparse
import torch
import json
import re
from pathlib import Path
from model.yolov2 import YoloV2
from dataset import DataLoader
from function.train import Train
from function.inference import Inference
from function.evaluate import Evaluate


def get_weights_path(key, cfg):
    weights_dir = Path(cfg['common']['weights_dir'])
    return max(weights_dir.glob(f'{key}-*.pt')).resolve().as_posix()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['train', 'evaluate', 'inference'], default='train')
    parser.add_argument('key', default='yolov2-voc')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    args = parser.parse_args()

    # load config
    with open(Path(__file__).parent / 'config.json', 'r') as f:
        cfg = json.load(f)[args.key]

    cfg['common']['log_dir'] += f'/{args.key}'
    cfg['common']['result_dir'] += f'/{args.key}'

    # prepare
    dataloader = DataLoader(batch_size=args.batch_size, key=args.key, is_train=args.task == 'train', **{**cfg['common'], **cfg['dataloader']})

    if 'yolov2' in args.key:
        model = YoloV2(anchors=cfg['dataloader']['anchors'], num_classes=len(dataloader.dataset.labels))
        weights_path = get_weights_path(key=args.key, cfg=cfg)
        model.load_state_dict(torch.load(weights_path))
    else:
        raise NotImplementedError(f'{args.key} is not expected')

    # run process
    if args.task == 'train':
        cfg = {**cfg['train'], **cfg['common']}
        cfg['key'] = args.key
        cfg['last_epoch'] = int(re.findall(rf'{args.key}-(.+?).pt', weights_path)[0])
        process = Train(model=model, dataloader=dataloader, **cfg)

    elif args.task == 'evaluate':
        cfg = {**cfg['evaluate'], **cfg['common']}
        process = Evaluate(model=model, dataloader=dataloader, **cfg)

    elif args.task == 'inference':
        cfg = {**cfg['inference'], **cfg['common']}
        process = Inference(model=model, dataloader=dataloader, **cfg)

    process.run()
