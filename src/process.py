import argparse
import torch
import json
import re
from pathlib import Path
from model.yolov2 import YoloV2
from dataset.dataloader import DataLoader
from function.train import Train
from function.inference import Inference
from function.evaluate import Evaluate


def get_weights_path(weights_dir, key, size):
    return max(weights_dir.glob(f'{key}-{size}-*.pt')).resolve().as_posix()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['train', 'evaluate', 'inference'], default='train')
    parser.add_argument('key', default='yolov2-voc')
    args = parser.parse_args()

    model_name, _ = args.key.split('-')

    # load config
    with open(Path(__file__).parent / 'config' / 'base_config.json', 'r') as f:
        cfg = json.load(f)
    cfg = {**cfg['common'], **cfg[args.task]}

    with open(Path(__file__).parent / 'config' / args.key / 'config.json', 'r') as f:
        cfg.update(json.load(f))

    cfg['data_dir'] = Path(cfg['root_dir']) / cfg['data_dir']
    cfg['weights_dir'] = Path(cfg['root_dir']) / cfg['weights_dir']
    cfg['log_dir'] = Path(cfg['root_dir']) / cfg['log_dir'] / args.key
    cfg['result_dir'] = Path(cfg['root_dir']) / cfg['result_dir'] / args.key

    # prepare
    dataloader = DataLoader(
        model_name=model_name,
        is_train=args.task == 'train',
        **cfg
    )

    if 'yolov2' in args.key:
        cfg['anchors'] = torch.as_tensor(cfg['anchors']) * cfg['input_size'] / 32
        model = YoloV2(anchors=cfg['anchors'], num_classes=len(dataloader.dataset.labels))
        weights_path = get_weights_path(
            weights_dir=cfg['weights_dir'],
            key=args.key,
            size=cfg['input_size']
        )
        model.load_state_dict(torch.load(weights_path))
    else:
        raise NotImplementedError(f'{args.key} is not expected')

    # run process
    if args.task == 'train':
        cfg['key'] = f'{args.key}-{cfg["input_size"]}'
        cfg['last_epoch'] = int(re.findall(rf'{cfg["key"]}-(.+?).pt', weights_path)[0])
        process = Train(model=model, dataloader=dataloader, **cfg)

    elif args.task == 'evaluate':
        process = Evaluate(model=model, dataloader=dataloader, **cfg)

    elif args.task == 'inference':
        process = Inference(model=model, dataloader=dataloader, **cfg)

    process.run()
