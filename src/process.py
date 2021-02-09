import argparse
import torch
from pathlib import Path
from model.yolov2 import YoloV2
from dataset import DataLoader
from function import Train, Evaluate, Inference


def get_weights_path(key):
    weights_dir = Path('./assets/weights')
    return max(weights_dir.glob(f'{key}-*.pt')).resolve().as_posix()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['train', 'evaluate', 'inference'], default='train')
    parser.add_argument('key', default='yolov2-voc')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--transfer_learning', '-t', action='store_true')
    args = parser.parse_args()

    # prepare
    if args.key == 'yolov2-voc':
        model = YoloV2()
        weights_path = get_weights_path(key=args.key)
        model.load_state_dict(torch.load(weights_path))

    dataloader = DataLoader(batch_size=args.batch_size, key=args.key, is_train=args.task == 'train')

    # run process
    if args.task == 'train':
        process = Train(model=model, dataloader=dataloader)
    elif args.task == 'evaluate':
        process = Evaluate(model=model, dataloader=dataloader)
    elif args.task == 'inference':
        process = Inference(model=model, dataloader=dataloader)

    process.run()
