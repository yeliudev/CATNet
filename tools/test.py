# -----------------------------------------------------
# Context Aggregation Network
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (csyeliu at comp.polyu.edu.hk)
# -----------------------------------------------------

import argparse

import nncore
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from nncore.engine import comm, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--format', help='format results', action='store_true')
    parser.add_argument('--launcher', help='job launcher')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = nncore.Config.from_file(args.config)

    launcher = comm.init_dist(launcher=args.launcher, **cfg.dist_params)
    distributed = launcher is not None

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, 1, cfg.data.workers_per_gpu, dist=distributed, shuffle=False)

    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']

    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        results = multi_gpu_test(model, data_loader)
    else:
        model = MMDataParallel(model, device_ids=[0])
        results = single_gpu_test(model, data_loader)

    if comm.is_main_process():
        if args.format:
            prefix = nncore.pure_name(args.config)
            dataset.format_results(results, jsonfile_prefix=prefix)
        else:
            for key in ('start', 'interval', 'by_epoch', 'save_best', 'rule'):
                cfg.evaluation.pop(key, None)
            metrics = dataset.evaluate(results, **cfg.evaluation)
            print(metrics)


if __name__ == '__main__':
    main()
