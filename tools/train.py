# -----------------------------------------------------
# Context Aggregation Network
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (csyeliu at comp.polyu.edu.hk)
# -----------------------------------------------------

import argparse

import mmdet
import nncore
from mmcv.utils.logging import logger_initialized
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from nncore.engine import comm, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--seed', help='random seed', type=int)
    parser.add_argument('--launcher', help='job launcher')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = nncore.Config.from_file(args.config)

    launcher = comm.init_dist(launcher=args.launcher, **cfg.dist_params)

    if comm.is_main_process():
        work_dir = nncore.mkdir(
            nncore.join('work_dirs', nncore.pure_name(args.config)),
            modify_path=True)
    else:
        work_dir = None

    cfg.resume_from = args.checkpoint
    cfg.work_dir = comm.broadcast(data=work_dir)
    cfg.seed = set_random_seed(args.seed)
    cfg.gpu_ids = range(comm.get_world_size())

    timestamp = nncore.get_timestamp()
    log_file = nncore.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger = nncore.get_logger(logger_or_name='mmdet', log_file=log_file)
    logger_initialized['mmdet'] = True

    logger.info(f'Environment info:\n{nncore.collect_env_info()}')
    logger.info(f'Elastic launcher: {launcher}')
    logger.info(f'Config: {cfg.text}')

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        dataset = cfg.data.val.copy()
        dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(dataset))

    model = build_detector(cfg.model)
    model.init_weights()

    logger.info(f'Model architecture:\n{model}')

    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            nncore_version=nncore.__version__,
            mmdet_version=mmdet.__version__,
            CLASSES=datasets[0].CLASSES)

    train_detector(
        model,
        datasets,
        cfg,
        distributed=launcher is not None,
        validate=True,
        timestamp=timestamp)


if __name__ == '__main__':
    main()
