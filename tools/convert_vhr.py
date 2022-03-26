# -----------------------------------------------------
# Context Aggregation Network
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (csyeliu at comp.polyu.edu.hk)
# -----------------------------------------------------

import argparse
import random
import re

import nncore
from nncore.engine import set_random_seed

from datasets import VHRDataset


def convert_anno(img_ids, split, img_dir, ann_dir, out_dir):
    (img_ids := img_ids[split]).sort()

    annos, img_id, ann_id = dict(
        images=[],
        annotations=[],
        categories=[
            dict(id=i + 1, name=name)
            for i, name in enumerate(VHRDataset.CLASSES)
        ]), 0, 0

    prog_bar = nncore.ProgressBar(num_tasks=len(img_ids))
    for id in img_ids:
        img_name = '{:03d}.jpg'.format(id)
        ann_name = '{:03d}.txt'.format(id)

        img_file = nncore.join(img_dir, img_name)
        ann_file = nncore.join(ann_dir, ann_name)

        img = nncore.imread(img_file)
        annos['images'].append(
            dict(
                id=img_id,
                file_name=img_name,
                width=img.shape[1],
                height=img.shape[0]))

        with open(ann_file, 'r') as f:
            for line in f:
                if line == '\n':
                    continue
                tokens = re.findall(r'\d+', line)
                x1, y1, x2, y2, cat_id = map(int, tokens)
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                annos['annotations'].append(
                    dict(
                        id=ann_id,
                        image_id=img_id,
                        category_id=cat_id,
                        iscrowd=0,
                        bbox=[x, y, w, h],
                        area=w * h))
                ann_id += 1

        img_id += 1
        prog_bar.update()

    out_file = nncore.join(out_dir, f'instances_{split}.json')
    nncore.dump(annos, out_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='data path', default='data/vhr')
    parser.add_argument(
        '--ratio',
        help='the ratio of training split',
        type=float,
        default=0.75)
    parser.add_argument('--out', help='output path')
    parser.add_argument(
        '--seed', help='random seed', type=int, default=14394022)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    out_dir = args.out or nncore.join(args.data_path, 'annotations')
    nncore.mkdir(out_dir)

    seed = set_random_seed(args.seed, sync=True)
    print(f'Using random seed: {seed}')

    img_dir = nncore.join(args.data_path, 'positive image set')
    ann_dir = nncore.join(args.data_path, 'ground truth')

    img_ids = dict()
    img_ids['train'] = random.sample(range(1, 651), int(650 * args.ratio))
    img_ids['val'] = [i for i in range(1, 651) if i not in img_ids['train']]

    for split in ['train', 'val']:
        print(f'Converting annotations of *{split}* split')
        convert_anno(img_ids, split, img_dir, ann_dir, out_dir)


if __name__ == '__main__':
    main()
