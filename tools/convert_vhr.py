# -----------------------------------------------------
# Context Aggregation Network
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (coco.ye.liu at connect.polyu.hk)
# -----------------------------------------------------

import argparse
import random

import nncore
from nncore.engine import set_random_seed

CLASSES = [
    'airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court',
    'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle'
]


def convert_anno(img_ids, split, src_ann, trg_dir):
    trg_ann = dict(
        images=[
            img_ann for img_ann in src_ann['images']
            if img_ann['id'] in img_ids[split]
        ],
        annotations=[
            ins_ann for ins_ann in src_ann['annotations']
            if ins_ann['image_id'] in img_ids[split]
        ],
        categories=[
            dict(id=i + 1, name=name) for i, name in enumerate(CLASSES)
        ])

    out_file = nncore.join(trg_dir, f'instances_{split}.json')
    nncore.dump(trg_ann, out_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='data path', default='data/vhr')
    parser.add_argument(
        '--ratio',
        help='the ratio of training split',
        type=float,
        default=0.75)
    parser.add_argument(
        '--seed', help='random seed', type=int, default=14394022)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    seed = set_random_seed(args.seed, sync=True)
    print(f'Using random seed: {seed}')

    ann_dir = nncore.join(args.data_path, 'annotations.json')
    src_ann = nncore.load(ann_dir)

    trg_dir = nncore.join(args.data_path, 'annotations')
    img_dir = nncore.join(args.data_path, 'positive image set')
    out_dir = nncore.join(args.data_path, 'images')

    img_ids = dict()
    img_ids['train'] = random.sample(range(650), int(650 * args.ratio))
    img_ids['test'] = [i for i in range(650) if i not in img_ids['train']]

    for split in ['train', 'test']:
        print(f'Converting annotations of *{split}* split')
        convert_anno(img_ids, split, src_ann, trg_dir)

    nncore.remove(ann_dir)
    nncore.mv(img_dir, out_dir)


if __name__ == '__main__':
    main()
